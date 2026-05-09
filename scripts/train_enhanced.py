import argparse
import os
import sys

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

import highway_env  # noqa: F401  — registers highway-v0

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.baseline_model import (
    CustomExtractor,
    env_kwargs,
)
from src.envs.custom_wrappers import (
    DynamicTrafficWrapper,
    EnhancedRewardWrapper,
    ObservationEnrichmentWrapper,
)
from src.models.custom_policy import AttentionRegularizedPPO
from src.evaluation.evaluation_metrics import (
    AggregatedMetrics,
    MetricsEvaluator,
    MetricsPlotter,
    compute_learning_efficiency,
)
from src.training.hyperparameter_search import random_search, print_ranked_results


ENHANCEMENT_CONFIG = {
    "enhanced_reward":        True,
    "dynamic_traffic":        True,
    "observation_enrichment": True,
    "attention_reg":          True,
    "hp_search":              False,
}

TRAIN_CONFIG = {
    "n_cpu":            4,
    "total_timesteps":  200_000,
    "eval_freq":        10_000,
    "n_eval_episodes":  10,
    "checkpoint_freq":  50_000,
    "seed":             42,
    "use_gpu":          True,
    "log_dir":          "enhanced_ppo_logs",
    "baseline_save":    "enhanced_ppo_logs/baseline_model",
    "enhanced_save":    "enhanced_ppo_logs/enhanced_model",
}

_BASELINE_ATTN_KWARGS = dict(
    in_size=7 * 15,
    embedding_layer_kwargs={"in_size": 7, "layer_sizes": [64, 64], "reshape": False},
    attention_layer_kwargs={"feature_size": 64, "heads": 2},
)

_ENHANCED_ATTN_KWARGS = dict(
    in_size=10 * 15,
    embedding_layer_kwargs={"in_size": 10, "layer_sizes": [64, 64], "reshape": False},
    attention_layer_kwargs={"feature_size": 64, "heads": 2},
)


def make_baseline_env(**kwargs):
    env = gym.make(
        kwargs.get("id", "highway-v0"),
        config=kwargs.get("config", env_kwargs["config"]),
    )
    env.reset()
    return env


def make_enhanced_env(enhancements=None, **kwargs):
    if enhancements is None:
        enhancements = ENHANCEMENT_CONFIG

    obs_cfg = kwargs.get("config", env_kwargs["config"]).get("observation", {})
    vehicles_count = obs_cfg.get("vehicles_count", 10)

    env = gym.make(
        kwargs.get("id", "highway-v0"),
        config=kwargs.get("config", env_kwargs["config"]),
    )
    env.reset()

    if enhancements.get("dynamic_traffic"):
        env = DynamicTrafficWrapper(
            env,
            enable_dynamic_density=True,
            enable_dynamic_events=True,
            event_probability=0.02,
        )

    if enhancements.get("observation_enrichment"):
        env = ObservationEnrichmentWrapper(env, vehicles_count=vehicles_count)

    if enhancements.get("enhanced_reward"):
        env = EnhancedRewardWrapper(
            env,
            ttc_threshold=3.0,
            ttc_penalty_weight=0.3,
            smooth_driving_weight=0.1,
            lane_centering_weight=0.1,
        )

    return env


class _RewardLogger(BaseCallback):

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                r = float(info["episode"]["r"])
                self.episode_rewards.append(r)
                self.logger.record(
                    "rollout/ep_rew_mean_20",
                    np.mean(self.episode_rewards[-20:]),
                )
            for key in ("ttc_penalty", "smooth_reward", "lane_centering_reward"):
                if key in info:
                    self.logger.record(f"train/{key}", float(info[key]))
        return True


def train_baseline(config: dict = TRAIN_CONFIG) -> PPO:
    print("\n" + "=" * 55)
    print("  Training BASELINE model")
    print("=" * 55)

    os.makedirs(config["log_dir"], exist_ok=True)
    device = _device(config)
    print(f"  Device: {device}")

    policy_kwargs = dict(
        features_extractor_class=CustomExtractor,
        features_extractor_kwargs=_BASELINE_ATTN_KWARGS,
    )

    env = make_vec_env(
        make_baseline_env,
        n_envs=config["n_cpu"],
        seed=config["seed"],
        vec_env_cls=DummyVecEnv,
        env_kwargs=env_kwargs,
    )

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=max(512 // config["n_cpu"], 64),
        batch_size=64,
        learning_rate=2e-3,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log=os.path.join(config["log_dir"], "baseline_tb"),
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[
            _RewardLogger(),
            CheckpointCallback(
                save_freq=config["checkpoint_freq"],
                save_path=os.path.join(config["log_dir"], "baseline_ckpts"),
            ),
        ],
    )

    model.save(config["baseline_save"])
    env.close()
    print(f"  Baseline saved → {config['baseline_save']}.zip")
    return model


def train_enhanced(
    config: dict = TRAIN_CONFIG,
    enhancements: dict = ENHANCEMENT_CONFIG,
) -> PPO:
    print("\n" + "=" * 55)
    print("  Training ENHANCED model")
    active = [k for k, v in enhancements.items() if v]
    print(f"  Active enhancements: {active}")
    print("=" * 55)

    os.makedirs(config["log_dir"], exist_ok=True)
    device = _device(config)
    print(f"  Device: {device}")

    if enhancements.get("hp_search"):
        print("\n  Running hyperparameter search (Enhancement 5)…")
        best_hp, all_hp_results = random_search(
            env_fn=lambda: make_enhanced_env(enhancements=enhancements, **env_kwargs),
            n_configs=12,
            total_timesteps=30_000,
            log_dir=os.path.join(config["log_dir"], "hp_search"),
            seed=config["seed"],
        )
        print_ranked_results(all_hp_results, top_n=5)
        lr         = best_hp.get("learning_rate", 2e-3)
        batch_size = best_hp.get("batch_size", 64)
        n_steps    = best_hp.get("n_steps", 512)
        ent_coef   = best_hp.get("ent_coef", 0.01)
        gamma      = best_hp.get("gamma", 0.99)
    else:
        lr, batch_size, n_steps = 2e-3, 64, 512
        ent_coef, gamma = 0.01, 0.99

    attn_kwargs = (
        _ENHANCED_ATTN_KWARGS
        if enhancements.get("observation_enrichment")
        else _BASELINE_ATTN_KWARGS
    )
    policy_kwargs = dict(
        features_extractor_class=CustomExtractor,
        features_extractor_kwargs=attn_kwargs,
    )

    env = make_vec_env(
        make_enhanced_env,
        n_envs=config["n_cpu"],
        seed=config["seed"],
        vec_env_cls=DummyVecEnv,
        env_kwargs={"enhancements": enhancements, **env_kwargs},
    )

    ModelClass = (
        AttentionRegularizedPPO if enhancements.get("attention_reg") else PPO
    )

    model_kwargs = dict(
        policy="MlpPolicy",
        env=env,
        n_steps=max(n_steps // config["n_cpu"], batch_size),
        batch_size=batch_size,
        learning_rate=lr,
        ent_coef=ent_coef,
        gamma=gamma,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        tensorboard_log=os.path.join(config["log_dir"], "enhanced_tb"),
    )

    if enhancements.get("attention_reg"):
        model_kwargs.update(
            lambda_high=0.01,
            lambda_low=0.01,
            n_observed_vehicles=10,
            enable_attention_reg=True,
        )

    model = ModelClass(**model_kwargs)

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[
            _RewardLogger(),
            CheckpointCallback(
                save_freq=config["checkpoint_freq"],
                save_path=os.path.join(config["log_dir"], "enhanced_ckpts"),
            ),
        ],
    )

    model.save(config["enhanced_save"])
    env.close()
    print(f"  Enhanced saved → {config['enhanced_save']}.zip")
    return model


def evaluate_and_compare(
    config: dict = TRAIN_CONFIG,
    enhancements: dict = ENHANCEMENT_CONFIG,
    n_eval_episodes: int = 100,
) -> dict:
    print("\n" + "=" * 55)
    print("  Evaluation")
    print("=" * 55)

    results_dir = os.path.join(config["log_dir"], "results")
    os.makedirs(results_dir, exist_ok=True)
    plotter = MetricsPlotter(save_dir=results_dir)

    outcomes = {}

    baseline_path = config["baseline_save"] + ".zip"
    baseline_agg, baseline_eps = None, None

    if os.path.exists(baseline_path):
        print(f"\n  Loading baseline from {baseline_path}")
        baseline_model = PPO.load(config["baseline_save"])
        baseline_env = make_baseline_env(**env_kwargs)

        evaluator = MetricsEvaluator(
            baseline_env, baseline_model, n_episodes=n_eval_episodes
        )
        baseline_agg = evaluator.evaluate()
        baseline_eps = evaluator.episode_metrics
        evaluator.save_metrics(
            baseline_agg, os.path.join(results_dir, "baseline_metrics.json")
        )
        outcomes["baseline"] = baseline_agg
        baseline_env.close()
    else:
        print(f"  Baseline model not found at {baseline_path} — skipping.")

    enhanced_path = config["enhanced_save"] + ".zip"
    enhanced_agg, enhanced_eps = None, None

    if os.path.exists(enhanced_path):
        print(f"\n  Loading enhanced from {enhanced_path}")
        ModelClass = (
            AttentionRegularizedPPO if enhancements.get("attention_reg") else PPO
        )
        enhanced_model = ModelClass.load(config["enhanced_save"])
        enhanced_env = make_enhanced_env(enhancements=enhancements, **env_kwargs)

        evaluator = MetricsEvaluator(
            enhanced_env, enhanced_model, n_episodes=n_eval_episodes
        )
        enhanced_agg = evaluator.evaluate()
        enhanced_eps = evaluator.episode_metrics
        evaluator.save_metrics(
            enhanced_agg, os.path.join(results_dir, "enhanced_metrics.json")
        )
        outcomes["enhanced"] = enhanced_agg
        enhanced_env.close()
    else:
        print(f"  Enhanced model not found at {enhanced_path} — skipping.")

    if baseline_agg and enhanced_agg:
        print("\n  Generating comparison plots…")
        plotter.plot_all(
            baseline_agg,
            enhanced_agg,
            baseline_episodes=baseline_eps,
            enhanced_episodes=enhanced_eps,
        )
    elif baseline_agg:
        plotter.print_summary_table(baseline_agg, baseline_agg)
    elif enhanced_agg:
        plotter.print_summary_table(enhanced_agg, enhanced_agg)

    return outcomes


def _device(config: dict) -> str:
    if config.get("use_gpu") and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Enhanced PPO training pipeline for autonomous highway driving"
    )
    parser.add_argument(
        "--mode",
        choices=["train_baseline", "train_enhanced", "evaluate", "full"],
        default="full",
        help=(
            "train_baseline  — train baseline only\n"
            "train_enhanced  — train enhanced only\n"
            "evaluate        — evaluate pre-trained models\n"
            "full            — train both then evaluate  (default)"
        ),
    )
    parser.add_argument("--timesteps", type=int, default=200_000,
                        help="Total training timesteps per model")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--n-cpu", type=int, default=4,
                        help="Number of parallel envs (use 1 on Colab free tier)")

    parser.add_argument("--no-reward",  dest="enhanced_reward",        action="store_false")
    parser.add_argument("--no-traffic", dest="dynamic_traffic",        action="store_false")
    parser.add_argument("--no-obs",     dest="observation_enrichment", action="store_false")
    parser.add_argument("--no-attn",    dest="attention_reg",          action="store_false")
    parser.add_argument("--hp-search",  dest="hp_search",              action="store_true")

    parser.set_defaults(
        enhanced_reward=True,
        dynamic_traffic=True,
        observation_enrichment=True,
        attention_reg=True,
        hp_search=False,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    TRAIN_CONFIG["total_timesteps"] = args.timesteps
    TRAIN_CONFIG["n_cpu"] = args.n_cpu

    enhancements = {
        "enhanced_reward":        args.enhanced_reward,
        "dynamic_traffic":        args.dynamic_traffic,
        "observation_enrichment": args.observation_enrichment,
        "attention_reg":          args.attention_reg,
        "hp_search":              args.hp_search,
    }

    if args.mode in ("train_baseline", "full"):
        train_baseline(TRAIN_CONFIG)

    if args.mode in ("train_enhanced", "full"):
        train_enhanced(TRAIN_CONFIG, enhancements)

    if args.mode in ("evaluate", "full"):
        evaluate_and_compare(
            config=TRAIN_CONFIG,
            enhancements=enhancements,
            n_eval_episodes=args.eval_episodes,
        )
