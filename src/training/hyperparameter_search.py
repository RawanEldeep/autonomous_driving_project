"""
Enhancement 5: Systematic Hyperparameter Optimization

Runs a random search over the PPO hyperparameter space defined in the proposal.
Designed to work within Google Colab constraints (single GPU, limited RAM).

Search space:
    learning_rate : [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
    batch_size    : [32, 64, 128]
    n_steps       : [256, 512, 1024]
    ent_coef      : [0.0, 0.01, 0.05]
    gamma         : [0.95, 0.99, 0.995]

Each configuration is trained for `total_timesteps` steps (default 50 000,
suitable for Colab).  Results are written incrementally to CSV and JSON so
they are not lost if the session crashes.

Usage:
    from src.training.hyperparameter_search import random_search

    best_config, all_results = random_search(
        env_fn=lambda: my_env_factory(),
        n_configs=12,
        total_timesteps=50_000,
        log_dir="hp_search_results",

    )
"""

import csv
import json
import os
import random
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv


# =============================================================================
# Hyperparameter search space
# =============================================================================

SEARCH_SPACE: Dict[str, list] = {
    "learning_rate": [1e-4, 5e-4, 1e-3, 2e-3, 5e-3],
    "batch_size":    [32, 64, 128],
    "n_steps":       [256, 512, 1024],
    "ent_coef":      [0.0, 0.01, 0.05],
    "gamma":         [0.95, 0.99, 0.995],
}

_CSV_FIELDS = list(SEARCH_SPACE.keys()) + ["mean_reward", "std_reward", "timestamp"]


# =============================================================================
# Callback: collects episode rewards during training
# =============================================================================

class _RewardCollector(BaseCallback):
    """Lightweight callback that records every finished episode's reward."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: List[float] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
        return True

    def mean_reward(self, last_n: int = 10) -> float:
        if not self.episode_rewards:
            return float("-inf")
        return float(np.mean(self.episode_rewards[-last_n:]))

    def std_reward(self, last_n: int = 10) -> float:
        if len(self.episode_rewards) < 2:
            return 0.0
        return float(np.std(self.episode_rewards[-last_n:]))


# =============================================================================
# Random search
# =============================================================================

def random_search(
    env_fn: Callable,
    n_configs: int = 12,
    total_timesteps: int = 50_000,
    log_dir: str = "hp_search_results",
    seed: int = 42,
    policy_kwargs: Optional[dict] = None,
    model_class=PPO,
    extra_model_kwargs: Optional[dict] = None,
) -> Tuple[Optional[dict], List[dict]]:
    """
    Random hyperparameter search for PPO.

    Parameters
    ----------
    env_fn : callable
        Zero-argument factory that returns a gymnasium environment.
        Called once per configuration inside a DummyVecEnv.
    n_configs : int
        Number of random configurations to try.  12 fits comfortably on Colab.
    total_timesteps : int
        Training budget per configuration.
    log_dir : str
        Directory for CSV, JSON, and TensorBoard logs.
    seed : int
        Controls which random configs are sampled.
    policy_kwargs : dict or None
        Passed directly to PPO as ``policy_kwargs`` (e.g. custom extractor).
    model_class : class
        PPO subclass to use (e.g. AttentionRegularizedPPO).
    extra_model_kwargs : dict or None
        Additional keyword arguments forwarded to ``model_class.__init__``.

    Returns
    -------
    best_config : dict or None
        Hyperparameter dict with the highest mean reward.
    all_results : list of dict
        One entry per configuration, including scores and timestamp.
    """
    os.makedirs(log_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)

    extra_model_kwargs = extra_model_kwargs or {}

    # Sample n_configs distinct random configurations
    configs = [{k: random.choice(v) for k, v in SEARCH_SPACE.items()}
               for _ in range(n_configs)]

    all_results: List[dict] = []
    best_score = float("-inf")
    best_config: Optional[dict] = None

    csv_path  = os.path.join(log_dir, "hp_results.csv")
    json_path = os.path.join(log_dir, "hp_results.json")
    tb_dir    = os.path.join(log_dir, "tensorboard")

    # Write CSV header
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=_CSV_FIELDS).writeheader()

    print(f"\n{'='*60}")
    print(f"Hyperparameter Search: {n_configs} configs × {total_timesteps} steps")
    print(f"Results → {log_dir}/")
    print(f"{'='*60}\n")

    for idx, config in enumerate(configs):
        print(f"[{idx + 1}/{n_configs}] {config}")

        try:
            env = make_vec_env(env_fn, n_envs=1, seed=seed, vec_env_cls=DummyVecEnv)

            model = model_class(
                policy="MlpPolicy",
                env=env,
                learning_rate=config["learning_rate"],
                batch_size=config["batch_size"],
                # n_steps must be >= batch_size; guard against that
                n_steps=max(config["n_steps"], config["batch_size"]),
                ent_coef=config["ent_coef"],
                gamma=config["gamma"],
                verbose=0,
                tensorboard_log=tb_dir,
                **({"policy_kwargs": policy_kwargs} if policy_kwargs else {}),
                **extra_model_kwargs,
            )

            collector = _RewardCollector()
            model.learn(total_timesteps=total_timesteps, callback=collector)

            score = collector.mean_reward(last_n=10)
            std   = collector.std_reward(last_n=10)
            print(f"  → mean_reward={score:.2f}  std={std:.2f}")

            result = {
                **config,
                "mean_reward": round(score, 4),
                "std_reward":  round(std, 4),
                "timestamp":   datetime.now().isoformat(),
            }
            all_results.append(result)

            # Track best
            if score > best_score:
                best_score = score
                best_config = config.copy()
                model.save(os.path.join(log_dir, "best_model"))
                print(f"  ** New best: score={best_score:.2f}")

            # Incremental CSV write (safe on Colab crashes)
            with open(csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=_CSV_FIELDS).writerow(result)

            env.close()

        except Exception as exc:
            print(f"  Config {idx + 1} failed: {exc}")
            all_results.append({
                **config,
                "mean_reward": float("nan"),
                "std_reward":  float("nan"),
                "timestamp":   datetime.now().isoformat(),
            })

    # Final JSON dump
    summary = {
        "best_config": best_config,
        "best_score":  best_score,
        "n_configs":   n_configs,
        "total_timesteps_per_config": total_timesteps,
        "all_results": all_results,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSearch complete.")
    print(f"Best score : {best_score:.2f}")
    print(f"Best config: {json.dumps(best_config, indent=2)}")
    print(f"Full results → {json_path}\n")

    return best_config, all_results


# =============================================================================
# Utility: print ranked results
# =============================================================================

def print_ranked_results(all_results: List[dict], top_n: int = 5):
    """Print the top-N configurations sorted by mean_reward."""
    valid = [r for r in all_results if not np.isnan(r.get("mean_reward", float("nan")))]
    ranked = sorted(valid, key=lambda r: r["mean_reward"], reverse=True)

    print(f"\nTop-{min(top_n, len(ranked))} configurations:")
    print("-" * 70)
    for i, r in enumerate(ranked[:top_n], 1):
        hp = {k: r[k] for k in SEARCH_SPACE}
        print(
            f"  #{i:2d}  reward={r['mean_reward']:.2f} ± {r.get('std_reward', 0):.2f}"
            f"  {hp}"
        )
    print("-" * 70)
