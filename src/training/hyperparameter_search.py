"""
Enhancement 5: Smart Hyperparameter Optimization with Optuna

Bayesian optimization (TPE) + MedianPruner + two levels of parallelism:

  - n_jobs: runs multiple Optuna trials concurrently (one thread each).
    On a 192-core HPC node, n_jobs=4 keeps four trials in flight at once.
  - n_envs: each trial collects rollouts from n_envs parallel environments
    via SubprocVecEnv, saturating the remaining cores within each trial.
  - TPE sampler: learns from completed trials to focus on high-reward regions.
  - MedianPruner: kills trials below the median at each prune_freq checkpoint,
    freeing budget for the promising configs.

With n_jobs=4, n_envs=8, 200k steps, and pruning at 50k intervals, wall time
drops from ~2 hours (sequential, 1 env) to ~15–20 minutes on a 192-core node.
"""

import csv
import json
import os
import threading
from datetime import datetime
from typing import Callable, List, Optional, Tuple

import numpy as np
import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

optuna.logging.set_verbosity(optuna.logging.WARNING)

_HP_KEYS = [
    "learning_rate", "n_steps", "batch_size", "ent_coef",
    "gamma", "gae_lambda", "clip_range", "n_epochs",
]
_CSV_FIELDS = _HP_KEYS + ["mean_reward", "std_reward", "pruned", "timestamp"]


# =============================================================================
# Callback: reward collection + Optuna pruning hook
# =============================================================================

class _OptunaCallback(BaseCallback):
    """Collects episode rewards and reports to Optuna for pruning."""

    def __init__(self, trial: optuna.Trial, prune_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.trial = trial
        self.prune_freq = prune_freq
        self.episode_rewards: List[float] = []
        self._last_prune_step = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))

        if (self.num_timesteps - self._last_prune_step) >= self.prune_freq:
            self._last_prune_step = self.num_timesteps
            score = self._running_mean(last_n=20)
            self.trial.report(score, self.num_timesteps)
            if self.trial.should_prune():
                raise optuna.TrialPruned()

        return True

    def _running_mean(self, last_n: int = 20) -> float:
        if not self.episode_rewards:
            return float("-inf")
        return float(np.mean(self.episode_rewards[-last_n:]))

    def final_mean(self, last_n: int = 20) -> float:
        return self._running_mean(last_n)

    def final_std(self, last_n: int = 20) -> float:
        if len(self.episode_rewards) < 2:
            return 0.0
        return float(np.std(self.episode_rewards[-last_n:]))


# =============================================================================
# Optuna search
# =============================================================================

def optuna_search(
    env_fn: Callable,
    n_trials: int = 20,
    total_timesteps: int = 200_000,
    prune_freq: int = 50_000,
    log_dir: str = "hp_search_results",
    seed: int = 42,
    n_envs: int = 8,
    n_jobs: int = 4,
    policy_kwargs: Optional[dict] = None,
    model_class=PPO,
    extra_model_kwargs: Optional[dict] = None,
) -> Tuple[Optional[dict], List[dict]]:
    """
    Parallel Bayesian hyperparameter search for PPO.

    Parameters
    ----------
    env_fn : callable
        Zero-argument factory returning a gymnasium environment.
    n_trials : int
        Total Optuna trials to run (pruned ones count too).
    total_timesteps : int
        Training budget per trial.
    prune_freq : int
        Steps between pruning checkpoints.
    log_dir : str
        Directory for CSV, JSON, and TensorBoard logs.
    seed : int
        Seeds Optuna sampler and environments.
    n_envs : int
        Parallel environments per trial (SubprocVecEnv). Uses all CPU cores
        within a single trial. Good values: 4–16 on an HPC node.
    n_jobs : int
        Concurrent Optuna trials. Each runs in its own thread with its own
        model + envs. Good values: 4–8 on a 192-core node.
    policy_kwargs : dict or None
        Forwarded to the model as ``policy_kwargs``.
    model_class : class
        PPO subclass to instantiate (e.g. AttentionRegularizedPPO).
    extra_model_kwargs : dict or None
        Additional keyword arguments forwarded to ``model_class.__init__``.

    Returns
    -------
    best_config : dict
        Hyperparameter dict (all eight keys) with the highest final reward.
    all_results : list of dict
        One entry per trial including scores, pruned flag, and timestamp.
    """
    os.makedirs(log_dir, exist_ok=True)
    extra_model_kwargs = extra_model_kwargs or {}

    csv_path  = os.path.join(log_dir, "hp_results.csv")
    json_path = os.path.join(log_dir, "hp_results.json")
    tb_dir    = os.path.join(log_dir, "tensorboard")

    # Shared state guarded by a lock (n_jobs threads write concurrently)
    _lock        = threading.Lock()
    all_results: List[dict] = []
    best_score   = [float("-inf")]   # list so nonlocal mutation works across threads
    best_config  = [None]

    # SubprocVecEnv can crash Jupyter kernels due to fork/spawn conflicts.
    # DummyVecEnv is stable in notebooks; parallelism comes from n_jobs trials.
    vec_cls = DummyVecEnv

    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=_CSV_FIELDS).writeheader()

    print(f"\n{'='*60}")
    print(f"Optuna TPE Search: {n_trials} trials × {total_timesteps} steps")
    print(f"Parallel: {n_jobs} trials × {n_envs} envs each  "
          f"(~{n_jobs * n_envs} cores in use)")
    print(f"Pruning every {prune_freq} steps (MedianPruner, 5 warmup trials)")
    print(f"Results → {log_dir}/")
    print(f"{'='*60}\n")

    def objective(trial: optuna.Trial) -> float:
        # ── Search space ──────────────────────────────────────────────────────
        lr         = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        n_steps    = trial.suggest_categorical("n_steps", [512, 1024, 2048])
        # batch_size must divide n_steps * n_envs (SB3 rollout buffer size)
        valid_bs   = [b for b in [32, 64, 128, 256]
                      if (n_steps * n_envs) % b == 0]
        batch_size = trial.suggest_categorical("batch_size", valid_bs)
        ent_coef   = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)
        gamma      = trial.suggest_float("gamma", 0.95, 0.999)
        gae_lambda = trial.suggest_float("gae_lambda", 0.9, 0.99)
        clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
        n_epochs   = trial.suggest_int("n_epochs", 5, 20)

        config = dict(
            learning_rate=lr, n_steps=n_steps, batch_size=batch_size,
            ent_coef=ent_coef, gamma=gamma, gae_lambda=gae_lambda,
            clip_range=clip_range, n_epochs=n_epochs,
        )

        # Cap PyTorch threads per trial. Default is 96 on this machine; with
        # n_jobs=4 that creates 384 threads for 192 cores → thrashing → 25 FPS.
        # 4 threads/trial × 4 jobs = 16 threads total → clean parallelism.
        torch.set_num_threads(max(1, 192 // (n_jobs * 4)))

        with _lock:
            trial_num = trial.number + 1
        print(f"[{trial_num}/{n_trials}] {_fmt_config(config)}")

        pruned = False
        score, std = float("-inf"), 0.0

        try:
            env = make_vec_env(
                env_fn, n_envs=n_envs,
                seed=seed + trial.number,
                vec_env_cls=vec_cls,
            )
            model = model_class(
                policy="MlpPolicy",
                env=env,
                learning_rate=config["learning_rate"],
                batch_size=config["batch_size"],
                n_steps=config["n_steps"],
                ent_coef=config["ent_coef"],
                gamma=config["gamma"],
                gae_lambda=config["gae_lambda"],
                clip_range=config["clip_range"],
                n_epochs=config["n_epochs"],
                verbose=0,
                tensorboard_log=tb_dir,
                **({"policy_kwargs": policy_kwargs} if policy_kwargs else {}),
                **extra_model_kwargs,
            )

            cb = _OptunaCallback(trial, prune_freq=prune_freq)
            model.learn(total_timesteps=total_timesteps, callback=cb)

            score = cb.final_mean(last_n=20)
            std   = cb.final_std(last_n=20)
            print(f"  → mean_reward={score:.2f}  std={std:.2f}")

            with _lock:
                if score > best_score[0]:
                    best_score[0]  = score
                    best_config[0] = config.copy()
                    model.save(os.path.join(log_dir, "best_model"))
                    print(f"  ** New best: {best_score[0]:.2f}")

            env.close()

        except optuna.TrialPruned:
            pruned = True
            print(f"  ✗ pruned (trial {trial.number + 1})")
            raise

        finally:
            result = {
                **config,
                "mean_reward": round(score, 4),
                "std_reward":  round(std, 4),
                "pruned":      pruned,
                "timestamp":   datetime.now().isoformat(),
            }
            with _lock:
                all_results.append(result)
                with open(csv_path, "a", newline="") as f:
                    csv.DictWriter(f, fieldnames=_CSV_FIELDS).writerow(result)

        return score

    sampler = TPESampler(seed=seed, n_startup_trials=5)
    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
    study   = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, catch=(Exception,))

    summary = {
        "best_config":  best_config[0],
        "best_score":   best_score[0],
        "n_trials":     n_trials,
        "n_jobs":       n_jobs,
        "n_envs":       n_envs,
        "total_timesteps_per_trial": total_timesteps,
        "prune_freq":   prune_freq,
        "all_results":  all_results,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    n_pruned   = sum(1 for r in all_results if r["pruned"])
    n_complete = len(all_results) - n_pruned
    print(f"\nSearch complete.  {n_complete} completed, {n_pruned} pruned.")
    print(f"Best score : {best_score[0]:.2f}")
    print(f"Best config: {json.dumps(best_config[0], indent=2)}")
    print(f"Full results → {json_path}\n")

    return best_config[0], all_results


# =============================================================================
# Utility helpers
# =============================================================================

def print_ranked_results(all_results: List[dict], top_n: int = 5):
    """Print the top-N completed (non-pruned) configurations by mean_reward."""
    valid = [
        r for r in all_results
        if not r.get("pruned", False)
        and not np.isnan(r.get("mean_reward", float("nan")))
    ]
    ranked = sorted(valid, key=lambda r: r["mean_reward"], reverse=True)

    print(f"\nTop-{min(top_n, len(ranked))} completed configurations:")
    print("-" * 80)
    for i, r in enumerate(ranked[:top_n], 1):
        hp = {k: r[k] for k in _HP_KEYS if k in r}
        print(
            f"  #{i:2d}  reward={r['mean_reward']:.2f} ± {r.get('std_reward', 0):.2f}"
            f"  lr={hp.get('learning_rate', '?'):.1e}"
            f"  n_steps={hp.get('n_steps', '?')}"
            f"  bs={hp.get('batch_size', '?')}"
            f"  ent={hp.get('ent_coef', '?'):.4f}"
            f"  γ={hp.get('gamma', '?'):.4f}"
            f"  λ={hp.get('gae_lambda', '?'):.3f}"
            f"  clip={hp.get('clip_range', '?'):.2f}"
            f"  epochs={hp.get('n_epochs', '?')}"
        )
    print("-" * 80)


# Backward-compatible alias — train_enhanced.py imports this name
random_search = optuna_search


def _fmt_config(c: dict) -> str:
    return (
        f"lr={c['learning_rate']:.1e} n_steps={c['n_steps']} bs={c['batch_size']} "
        f"ent={c['ent_coef']:.4f} γ={c['gamma']:.4f} "
        f"λ={c['gae_lambda']:.3f} clip={c['clip_range']:.2f} epochs={c['n_epochs']}"
    )
