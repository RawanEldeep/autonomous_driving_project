import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass, field, asdict
from typing import List, Optional


@dataclass
class EpisodeMetrics:
    cumulative_reward: float = 0.0
    collided: bool = False
    success: bool = False
    timesteps: int = 0
    traffic_scenario: str = "unknown"

    speeds: List[float] = field(default_factory=list)
    positions: List[object] = field(default_factory=list)
    lane_deviations: List[float] = field(default_factory=list)
    accelerations: List[float] = field(default_factory=list)
    steerings: List[float] = field(default_factory=list)


@dataclass
class AggregatedMetrics:
    n_episodes: int = 0

    mean_reward: float = 0.0
    std_reward: float = 0.0
    min_reward: float = 0.0
    max_reward: float = 0.0

    collision_rate: float = 0.0
    success_rate: float = 0.0

    mean_lane_deviation: float = 0.0
    max_lane_deviation: float = 0.0

    mean_speed: float = 0.0
    speed_variance: float = 0.0

    mean_jerk: float = 0.0
    max_jerk: float = 0.0

    mean_distance: float = 0.0
    total_distance: float = 0.0

    timesteps_to_target: Optional[int] = None


class MetricsEvaluator:

    _A_RANGE = 10.0
    _W_RANGE = 1.0

    def __init__(
        self,
        env,
        model,
        n_episodes: int = 100,
        target_reward: float = 30.0,
        target_window: int = 10,
        render: bool = False,
    ):
        self.env = env
        self.model = model
        self.n_episodes = n_episodes
        self.target_reward = target_reward
        self.target_window = target_window
        self.render = render

        self.episode_metrics: List[EpisodeMetrics] = []

    def evaluate(self) -> AggregatedMetrics:
        self.episode_metrics = []

        for ep_idx in range(self.n_episodes):
            ep = self._run_episode()
            self.episode_metrics.append(ep)

            if (ep_idx + 1) % 10 == 0:
                recent_mean = np.mean(
                    [e.cumulative_reward for e in self.episode_metrics[-10:]]
                )
                print(
                    f"  Episode {ep_idx + 1:3d}/{self.n_episodes} | "
                    f"reward={ep.cumulative_reward:7.2f} | "
                    f"collided={ep.collided} | "
                    f"recent_mean={recent_mean:.2f}"
                )

        return self._aggregate()

    def save_metrics(self, metrics: AggregatedMetrics, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = asdict(metrics)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Metrics saved → {path}")

    def _run_episode(self) -> EpisodeMetrics:
        obs, info = self.env.reset()
        ep = EpisodeMetrics()
        ep.traffic_scenario = info.get("traffic_scenario", "unknown")

        prev_vx = None
        prev_steer = None
        done = False

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)

            if self.render:
                self.env.render()

            ep.cumulative_reward += float(reward)
            ep.timesteps += 1

            ego = self.env.unwrapped.vehicle

            ep.speeds.append(float(ego.speed))

            ep.positions.append(ego.position.copy())

            if not self._is_lane_changing(ego):
                dev = self._lane_deviation(ego)
                if dev is not None:
                    ep.lane_deviations.append(dev)

            curr_vx = float(ego.velocity[0]) if hasattr(ego, "velocity") else float(ego.speed)
            if prev_vx is not None:
                ep.accelerations.append(curr_vx - prev_vx)
            prev_vx = curr_vx

            if hasattr(action, "__len__") and getattr(action, 'ndim', 0) > 0 and len(action) > 1:
                steer = float(action[1])
                if prev_steer is not None:
                    ep.steerings.append(steer - prev_steer)
                prev_steer = steer

            if info.get("crashed", False):
                ep.collided = True

            done = terminated or truncated

        ep.success = not ep.collided
        return ep

    def _is_lane_changing(self, vehicle) -> bool:
        if not hasattr(vehicle, "target_lane_index"):
            return False
        return vehicle.lane_index != vehicle.target_lane_index

    def _lane_deviation(self, vehicle) -> Optional[float]:
        try:
            road = self.env.unwrapped.road
            lane = road.network.get_lane(vehicle.lane_index)
            lat = abs(lane.local_coordinates(vehicle.position)[1])
            return float(lat)
        except Exception:
            return None

    def _compute_jerk(
        self,
        accelerations: List[float],
        steerings: List[float],
    ):
        j_accs = [
            abs(accelerations[i] - accelerations[i - 1]) / self._A_RANGE
            for i in range(1, len(accelerations))
        ]
        j_steers = [abs(ds) / self._W_RANGE for ds in steerings]

        if not j_accs and not j_steers:
            return 0.0, 0.0

        if j_accs and j_steers:
            n = min(len(j_accs), len(j_steers))
            j_total = [(j_accs[i] + j_steers[i]) / 2.0 for i in range(n)]
        else:
            j_total = j_accs or j_steers

        return float(np.mean(j_total)), float(np.max(j_total))

    def _driving_distance(self, positions: list) -> float:
        if len(positions) < 2:
            return 0.0
        return float(
            sum(
                np.linalg.norm(positions[i] - positions[i - 1])
                for i in range(1, len(positions))
            )
        )

    def _aggregate(self) -> AggregatedMetrics:
        agg = AggregatedMetrics(n_episodes=len(self.episode_metrics))

        rewards = [ep.cumulative_reward for ep in self.episode_metrics]
        agg.mean_reward = float(np.mean(rewards))
        agg.std_reward = float(np.std(rewards))
        agg.min_reward = float(np.min(rewards))
        agg.max_reward = float(np.max(rewards))

        collisions = [ep.collided for ep in self.episode_metrics]
        agg.collision_rate = float(np.mean(collisions) * 100.0)
        agg.success_rate = float((1.0 - np.mean(collisions)) * 100.0)

        all_devs = [d for ep in self.episode_metrics for d in ep.lane_deviations]
        if all_devs:
            agg.mean_lane_deviation = float(np.mean(all_devs))
            agg.max_lane_deviation = float(np.max(all_devs))

        all_speeds = [s for ep in self.episode_metrics for s in ep.speeds]
        if all_speeds:
            agg.mean_speed = float(np.mean(all_speeds))
            agg.speed_variance = float(np.var(all_speeds))

        jerk_means, jerk_maxes = [], []
        for ep in self.episode_metrics:
            jm, jx = self._compute_jerk(ep.accelerations, ep.steerings)
            jerk_means.append(jm)
            jerk_maxes.append(jx)
        agg.mean_jerk = float(np.mean(jerk_means))
        agg.max_jerk = float(np.mean(jerk_maxes))

        distances = [self._driving_distance(ep.positions) for ep in self.episode_metrics]
        agg.mean_distance = float(np.mean(distances))
        agg.total_distance = float(np.sum(distances))

        return agg


def compute_learning_efficiency(
    reward_log: List[float],
    target_reward: float = 30.0,
    window: int = 10,
    timestep_interval: int = 1,
) -> Optional[int]:
    if len(reward_log) < window:
        return None
    for i in range(window - 1, len(reward_log)):
        if np.mean(reward_log[i - window + 1 : i + 1]) >= target_reward:
            return (i + 1) * timestep_interval
    return None


class MetricsPlotter:

    def __init__(self, save_dir: str = "results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def plot_all(
        self,
        baseline: AggregatedMetrics,
        enhanced: AggregatedMetrics,
        baseline_episodes: Optional[List[EpisodeMetrics]] = None,
        enhanced_episodes: Optional[List[EpisodeMetrics]] = None,
    ):
        self.plot_bar_comparison(baseline, enhanced)
        if baseline_episodes and enhanced_episodes:
            self.plot_reward_distribution(baseline_episodes, enhanced_episodes)
            self.plot_lane_deviation_boxplot(baseline_episodes, enhanced_episodes)
            self.plot_speed_profile(baseline_episodes, enhanced_episodes)
            self.plot_jerk_profile(baseline_episodes, enhanced_episodes)
        self.print_summary_table(baseline, enhanced)

    def plot_learning_curves(
        self,
        baseline_rewards: List[float],
        enhanced_rewards: List[float],
        timestep_interval: int = 1000,
    ):
        fig, ax = plt.subplots(figsize=(10, 5))

        def _smooth(data, w=10):
            kernel = np.ones(w) / w
            return np.convolve(data, kernel, mode="valid")

        for rewards, color, label in [
            (baseline_rewards, "steelblue", "Baseline"),
            (enhanced_rewards, "darkorange", "Enhanced"),
        ]:
            if not rewards:
                continue
            steps = np.arange(len(rewards)) * timestep_interval
            ax.plot(steps, rewards, alpha=0.25, color=color)
            if len(rewards) >= 10:
                ax.plot(
                    steps[9:], _smooth(rewards),
                    color=color, linewidth=2, label=f"{label} (smoothed)"
                )

        ax.set_xlabel("Timesteps")
        ax.set_ylabel("Episode Reward")
        ax.set_title("Learning Curves: Baseline vs Enhanced")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("learning_curves.png")

    def plot_bar_comparison(
        self, baseline: AggregatedMetrics, enhanced: AggregatedMetrics
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        cats = ["Baseline", "Enhanced"]
        colors = ["steelblue", "darkorange"]

        for ax, title, vals in [
            (axes[0], "Collision Rate (%)",
             [baseline.collision_rate, enhanced.collision_rate]),
            (axes[1], "Success Rate (%)",
             [baseline.success_rate, enhanced.success_rate]),
        ]:
            bars = ax.bar(cats, vals, color=colors)
            ax.set_title(title)
            ax.set_ylabel(title)
            if "Success" in title:
                ax.set_ylim(0, 110)
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + 1.0,
                    f"{v:.1f}%",
                    ha="center", fontweight="bold",
                )

        plt.tight_layout()
        self._save("collision_success_rates.png")

    def plot_reward_distribution(
        self,
        baseline_eps: List[EpisodeMetrics],
        enhanced_eps: List[EpisodeMetrics],
    ):
        fig, ax = plt.subplots(figsize=(10, 5))
        b_rewards = [ep.cumulative_reward for ep in baseline_eps]
        e_rewards = [ep.cumulative_reward for ep in enhanced_eps]
        ax.hist(b_rewards, bins=20, alpha=0.6, color="steelblue",  label="Baseline")
        ax.hist(e_rewards, bins=20, alpha=0.6, color="darkorange", label="Enhanced")
        ax.set_xlabel("Cumulative Episode Reward")
        ax.set_ylabel("Frequency")
        ax.set_title("Episode Reward Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("reward_distribution.png")

    def plot_lane_deviation_boxplot(
        self,
        baseline_eps: List[EpisodeMetrics],
        enhanced_eps: List[EpisodeMetrics],
    ):
        b_devs = [d for ep in baseline_eps for d in ep.lane_deviations]
        e_devs = [d for ep in enhanced_eps for d in ep.lane_deviations]

        data, labels, colors = [], [], ["steelblue", "darkorange"]
        if b_devs:
            data.append(b_devs); labels.append("Baseline")
        if e_devs:
            data.append(e_devs); labels.append("Enhanced")

        if not data:
            return

        fig, ax = plt.subplots(figsize=(8, 5))
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors[: len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel("Lane Deviation (m)")
        ax.set_title("Lane Deviation Distribution")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("lane_deviation_boxplot.png")

    def plot_speed_profile(
        self,
        baseline_eps: List[EpisodeMetrics],
        enhanced_eps: List[EpisodeMetrics],
        episode_idx: int = 0,
    ):
        fig, ax = plt.subplots(figsize=(10, 5))
        if episode_idx < len(baseline_eps):
            ax.plot(
                baseline_eps[episode_idx].speeds,
                color="steelblue", alpha=0.8, label="Baseline",
            )
        if episode_idx < len(enhanced_eps):
            ax.plot(
                enhanced_eps[episode_idx].speeds,
                color="darkorange", alpha=0.8, label="Enhanced",
            )
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Speed (m/s)")
        ax.set_title(f"Speed Profile — Episode {episode_idx + 1}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("speed_profile.png")

    def plot_jerk_profile(
        self,
        baseline_eps: List[EpisodeMetrics],
        enhanced_eps: List[EpisodeMetrics],
        episode_idx: int = 0,
    ):

        def _jerk_series(accels, a_range=10.0):
            return [
                abs(accels[i] - accels[i - 1]) / a_range
                for i in range(1, len(accels))
            ]

        fig, ax = plt.subplots(figsize=(10, 5))
        if episode_idx < len(baseline_eps) and baseline_eps[episode_idx].accelerations:
            ax.plot(
                _jerk_series(baseline_eps[episode_idx].accelerations),
                color="steelblue", alpha=0.8, label="Baseline",
            )
        if episode_idx < len(enhanced_eps) and enhanced_eps[episode_idx].accelerations:
            ax.plot(
                _jerk_series(enhanced_eps[episode_idx].accelerations),
                color="darkorange", alpha=0.8, label="Enhanced",
            )
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Normalised Jerk")
        ax.set_title(f"Jerk Profile — Episode {episode_idx + 1}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        self._save("jerk_profile.png")

    def print_summary_table(
        self, baseline: AggregatedMetrics, enhanced: AggregatedMetrics
    ):
        W = 65
        print("\n" + "=" * W)
        print(f"{'METRIC':<30} {'BASELINE':>16} {'ENHANCED':>16}")
        print("=" * W)

        rows = [
            (
                "Mean Reward",
                f"{baseline.mean_reward:.2f} ± {baseline.std_reward:.2f}",
                f"{enhanced.mean_reward:.2f} ± {enhanced.std_reward:.2f}",
            ),
            ("Collision Rate", f"{baseline.collision_rate:.1f} %", f"{enhanced.collision_rate:.1f} %"),
            ("Success Rate",   f"{baseline.success_rate:.1f} %",   f"{enhanced.success_rate:.1f} %"),
            ("Mean Lane Dev (m)", f"{baseline.mean_lane_deviation:.3f}", f"{enhanced.mean_lane_deviation:.3f}"),
            ("Max Lane Dev (m)",  f"{baseline.max_lane_deviation:.3f}",  f"{enhanced.max_lane_deviation:.3f}"),
            ("Mean Speed (m/s)",  f"{baseline.mean_speed:.2f}",          f"{enhanced.mean_speed:.2f}"),
            ("Speed Variance",    f"{baseline.speed_variance:.3f}",       f"{enhanced.speed_variance:.3f}"),
            ("Mean Jerk",         f"{baseline.mean_jerk:.4f}",            f"{enhanced.mean_jerk:.4f}"),
            ("Mean Distance (m)", f"{baseline.mean_distance:.1f}",        f"{enhanced.mean_distance:.1f}"),
        ]

        for name, b_val, e_val in rows:
            print(f"{name:<30} {b_val:>16} {e_val:>16}")

        print("=" * W + "\n")

    def _save(self, filename: str):
        path = os.path.join(self.save_dir, filename)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {path}")
