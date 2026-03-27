"""
Custom environment wrappers implementing three enhancements:

Enhancement 1 - EnhancedRewardWrapper:
    Adds near-miss TTC penalties, smooth driving rewards, and lane-centering
    rewards on top of the existing highway-env reward signal.

Enhancement 2 - DynamicTrafficWrapper:
    Randomizes vehicle density each episode (sparse / moderate / dense) and
    injects random driving events (sudden braking, slow vehicles, aggressive
    lane changes) during episodes.

Enhancement 3 - ObservationEnrichmentWrapper:
    Extends the kinematic observation from 7 to 10 features per vehicle by
    appending Time-to-Collision (TTC), lateral distance to lane boundary, and
    a binary lane-change-in-progress indicator.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# =============================================================================
# Enhancement 1: Enhanced Reward Shaping
# =============================================================================

class EnhancedRewardWrapper(gym.Wrapper):
    """
    Adds three reward components on top of the existing highway-env reward:

    1. Near-miss penalty  — negative reward when TTC to any nearby vehicle
                            falls below `ttc_threshold` seconds.
    2. Smooth driving     — positive reward proportional to how little the
                            ego speed changed since the last step.
    3. Lane centering     — positive reward proportional to how close the ego
                            vehicle is to the center of its current lane.

    Each component can be toggled independently via the enable_* flags,
    allowing ablation studies.
    """

    def __init__(
        self,
        env,
        ttc_threshold: float = 3.0,
        ttc_penalty_weight: float = 0.3,
        smooth_driving_weight: float = 0.1,
        lane_centering_weight: float = 0.1,
        enable_ttc_penalty: bool = True,
        enable_smooth_reward: bool = True,
        enable_lane_centering: bool = True,
    ):
        super().__init__(env)
        self.ttc_threshold = ttc_threshold
        self.ttc_penalty_weight = ttc_penalty_weight
        self.smooth_driving_weight = smooth_driving_weight
        self.lane_centering_weight = lane_centering_weight
        self.enable_ttc_penalty = enable_ttc_penalty
        self.enable_smooth_reward = enable_smooth_reward
        self.enable_lane_centering = enable_lane_centering

        self._prev_speed = None

    def reset(self, **kwargs):
        self._prev_speed = None
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        ego = self.env.unwrapped.vehicle

        # --- Near-miss TTC penalty ---
        if self.enable_ttc_penalty:
            ttc_pen = self._ttc_penalty(ego)
            reward += ttc_pen
            info["ttc_penalty"] = ttc_pen

        # --- Smooth driving reward ---
        if self.enable_smooth_reward:
            smooth_r = self._smooth_reward(ego)
            reward += smooth_r
            info["smooth_reward"] = smooth_r

        # --- Lane-centering reward ---
        if self.enable_lane_centering:
            lane_r = self._lane_centering_reward(ego)
            reward += lane_r
            info["lane_centering_reward"] = lane_r

        self._prev_speed = float(ego.speed)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ttc(self, ego, other) -> float:
        """
        Time-to-Collision (seconds) from ego to `other`.
        Returns inf when the other vehicle is behind or moving away.
        """
        rel_x = other.position[0] - ego.position[0]
        rel_vx = other.velocity[0] - ego.velocity[0]
        if rel_x <= 0 or rel_vx >= 0:
            return float("inf")
        return max(0.0, rel_x / (-rel_vx + 1e-6))

    def _ttc_penalty(self, ego) -> float:
        """Sum of near-miss penalties across all relevant vehicles."""
        penalty = 0.0
        for vehicle in self.env.unwrapped.road.vehicles:
            if vehicle is ego:
                continue
            # Only penalise same lane or immediate neighbours
            if abs(vehicle.lane_index[2] - ego.lane_index[2]) > 1:
                continue
            ttc = self._ttc(ego, vehicle)
            if ttc < self.ttc_threshold:
                # Penalty increases linearly as TTC → 0
                penalty -= self.ttc_penalty_weight * (1.0 - ttc / self.ttc_threshold)
        return penalty

    def _smooth_reward(self, ego) -> float:
        """Reward small speed changes (proxy for smooth acceleration)."""
        if self._prev_speed is None:
            return 0.0
        delta_v = abs(float(ego.speed) - self._prev_speed)
        # Normalise: assume max meaningful speed change per step is 5 m/s
        normalised = min(delta_v / 5.0, 1.0)
        return self.smooth_driving_weight * (1.0 - normalised)

    def _lane_centering_reward(self, ego) -> float:
        """Reward proximity to lane centre (lateral deviation → 0)."""
        try:
            lane = self.env.unwrapped.road.network.get_lane(ego.lane_index)
            lat_dev = abs(lane.local_coordinates(ego.position)[1])
            half_width = (lane.width if hasattr(lane, "width") else 4.0) / 2.0
            normalised = min(lat_dev / half_width, 1.0)
            return self.lane_centering_weight * (1.0 - normalised)
        except Exception:
            return 0.0


# =============================================================================
# Enhancement 2: Dynamic Traffic Scenarios
# =============================================================================

class DynamicTrafficWrapper(gym.Wrapper):
    """
    Introduces two sources of traffic variability:

    Density randomisation (at reset):
        Each episode the number of vehicles is sampled uniformly from one of
        three scenarios chosen with equal probability:
            Sparse   — 5–10  vehicles  (off-peak)
            Moderate — 15–20 vehicles  (baseline)
            Dense    — 25–30 vehicles  (rush hour)

    Dynamic events (during episode):
        At each step, with probability `event_probability`, each surrounding
        vehicle may experience one of:
            • slow          — target speed halved (slow-moving obstacle)
            • brake         — current speed reduced by 30 % (sudden braking)
            • lane_change   — forced lane change to an adjacent lane
    """

    SPARSE = "sparse"
    MODERATE = "moderate"
    DENSE = "dense"

    def __init__(
        self,
        env,
        enable_dynamic_density: bool = True,
        enable_dynamic_events: bool = True,
        event_probability: float = 0.02,
    ):
        super().__init__(env)
        self.enable_dynamic_density = enable_dynamic_density
        self.enable_dynamic_events = enable_dynamic_events
        self.event_probability = event_probability
        self.current_scenario = self.MODERATE

    def reset(self, **kwargs):
        if self.enable_dynamic_density:
            scenario = np.random.choice([self.SPARSE, self.MODERATE, self.DENSE])
            self.current_scenario = scenario

            if scenario == self.SPARSE:
                n_vehicles = int(np.random.randint(5, 11))
            elif scenario == self.MODERATE:
                n_vehicles = int(np.random.randint(15, 21))
            else:
                n_vehicles = int(np.random.randint(25, 31))

            self.env.unwrapped.config["vehicles_count"] = n_vehicles

        obs, info = self.env.reset(**kwargs)
        info["traffic_scenario"] = self.current_scenario
        info["vehicles_count"] = self.env.unwrapped.config.get("vehicles_count", 15)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.enable_dynamic_events and not terminated:
            self._trigger_events()

        info["traffic_scenario"] = self.current_scenario
        return obs, reward, terminated, truncated, info

    def _trigger_events(self):
        """Randomly perturb surrounding vehicles to create dynamic scenarios."""
        ego = self.env.unwrapped.vehicle
        lanes_count = self.env.unwrapped.config.get("lanes_count", 3)

        for vehicle in self.env.unwrapped.road.vehicles:
            if vehicle is ego:
                continue
            if np.random.random() >= self.event_probability:
                continue

            event = np.random.choice(["slow", "brake", "lane_change"])

            if event == "slow" and hasattr(vehicle, "target_speed"):
                vehicle.target_speed = max(5.0, vehicle.target_speed * 0.5)

            elif event == "brake" and hasattr(vehicle, "speed"):
                vehicle.speed = max(0.0, vehicle.speed * 0.7)

            elif event == "lane_change" and hasattr(vehicle, "target_lane_index"):
                cur_lane = vehicle.lane_index[2]
                new_lane = cur_lane + np.random.choice([-1, 1])
                if 0 <= new_lane < lanes_count:
                    vehicle.target_lane_index = (
                        vehicle.lane_index[0],
                        vehicle.lane_index[1],
                        new_lane,
                    )


# =============================================================================
# Enhancement 3: Observation Space Enrichment (7 → 10 features)
# =============================================================================

class ObservationEnrichmentWrapper(gym.Wrapper):
    """
    Extends kinematic observations from 7 to 10 features per vehicle slot:

        Original  [0–6]: presence, x, y, vx, vy, cos_h, sin_h
        Added     [7]:   Time-to-Collision (normalised to [0, 1])
                  [8]:   Lateral distance to nearest lane boundary (normalised)
                  [9]:   Binary: 1 if vehicle is mid lane-change, else 0

    The feature extractor in custom_policy / train_enhanced must be configured
    with `in_size=10` per vehicle to match this extended observation.
    """

    TTC_MAX = 10.0          # seconds — cap for normalisation
    HEADING_THRESHOLD = 0.1  # radians — heading deviation indicating lane change

    def __init__(self, env, vehicles_count: int = 10):
        super().__init__(env)
        self.vehicles_count = vehicles_count

        # Replace observation space to reflect the extra 3 features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(vehicles_count, 10),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._enrich(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._enrich(obs), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Observation enrichment
    # ------------------------------------------------------------------

    def _enrich(self, obs: np.ndarray) -> np.ndarray:
        """Append TTC, lateral distance, and lane-change flag to each row."""
        env = self.env.unwrapped
        ego = env.vehicle
        road = env.road

        enriched = np.zeros((self.vehicles_count, 10), dtype=np.float32)

        # Copy the original 7 features (clip rows if obs has fewer than vehicles_count)
        n_src = min(obs.shape[0], self.vehicles_count)
        enriched[:n_src, :7] = obs[:n_src, :7]

        # Build an ordered list: ego first, then others
        others = [v for v in road.vehicles if v is not ego]
        ordered = [ego] + others

        for i in range(min(self.vehicles_count, len(ordered))):
            vehicle = ordered[i]

            # Feature 7: TTC (normalised)
            if vehicle is ego:
                ttc = self._ego_min_ttc(ego, road)
            else:
                ttc = self._ttc(ego, vehicle)
            enriched[i, 7] = min(ttc, self.TTC_MAX) / self.TTC_MAX

            # Feature 8: lateral distance to nearest lane boundary (normalised)
            enriched[i, 8] = self._lateral_dist(vehicle, road)

            # Feature 9: lane-change indicator
            enriched[i, 9] = self._is_lane_changing(vehicle)

        return enriched

    def _ttc(self, ego, other) -> float:
        rel_x = other.position[0] - ego.position[0]
        rel_vx = other.velocity[0] - ego.velocity[0]
        if rel_x <= 0 or rel_vx >= 0:
            return self.TTC_MAX
        return min(rel_x / (-rel_vx + 1e-6), self.TTC_MAX)

    def _ego_min_ttc(self, ego, road) -> float:
        """Minimum TTC from ego to any vehicle ahead."""
        return min(
            (self._ttc(ego, v) for v in road.vehicles if v is not ego),
            default=self.TTC_MAX,
        )

    def _lateral_dist(self, vehicle, road) -> float:
        """Normalised lateral distance to nearest lane boundary ∈ [0, 1]."""
        try:
            lane = road.network.get_lane(vehicle.lane_index)
            lat = abs(lane.local_coordinates(vehicle.position)[1])
            half_width = (lane.width if hasattr(lane, "width") else 4.0) / 2.0
            return float(min(lat / half_width, 1.0))
        except Exception:
            return 0.0

    def _is_lane_changing(self, vehicle) -> float:
        """Return 1.0 if vehicle is actively changing lanes, else 0.0."""
        if not hasattr(vehicle, "target_lane_index"):
            return 0.0
        if vehicle.lane_index != vehicle.target_lane_index:
            return 1.0
        heading = float(getattr(vehicle, "heading", 0.0))
        return 1.0 if abs(heading) > self.HEADING_THRESHOLD else 0.0
