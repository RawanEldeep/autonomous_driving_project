import numpy as np
import gymnasium as gym
from gymnasium import spaces


class EnhancedRewardWrapper(gym.Wrapper):

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

        if self.enable_ttc_penalty:
            ttc_pen = self._ttc_penalty(ego)
            reward += ttc_pen
            info["ttc_penalty"] = ttc_pen

        if self.enable_smooth_reward:
            smooth_r = self._smooth_reward(ego)
            reward += smooth_r
            info["smooth_reward"] = smooth_r

        if self.enable_lane_centering:
            lane_r = self._lane_centering_reward(ego)
            reward += lane_r
            info["lane_centering_reward"] = lane_r

        self._prev_speed = float(ego.speed)
        return obs, reward, terminated, truncated, info

    def _ttc(self, ego, other) -> float:
        rel_x = other.position[0] - ego.position[0]
        rel_vx = other.velocity[0] - ego.velocity[0]
        if rel_x <= 0 or rel_vx >= 0:
            return float("inf")
        return max(0.0, rel_x / (-rel_vx + 1e-6))

    def _ttc_penalty(self, ego) -> float:
        penalty = 0.0
        for vehicle in self.env.unwrapped.road.vehicles:
            if vehicle is ego:
                continue
            if abs(vehicle.lane_index[2] - ego.lane_index[2]) > 1:
                continue
            ttc = self._ttc(ego, vehicle)
            if ttc < self.ttc_threshold:
                penalty -= self.ttc_penalty_weight * (1.0 - ttc / self.ttc_threshold)
        return penalty

    def _smooth_reward(self, ego) -> float:
        if self._prev_speed is None:
            return 0.0
        delta_v = abs(float(ego.speed) - self._prev_speed)
        normalised = min(delta_v / 5.0, 1.0)
        return self.smooth_driving_weight * (1.0 - normalised)

    def _lane_centering_reward(self, ego) -> float:
        try:
            lane = self.env.unwrapped.road.network.get_lane(ego.lane_index)
            lat_dev = abs(lane.local_coordinates(ego.position)[1])
            half_width = (lane.width if hasattr(lane, "width") else 4.0) / 2.0
            normalised = min(lat_dev / half_width, 1.0)
            return self.lane_centering_weight * (1.0 - normalised)
        except Exception:
            return 0.0


class DynamicTrafficWrapper(gym.Wrapper):

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


class ObservationEnrichmentWrapper(gym.Wrapper):

    TTC_MAX = 10.0
    HEADING_THRESHOLD = 0.1

    def __init__(self, env, vehicles_count: int = 10):
        super().__init__(env)
        self.vehicles_count = vehicles_count

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

    def _enrich(self, obs: np.ndarray) -> np.ndarray:
        env = self.env.unwrapped
        ego = env.vehicle
        road = env.road

        enriched = np.zeros((self.vehicles_count, 10), dtype=np.float32)

        n_src = min(obs.shape[0], self.vehicles_count)
        enriched[:n_src, :7] = obs[:n_src, :7]

        others = [v for v in road.vehicles if v is not ego]
        ordered = [ego] + others

        for i in range(min(self.vehicles_count, len(ordered))):
            vehicle = ordered[i]

            if vehicle is ego:
                ttc = self._ego_min_ttc(ego, road)
            else:
                ttc = self._ttc(ego, vehicle)
            enriched[i, 7] = min(ttc, self.TTC_MAX) / self.TTC_MAX

            enriched[i, 8] = self._lateral_dist(vehicle, road)

            enriched[i, 9] = self._is_lane_changing(vehicle)

        return enriched

    def _ttc(self, ego, other) -> float:
        rel_x = other.position[0] - ego.position[0]
        rel_vx = other.velocity[0] - ego.velocity[0]
        if rel_x <= 0 or rel_vx >= 0:
            return self.TTC_MAX
        return min(rel_x / (-rel_vx + 1e-6), self.TTC_MAX)

    def _ego_min_ttc(self, ego, road) -> float:
        return min(
            (self._ttc(ego, v) for v in road.vehicles if v is not ego),
            default=self.TTC_MAX,
        )

    def _lateral_dist(self, vehicle, road) -> float:
        try:
            lane = road.network.get_lane(vehicle.lane_index)
            lat = abs(lane.local_coordinates(vehicle.position)[1])
            half_width = (lane.width if hasattr(lane, "width") else 4.0) / 2.0
            return float(min(lat / half_width, 1.0))
        except Exception:
            return 0.0

    def _is_lane_changing(self, vehicle) -> float:
        if not hasattr(vehicle, "target_lane_index"):
            return 0.0
        if vehicle.lane_index != vehicle.target_lane_index:
            return 1.0
        heading = float(getattr(vehicle, "heading", 0.0))
        return 1.0 if abs(heading) > self.HEADING_THRESHOLD else 0.0
