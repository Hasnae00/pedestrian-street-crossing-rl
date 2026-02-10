from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np


@dataclass
class EnvParams:
    """Centralized knobs for the simulator; tweak here during live demos."""

    # --- Simulation clock ---
    dt: float = 0.2
    episode_time_limit: float = 60.0

    # --- Layout / geometry ---
    spawn_distance_m: float = 120.0

    # Pedestrian: time to cross ONE lane
    t_lane_s: float = 4.0

    # --- Vehicle speed model (50–55 km/h typical) ---
    v_mean_mps: float = 52.5 / 3.6   # ≈ 14.58 m/s
    v_std_mps: float = 1.0
    v_min_mps: float = 45.0 / 3.6   # ≈ 12.50 m/s
    v_max_mps: float = 60.0 / 3.6   # ≈ 16.67 m/s

    # --- Traffic arrivals ---
    min_headway_s: float = 0.8
    warmup_time_s: float = 10.0

    # Visual/realism: prevent cars overlapping in same lane (1D no-overtake)
    min_car_spacing_m: float = 12.0

    # --- Reward knobs (highlight when presenting) ---
    r_collision: float = -120.0   # Collisions are catastrophic
    r_success: float = 10.0       # Reward for clearing both lanes
    r_step: float = -0.003        # dt=0.2, horizon 60s -> 300 steps max => at worst 300 * -0.003 = -0.9 per episode
    r_timeout: float = -3.0       # Timing out hurts but far less than a crash
    r_reach_median: float = 2.0

    # Geometry-derived helper
    safety_margin_s: float = 1.0  # Extra buffer seconds added on top of t_lane_s when evaluating safe gaps


@dataclass
class Scenario:
    lambda_lane1: float
    lambda_lane2: float


class PedestrianCrossingEnv:
    """
    Two-step crossing with a median:
      CURB -> CROSS_LANE1 -> MEDIAN -> CROSS_LANE2 -> SUCCESS(done)

    Lanes are 1D with crossing line at x=0:
      lane 0: cars spawn at +D and move left towards 0 (x decreases)
      lane 1: cars spawn at -D and move right towards 0 (x increases)

    Actions: WAIT=0, GO=1

    ped_stage:
      0 = CURB (safe)
      1 = CROSS_LANE1 (exposed to lane 0)
      2 = MEDIAN (safe)
      3 = CROSS_LANE2 (exposed to lane 1)
    """

    ACTION_WAIT = 0
    ACTION_GO = 1

    STAGE_CURB = 0
    STAGE_CROSS_LANE1 = 1
    STAGE_MEDIAN = 2
    STAGE_CROSS_LANE2 = 3

    def __init__(self, params: EnvParams, scenario: Scenario, seed: Optional[int] = None):
        self.params = params
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)

        self.cars: List[List[Dict[str, float]]] = [[], []]
        self._t_next_arrival: List[float] = [0.0, 0.0]

        self.ped_stage: int = self.STAGE_CURB
        self.t_in_stage: float = 0.0

        self.t: float = 0.0

    # -----------------------
    # Core API
    # -----------------------
    def reset(self) -> np.ndarray:
        self.t = 0.0
        self.ped_stage = self.STAGE_CURB
        self.t_in_stage = 0.0

        self.cars = [[], []]
        self._t_next_arrival = [
            self._sample_interarrival(self.scenario.lambda_lane1),
            self._sample_interarrival(self.scenario.lambda_lane2),
        ]

        # Warm-up: populate traffic (does not count as episode time)
        warmup_steps = int(self.params.warmup_time_s / self.params.dt)
        for _ in range(warmup_steps):
            self._advance_lane(0, self.scenario.lambda_lane1)
            self._advance_lane(1, self.scenario.lambda_lane2)

        self.t = 0.0
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        action = int(action)
        done = False
        terminal_event = None

        # time always passes -> step penalty
        reward = float(self.params.r_step)

        # 1) Apply action (meaningful only at CURB or MEDIAN)
        if action == self.ACTION_GO:
            if self.ped_stage == self.STAGE_CURB:
                self.ped_stage = self.STAGE_CROSS_LANE1
                self.t_in_stage = 0.0

            elif self.ped_stage == self.STAGE_MEDIAN:
                self.ped_stage = self.STAGE_CROSS_LANE2
                self.t_in_stage = 0.0
        else:
            pass

        # 2) Advance traffic
        self._advance_lane(0, self.scenario.lambda_lane1)
        self._advance_lane(1, self.scenario.lambda_lane2)

        # 3) Advance time and crossing timer
        self.t += self.params.dt
        if self.ped_stage in (self.STAGE_CROSS_LANE1, self.STAGE_CROSS_LANE2):
            self.t_in_stage += self.params.dt

        # 4) Collision (only when exposed)
        if self._check_collision():
            reward += self.params.r_collision # override
            done = True
            terminal_event = "collision"

        # 5) Stage transitions / success
        if not done:
            if self.ped_stage == self.STAGE_CROSS_LANE1 and self.t_in_stage >= self.params.t_lane_s:
                self.ped_stage = self.STAGE_MEDIAN
                self.t_in_stage = 0.0
                reward += self.params.r_reach_median

            elif self.ped_stage == self.STAGE_CROSS_LANE2 and self.t_in_stage >= self.params.t_lane_s:
                reward += float(self.params.r_success)
                done = True
                terminal_event = "success"

        # 6) Timeout
        if (not done) and (self.t >= self.params.episode_time_limit):
            reward += float(self.params.r_timeout)
            done = True
            terminal_event = "timeout"

        state = self._get_state()
        info = {
            "t": float(self.t),
            "tta_lane1": float(self._tta_lane(0)),
            "tta_lane2": float(self._tta_lane(1)),
            "ped_stage": int(self.ped_stage),
            "stage_progress": float(self._stage_progress()),
            "n_cars_lane1": int(len(self.cars[0])),
            "n_cars_lane2": int(len(self.cars[1])),
            "terminal_event": terminal_event,
        }
        return state, float(reward), bool(done), info


    # -----------------------
    # State helpers
    # -----------------------
    def _stage_progress(self) -> float:
        if self.ped_stage in (self.STAGE_CROSS_LANE1, self.STAGE_CROSS_LANE2):
            return float(np.clip(self.t_in_stage / self.params.t_lane_s, 0.0, 1.0))
        return 0.0

    def _safe_threshold(self) -> float:
        return float(self.params.t_lane_s + self.params.safety_margin_s + self.params.dt)

    def _get_state(self) -> np.ndarray:
        """Return normalized state: [tta1_rel, tta2_rel, stage, progress, active1, active2]."""
        tta1 = self._tta_lane(0)
        tta2 = self._tta_lane(1)
        threshold = self._safe_threshold()

        lane1_active = 1.0 if self.ped_stage in (self.STAGE_CURB, self.STAGE_CROSS_LANE1) else 0.0
        lane2_active = 1.0 if self.ped_stage in (self.STAGE_MEDIAN, self.STAGE_CROSS_LANE2) else 0.0

        if lane1_active == 0.0:
            tta1 = 0.0
        if lane2_active == 0.0:
            tta2 = 0.0

        scale = threshold if threshold > 1e-6 else 1.0
        tta1_rel = float(np.clip(tta1 / scale, 0.0, 6.0))
        tta2_rel = float(np.clip(tta2 / scale, 0.0, 6.0))

        return np.array(
            [tta1_rel, tta2_rel, float(self.ped_stage), self._stage_progress(), lane1_active, lane2_active],
            dtype=np.float32,
        )


    def _tta_lane(self, lane: int) -> float:
        ttas = []
        for c in self.cars[lane]:
            v = max(c["v"], 1e-6)
            x = c["x"]
            if lane == 0 and x > 0.0:
                ttas.append(x / v)
            if lane == 1 and x < 0.0:
                ttas.append((-x) / v)
        return float(min(ttas)) if ttas else 999.0

    # -----------------------
    # Traffic dynamics
    # -----------------------
    def _sample_interarrival(self, lam: float) -> float:
        lam = max(float(lam), 1e-9)
        return float(self.params.min_headway_s + self.rng.exponential(1.0 / lam))

    def _sample_speed(self) -> float:
        v = float(self.rng.normal(self.params.v_mean_mps, self.params.v_std_mps))
        return float(np.clip(v, self.params.v_min_mps, self.params.v_max_mps))

    def _spawn_car(self, lane: int) -> None:
        x0 = float(self.params.spawn_distance_m)
        if lane == 1:
            x0 = -x0

        # avoid stacking at spawn point
        d = float(self.params.min_car_spacing_m)
        for c in self.cars[lane]:
            if abs(c["x"] - x0) < d:
                return
        self.cars[lane].append({"x": x0, "x_prev": x0, "v": self._sample_speed()})


    def _enforce_min_spacing(self, lane: int) -> None:
        d = float(self.params.min_car_spacing_m)
        if len(self.cars[lane]) <= 1:
            return

        if lane == 0:
            # lane 0 moves left; front has smaller x
            cars = sorted(self.cars[lane], key=lambda c: c["x"])
            for i in range(1, len(cars)):
                min_x = cars[i - 1]["x"] + d
                if cars[i]["x"] < min_x:
                    cars[i]["x"] = min_x
            self.cars[lane] = cars
        else:
            # lane 1 moves right; front has larger x
            cars = sorted(self.cars[lane], key=lambda c: c["x"], reverse=True)
            for i in range(1, len(cars)):
                max_x = cars[i - 1]["x"] - d
                if cars[i]["x"] > max_x:
                    cars[i]["x"] = max_x
            self.cars[lane] = cars

    def _advance_lane(self, lane: int, lam: float) -> None:
        # --- 1) spawn arrivals according to Poisson gap model ---
        self._t_next_arrival[lane] -= self.params.dt
        while self._t_next_arrival[lane] <= 0.0:
            self._spawn_car(lane)
            self._t_next_arrival[lane] += self._sample_interarrival(lam)

        # --- 2) integrate vehicle motion ---
        for c in self.cars[lane]:
            c["x_prev"] = c["x"]  
            if lane == 0:
                c["x"] -= c["v"] * self.params.dt
            else:
                c["x"] += c["v"] * self.params.dt

        # --- 3) enforce spacing + prune passed cars ---
        self._enforce_min_spacing(lane)

        # cleanup (cars far past crossing line)
        if lane == 0:
            self.cars[lane] = [c for c in self.cars[lane] if c["x"] > -20.0]
        else:
            self.cars[lane] = [c for c in self.cars[lane] if c["x"] < 20.0]

    # -----------------------
    # Collision logic
    # -----------------------
    def _check_collision(self) -> bool:
        """Check whether any car crossed x=0 while the pedestrian is exposed."""
        if self.ped_stage == self.STAGE_CROSS_LANE1:
            # lane 0: cars move from + towards 0 (right to left)
            return any((c.get("x_prev", c["x"]) > 0.0) and (c["x"] <= 0.0) for c in self.cars[0])

        if self.ped_stage == self.STAGE_CROSS_LANE2:
            # lane 1: cars move from - towards 0 (left to right)
            return any((c.get("x_prev", c["x"]) < 0.0) and (c["x"] >= 0.0) for c in self.cars[1])

        return False
