from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np


@dataclass
class EnvParams:
    # Simulation
    dt: float = 0.2
    episode_time_limit: float = 90.0

    # Geometry / timing
    spawn_distance_m: float = 120.0

    # Pedestrian: time to cross ONE lane (older pedestrian)
    t_lane_s: float = 4.5

    # Vehicle speed model (50–55 km/h typical)
    v_mean_mps: float = 52.5 / 3.6   # ≈ 14.58 m/s
    v_std_mps: float = 1.0
    v_min_mps: float = 45.0 / 3.6   # ≈ 12.50 m/s
    v_max_mps: float = 60.0 / 3.6   # ≈ 16.67 m/s

    # Traffic arrivals
    min_headway_s: float = 0.8
    warmup_time_s: float = 10.0

    # Visual/realism: prevent cars overlapping in same lane (1D no-overtake)
    min_car_spacing_m: float = 12.0

    # Reward
    r_collision: float = -12.0
    r_success: float = +15.0
    r_step: float = -0.05
    r_timeout: float = -12.0


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

    # def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
    #     action = int(action)
    #     done = False
    #     terminal_event = None

    #     # time always passes -> apply step penalty every step
    #     reward = float(self.params.r_step)

    #     # 1) Apply action (only meaningful at CURB or MEDIAN)
    #     if action == self.ACTION_GO:
    #         if self.ped_stage == self.STAGE_CURB:
    #             self.ped_stage = self.STAGE_CROSS_LANE1
    #             self.t_in_stage = 0.0
    #         elif self.ped_stage == self.STAGE_MEDIAN:
    #             self.ped_stage = self.STAGE_CROSS_LANE2
    #             self.t_in_stage = 0.0
    #         # else: already crossing -> no-op

    #     # 2) Advance traffic
    #     self._advance_lane(0, self.scenario.lambda_lane1)
    #     self._advance_lane(1, self.scenario.lambda_lane2)

    #     # 3) Advance time and crossing timer
    #     self.t += self.params.dt
    #     if self.ped_stage in (self.STAGE_CROSS_LANE1, self.STAGE_CROSS_LANE2):
    #         self.t_in_stage += self.params.dt

    #     # 4) Collision (only when exposed)
    #     if self._check_collision():
    #         reward = float(self.params.r_collision)  # override
    #         done = True
    #         terminal_event = "collision"

    #     # 5) If safe: stage transitions / success
    #     if not done:
    #         if self.ped_stage == self.STAGE_CROSS_LANE1 and self.t_in_stage >= self.params.t_lane_s:
    #             self.ped_stage = self.STAGE_MEDIAN
    #             self.t_in_stage = 0.0

    #         elif self.ped_stage == self.STAGE_CROSS_LANE2 and self.t_in_stage >= self.params.t_lane_s:
    #             reward += float(self.params.r_success)
    #             done = True
    #             terminal_event = "success"

    #     # 6) Timeout (only if still not done)
    #     if (not done) and (self.t >= self.params.episode_time_limit):
    #         reward += float(self.params.r_timeout)
    #         done = True
    #         terminal_event = "timeout"

    #     state = self._get_state()
    #     info = {
    #         "t": float(self.t),
    #         "tta_lane1": float(self._tta_lane(0)),
    #         "tta_lane2": float(self._tta_lane(1)),
    #         "ped_stage": int(self.ped_stage),
    #         "stage_progress": float(self._stage_progress()),
    #         "n_cars_lane1": int(len(self.cars[0])),
    #         "n_cars_lane2": int(len(self.cars[1])),
    #         "terminal_event": terminal_event,
    #     }
    #     return state, float(reward), bool(done), info
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        action = int(action)
        done = False
        terminal_event = None

        # time always passes -> step penalty
        reward = float(self.params.r_step)

        # ----------------------------
        # Stronger decision-point shaping
        # ----------------------------
        tau = 0.8               # bigger safety margin -> less risky crossings
        r_safe_go = 1.0         # reward if GO is taken in a safe gap
        r_unsafe_go = -2        # stronger penalty if GO is taken in an unsafe gap
        r_reach_median = 2.0    # subgoal reward after crossing lane 1
        r_spam_go = -0.02       # discourage pressing GO while already crossing

        # 1) Apply action (meaningful only at CURB or MEDIAN)
        if action == self.ACTION_GO:
            if self.ped_stage == self.STAGE_CURB:
                tta1 = self._tta_lane(0)
                safe = (tta1 > (self.params.t_lane_s + tau))
                reward += r_safe_go if safe else r_unsafe_go

                self.ped_stage = self.STAGE_CROSS_LANE1
                self.t_in_stage = 0.0

            elif self.ped_stage == self.STAGE_MEDIAN:
                tta2 = self._tta_lane(1)
                safe = (tta2 > (self.params.t_lane_s + tau))
                reward += r_safe_go if safe else r_unsafe_go

                self.ped_stage = self.STAGE_CROSS_LANE2
                self.t_in_stage = 0.0

            else:
                reward += r_spam_go

        # 2) Advance traffic
        self._advance_lane(0, self.scenario.lambda_lane1)
        self._advance_lane(1, self.scenario.lambda_lane2)

        # 3) Advance time and crossing timer
        self.t += self.params.dt
        if self.ped_stage in (self.STAGE_CROSS_LANE1, self.STAGE_CROSS_LANE2):
            self.t_in_stage += self.params.dt

        # 4) Collision (only when exposed)
        if self._check_collision():
            reward = float(self.params.r_collision)  # override
            done = True
            terminal_event = "collision"

        # 5) Stage transitions / success
        if not done:
            if self.ped_stage == self.STAGE_CROSS_LANE1 and self.t_in_stage >= self.params.t_lane_s:
                self.ped_stage = self.STAGE_MEDIAN
                self.t_in_stage = 0.0
                reward += r_reach_median

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

    # def _get_state(self) -> np.ndarray:
    #     tta1 = self._tta_lane(0)
    #     tta2 = self._tta_lane(1)
    #     return np.array([tta1, tta2, float(self.ped_stage), self._stage_progress()], dtype=np.float32)
    # def _get_state(self) -> np.ndarray:
    #     tta1 = self._tta_lane(0)
    #     tta2 = self._tta_lane(1)

    #     # Only observe the relevant direction:
    #     # - CURB / crossing lane1: lane 0 relevant
    #     # - MEDIAN / crossing lane2: lane 1 relevant
    #     if self.ped_stage in (self.STAGE_CURB, self.STAGE_CROSS_LANE1):
    #         tta2 = 999.0
    #     elif self.ped_stage in (self.STAGE_MEDIAN, self.STAGE_CROSS_LANE2):
    #         tta1 = 999.0

    #     return np.array([tta1, tta2, float(self.ped_stage), self._stage_progress()], dtype=np.float32)
    def _get_state(self) -> np.ndarray:
        tta1 = self._tta_lane(0)
        tta2 = self._tta_lane(1)

        # Clip TTAs (prevents huge values)
        tta1 = min(tta1, 30.0)
        tta2 = min(tta2, 30.0)

        # Mask irrelevant lane using same scale
        if self.ped_stage in (self.STAGE_CURB, self.STAGE_CROSS_LANE1):
            tta2 = 30.0
        elif self.ped_stage in (self.STAGE_MEDIAN, self.STAGE_CROSS_LANE2):
            tta1 = 30.0

        return np.array([tta1, tta2, float(self.ped_stage), self._stage_progress()], dtype=np.float32)


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
        # spawn
        self._t_next_arrival[lane] -= self.params.dt
        while self._t_next_arrival[lane] <= 0.0:
            self._spawn_car(lane)
            self._t_next_arrival[lane] += self._sample_interarrival(lam)

        # move
        for c in self.cars[lane]:
            c["x_prev"] = c["x"]  
            if lane == 0:
                c["x"] -= c["v"] * self.params.dt
            else:
                c["x"] += c["v"] * self.params.dt


        # keep spacing (helps realism + visuals)
        #self._enforce_min_spacing(lane)

        # cleanup (cars far past crossing line)
        if lane == 0:
            self.cars[lane] = [c for c in self.cars[lane] if c["x"] > -20.0]
        else:
            self.cars[lane] = [c for c in self.cars[lane] if c["x"] < 20.0]

    # -----------------------
    # Collision logic
    # -----------------------
    # def _check_collision(self) -> bool:
    #     if self.ped_stage == self.STAGE_CROSS_LANE1:
    #         return any(c["x"] <= 0.0 for c in self.cars[0])
    #     if self.ped_stage == self.STAGE_CROSS_LANE2:
    #         return any(c["x"] >= 0.0 for c in self.cars[1])
    #     return False
    def _check_collision(self) -> bool:
        if self.ped_stage == self.STAGE_CROSS_LANE1:
            # lane 0 crosses from + to 0
            return any((c.get("x_prev", c["x"]) > 0.0) and (c["x"] <= 0.0) for c in self.cars[0])

        if self.ped_stage == self.STAGE_CROSS_LANE2:
            # lane 1 crosses from - to 0
            return any((c.get("x_prev", c["x"]) < 0.0) and (c["x"] >= 0.0) for c in self.cars[1])

        return False


    # -----------------------
    # Rendering
    # -----------------------
    def render_data(self):
        return {
            "cars_lane1": [(c["x"], c["v"]) for c in self.cars[0]],
            "cars_lane2": [(c["x"], c["v"]) for c in self.cars[1]],
            "ped_stage": int(self.ped_stage),
            "stage_progress": float(self._stage_progress()),
            "t": float(self.t),
        }
