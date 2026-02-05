# train.py
from __future__ import annotations
from typing import Dict, List
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from env import EnvParams, Scenario, PedestrianCrossingEnv
from reinforce import PolicyNet, ReinforceConfig, reinforce_update

# OPTIONAL VISUALIZATION (DEMO)
ENABLE_DEMO = False
DEMO_EVERY = 500
DEMO_SCENARIO = "low"        # "low" / "medium" / "high"
DEMO_TEMPERATURE = 0.7       # 1.0 more stochastic, 0.25 more decisive

if ENABLE_DEMO:
    from visualize import visualize_step


# Utils
def make_env(seed: int, scenario: Scenario) -> PedestrianCrossingEnv:
    return PedestrianCrossingEnv(params=EnvParams(), scenario=scenario, seed=seed)


def preprocess_state(s: np.ndarray) -> np.ndarray:
    """
    State: [tta1, tta2, ped_stage, stage_progress]
    We normalize/clamp to avoid huge TTAs (999) making logits explode.
    """
    s = np.asarray(s, dtype=np.float32).copy()
    s[0] = np.clip(s[0] / 30.0, 0.0, 1.0)   # tta1
    s[1] = np.clip(s[1] / 30.0, 0.0, 1.0)   # tta2
    s[2] = np.clip(s[2] / 3.0, 0.0, 1.0)    # stage 0..3 -> 0..1
    s[3] = np.clip(s[3], 0.0, 1.0)          # progress
    return s


def rolling_mean(x: List[float], window: int = 100) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) < window:
        return x
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="valid")

def pick_scenario_name(epi: int, rng: np.random.Generator) -> str:
    """
    Curriculum (stable for REINFORCE):
      - first learn in low
      - then mix low+medium
      - then mix all (incl high)

    You can change probabilities, but DON'T start with a lot of high early.
    """
    if epi <= 5000:
        return "low"
    elif epi <= 12000:
        return rng.choice(["low", "medium"], p=[0.4, 0.6])
    else:
        return rng.choice(["low", "medium", "high"], p=[0.2, 0.5, 0.3])

# Rollouts
def run_episode(env: PedestrianCrossingEnv, pi: PolicyNet, max_steps: int = 2000):
    """
    Decision-point / action-masking rollout:

    - In CURB or MEDIAN: sample action from policy (store log_prob, entropy)
    - In CROSS_LANE1/CROSS_LANE2: force WAIT (do NOT store log_prob)
    - Rewards are aggregated per decision so REINFORCE stays consistent:
        len(pi.log_probs) == len(pi.rewards)
    """
    state = preprocess_state(env.reset())
    done = False
    steps = 0

    total_reward = 0.0
    terminal_event = None
    go_count = 0

    def is_decision_stage(stage: int) -> bool:
        return stage in (env.STAGE_CURB, env.STAGE_MEDIAN)

    # bucket rewards per decision
    pi.rewards = []

    while (not done) and steps < max_steps:
        stage = int(env.ped_stage)

        if is_decision_stage(stage):
            action = pi.act(state)  # stores log_prob + entropy
            if action == env.ACTION_GO:
                go_count += 1

            # start bucket for this decision
            pi.rewards.append(0.0)

            next_state, reward, done, info = env.step(action)
            pi.rewards[-1] += float(reward)
        else:
            # crossing -> forced WAIT
            next_state, reward, done, info = env.step(env.ACTION_WAIT)

            # still add reward to last decision bucket
            if len(pi.rewards) == 0:
                pi.rewards.append(0.0)
            pi.rewards[-1] += float(reward)

        total_reward += float(reward)
        state = preprocess_state(next_state)
        steps += 1
        terminal_event = info.get("terminal_event", None)

    return float(total_reward), int(steps), terminal_event, int(go_count)

@torch.no_grad()
def eval_policy(
    env: PedestrianCrossingEnv,
    pi: PolicyNet,
    episodes: int = 200,
    mode: str = "sampled",      # "sampled" / "greedy" / "threshold"
    temperature: float = 1.0,
    threshold: float = 0.5,
):
    """
    Evaluation with same masking:
    - decision only at CURB/MEDIAN
    - force WAIT during crossing
    """
    assert temperature > 0.0
    succ = col = tout = 0

    def is_decision_stage(stage: int) -> bool:
        return stage in (env.STAGE_CURB, env.STAGE_MEDIAN)

    for _ in range(episodes):
        s = preprocess_state(env.reset())
        done = False
        terminal_event = None
        steps = 0

        while not done and steps < 2000:
            if is_decision_stage(int(env.ped_stage)):
                x = torch.as_tensor(s, dtype=torch.float32)
                logits = pi.forward(x) / temperature
                dist = Categorical(logits=logits)

                if mode == "greedy":
                    a = int(torch.argmax(logits).item())
                elif mode == "threshold":
                    p_go = float(dist.probs[env.ACTION_GO].item())
                    a = env.ACTION_GO if p_go > threshold else env.ACTION_WAIT
                else:
                    a = int(dist.sample().item())
            else:
                a = env.ACTION_WAIT

            s, _, done, info = env.step(a)
            s = preprocess_state(s)
            terminal_event = info.get("terminal_event", None)
            steps += 1

        succ += 1 if terminal_event == "success" else 0
        col += 1 if terminal_event == "collision" else 0
        tout += 1 if terminal_event == "timeout" else 0

    return succ, col, tout

# Demo
def demo_rollout(env: PedestrianCrossingEnv, pi: PolicyNet, max_steps: int = 2000, temperature: float = 1.0):
    if not ENABLE_DEMO:
        return

    plt.ion()
    fig = plt.figure(figsize=(12, 4))

    env.reset()
    done = False
    steps = 0

    while not done and steps < max_steps:
        s = preprocess_state(env._get_state())
        x = torch.as_tensor(s, dtype=torch.float32)
        logits = pi.forward(x) / temperature
        dist = Categorical(logits=logits)
        a = int(dist.sample().item())

        _, _, done, _ = env.step(a)
        visualize_step(env.render_data(), spawn_distance=env.params.spawn_distance_m)
        steps += 1

    plt.ioff()
    plt.show()
    plt.close(fig)

# Plotting
def plot_training_curves(log: Dict[str, List], window: int = 200):
    ep = np.asarray(log["episode"], dtype=np.int32)

    ret_s  = rolling_mean(log["return"], window)
    succ_s = rolling_mean(log["success"], window)
    col_s  = rolling_mean(log["collision"], window)
    tout_s = rolling_mean(log["timeout"], window)
    go_s   = rolling_mean(log["go"], window)
    loss_s = rolling_mean(log["loss"], window)

    ep_s = ep[: len(ret_s)]

    # 1) Return
    plt.figure(figsize=(10, 4))
    plt.plot(ep_s, ret_s)
    plt.xlabel("Episode")
    plt.ylabel(f"Avg Return ({window})")
    plt.title("Learning Curve: Return")
    plt.grid(True)
    plt.tight_layout()

    # 2) Outcome rates
    plt.figure(figsize=(10, 4))
    plt.plot(ep_s, succ_s, label="Success")
    plt.plot(ep_s, col_s, label="Collision")
    plt.plot(ep_s, tout_s, label="Timeout")
    plt.xlabel("Episode")
    plt.ylabel(f"Rate ({window})")
    plt.title("Outcome Rates (Rolling Avg)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 3) GO frequency
    plt.figure(figsize=(10, 4))
    plt.plot(ep_s, go_s)
    plt.xlabel("Episode")
    plt.ylabel(f"Avg GO ({window})")
    plt.title("Policy Behaviour: GO frequency")
    plt.grid(True)
    plt.tight_layout()

    # 4) Loss
    plt.figure(figsize=(10, 4))
    plt.plot(ep_s, loss_s)
    plt.xlabel("Episode")
    plt.ylabel(f"Loss ({window})")
    plt.title("Training Loss (Rolling Avg)")
    plt.grid(True)
    plt.tight_layout()

    # 5) Entropy schedule
    plt.figure(figsize=(10, 4))
    plt.plot(ep, log["entropy_beta"])
    plt.xlabel("Episode")
    plt.ylabel("entropy_beta")
    plt.title("Entropy Annealing Schedule")
    plt.grid(True)
    plt.tight_layout()

    plt.show()

# Main
def main():
    torch.manual_seed(0)
    np.random.seed(0)
    rng = np.random.default_rng(0)

    # traffic intensity:
    # - low  : quiet road
    # - med  : normal traffic
    # - high : rush-hour-like
    scenarios = {
        "low":    Scenario(lambda_lane1=0.12, lambda_lane2=0.12),
        "medium": Scenario(lambda_lane1=0.30, lambda_lane2=0.30),
        "high":   Scenario(lambda_lane1=0.50, lambda_lane2=0.50),
    }

    # --- REINFORCE config ---
    cfg = ReinforceConfig(
        gamma=0.99,
        lr=1e-3,
        hidden=64,
        use_return_normalization=True,
        use_baseline=True,
        entropy_beta=0.02,
        grad_clip_norm=1.0,
    )

    pi = PolicyNet(in_dim=4, out_dim=2, hidden=cfg.hidden)
    optimizer = optim.Adam(pi.parameters(), lr=cfg.lr)

    num_episodes = 20000
    print_every = 100

    # log everything (ALL episodes)
    log: Dict[str, List] = {
        "episode": [],
        "return": [],
        "success": [],
        "collision": [],
        "timeout": [],
        "go": [],
        "loss": [],
        "entropy_beta": [],
        "scenario": [],
    }

    # rolling windows for printing
    return_window, go_window = [], []
    success_window, collision_window, timeout_window = [], [], []

    for epi in range(1, num_episodes + 1):
        scen_name = pick_scenario_name(epi, rng)
        env = make_env(seed=epi, scenario=scenarios[scen_name])

        total_reward, steps, terminal_event, go_count = run_episode(env, pi)

        loss = reinforce_update(pi, optimizer, cfg)
        pi.onpolicy_reset()

        # log per-episode
        log["episode"].append(epi)
        log["return"].append(float(total_reward))
        log["success"].append(1 if terminal_event == "success" else 0)
        log["collision"].append(1 if terminal_event == "collision" else 0)
        log["timeout"].append(1 if terminal_event == "timeout" else 0)
        log["go"].append(int(go_count))
        log["loss"].append(float(loss))
        log["entropy_beta"].append(float(cfg.entropy_beta))
        log["scenario"].append(scen_name)

        # rolling stats (for printing)
        return_window.append(total_reward)
        go_window.append(go_count)
        success_window.append(1 if terminal_event == "success" else 0)
        collision_window.append(1 if terminal_event == "collision" else 0)
        timeout_window.append(1 if terminal_event == "timeout" else 0)

        # entropy annealing (slower + higher minimum to avoid "always WAIT")
        cfg.entropy_beta = max(0.01, cfg.entropy_beta * 0.9995)

        if epi % print_every == 0:
            avg_ret = float(np.mean(return_window[-print_every:]))
            succ = float(np.mean(success_window[-print_every:]))
            col = float(np.mean(collision_window[-print_every:]))
            tout = float(np.mean(timeout_window[-print_every:]))
            avg_go = float(np.mean(go_window[-print_every:]))

            print(
                f"epi {epi:5d} | loss {loss:8.3f} | avg_return {avg_ret:7.3f} "
                f"| success {succ:5.2f} | collision {col:5.2f} | timeout {tout:5.2f} "
                f"| avg_GO {avg_go:6.2f} | entropy_beta {cfg.entropy_beta:.5f} | scen {scen_name}"
            )

        # demo occasionally
        if ENABLE_DEMO and (epi % DEMO_EVERY == 0):
            demo_env = make_env(seed=999, scenario=scenarios[DEMO_SCENARIO])
            demo_rollout(demo_env, pi, temperature=DEMO_TEMPERATURE)

    # plots
    plot_training_curves(log, window=200)

    # evaluation
    for name in ["low", "medium", "high"]:
        eval_env = make_env(seed=123, scenario=scenarios[name])

        succ_s, col_s, tout_s = eval_policy(eval_env, pi, episodes=200, mode="sampled", temperature=1.0)
        succ_g, col_g, tout_g = eval_policy(eval_env, pi, episodes=200, mode="greedy", temperature=0.25)
        succ_t, col_t, tout_t = eval_policy(eval_env, pi, episodes=200, mode="threshold", temperature=0.25, threshold=0.35)

        print(f"\n=== Scenario: {name} ===")
        print("Evaluation (sampled, T=1.0):")
        print(f"  success  : {succ_s}/200")
        print(f"  collision: {col_s}/200")
        print(f"  timeout  : {tout_s}/200")

        print("Evaluation (greedy, T=0.25):")
        print(f"  success  : {succ_g}/200")
        print(f"  collision: {col_g}/200")
        print(f"  timeout  : {tout_g}/200")

        print("Evaluation (threshold, T=0.25, p_go>0.35):")
        print(f"  success  : {succ_t}/200")
        print(f"  collision: {col_t}/200")
        print(f"  timeout  : {tout_t}/200")


if __name__ == "__main__":
    main()
