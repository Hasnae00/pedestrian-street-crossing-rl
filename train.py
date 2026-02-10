from __future__ import annotations
from typing import Dict, List
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from env import EnvParams, Scenario, PedestrianCrossingEnv
from reinforce import PolicyNet, ReinforceConfig, reinforce_update


ENABLE_TIMING_DEBUG = True    # quick toggle to include/exclude episode time from printouts 

# Entropy schedule (linear decay makes tuning easier)
ENTROPY_START = 0.04
ENTROPY_END = 0.02
ENTROPY_DECAY_STEPS = 30000

# Utils
def make_env(seed: int, scenario: Scenario) -> PedestrianCrossingEnv:
    return PedestrianCrossingEnv(params=EnvParams(), scenario=scenario, seed=seed)


def preprocess_state(s: np.ndarray) -> np.ndarray:
    """
    State: [tta1_rel, tta2_rel, ped_stage, stage_progress, lane1_active, lane2_active]
    TTAs are already normalized by the environment; clipping just keeps them bounded.
    """
    s = np.asarray(s, dtype=np.float32).copy()
    s[0] = np.clip(s[0], 0.0, 6.0)
    s[1] = np.clip(s[1], 0.0, 6.0)
    s[2] = np.clip(s[2], 0.0, 3.0)
    s[3] = np.clip(s[3], 0.0, 1.0)
    s[4] = np.clip(s[4], 0.0, 1.0)
    s[5] = np.clip(s[5], 0.0, 1.0)
    return s


def rolling_mean(x: List[float], window: int = 100) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if len(x) < window:
        return x
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, kernel, mode="valid")


def entropy_beta_at_episode(epi: int) -> float:
    """Linear entropy decay from ENTROPY_START to ENTROPY_END over ENTROPY_DECAY_STEPS."""
    if ENTROPY_DECAY_STEPS <= 0:
        return ENTROPY_END
    if epi <= ENTROPY_DECAY_STEPS:
        frac = epi / float(ENTROPY_DECAY_STEPS)
        return float(ENTROPY_START - frac * (ENTROPY_START - ENTROPY_END))
    return float(ENTROPY_END)


def _gap_stats(lam: float, params: EnvParams) -> Dict[str, float]:
    """Return mean gap and P(gap >= safe_threshold) for a Poisson lane."""
    lam = max(float(lam), 1e-9)
    min_gap = float(params.min_headway_s)
    safe_gap = float(params.t_lane_s + params.safety_margin_s + params.dt)
    mean_gap = min_gap + 1.0 / lam
    delta = max(safe_gap - min_gap, 0.0)
    safe_prob = float(np.exp(-lam * delta))
    return {
        "mean_gap": mean_gap,
        "safe_prob": safe_prob,
        "safe_gap": safe_gap,
    }


def log_env_diagnostics(scenarios: Dict[str, Scenario], params: EnvParams) -> None:
    """Quick human-readable summary of speeds and safe gaps per scenario."""
    ped_speed = (params.spawn_distance_m / 2.0) / params.t_lane_s if params.t_lane_s > 0 else 0.0
    mean_speed_kmh = params.v_mean_mps * 3.6
    min_speed_kmh = params.v_min_mps * 3.6
    max_speed_kmh = params.v_max_mps * 3.6
    safe_gap = params.t_lane_s + params.safety_margin_s + params.dt

    print("\n=== Environment diagnostics ===")
    print(f"Cars drive between {min_speed_kmh:.1f} and {max_speed_kmh:.1f} km/h (mean {mean_speed_kmh:.1f})."
    )
    print(f"Minimum safe time window per lane: {safe_gap:.2f}s (walk + margin).\n")

    for name, scen in scenarios.items():
        print(f"Scenario '{name}':")
        for lane_idx, lam in enumerate((scen.lambda_lane1, scen.lambda_lane2), start=1):
            stats = _gap_stats(lam, params)
            print(
                f"  Lane {lane_idx}: avg gap {stats['mean_gap']:.2f}s | "
                f"P(gap >= safe) {stats['safe_prob']*100:5.1f}% for safe gap {stats['safe_gap']:.2f}s"
            )
        print()
    print("===============================\n")

def pick_scenario_name(epi: int, rng: np.random.Generator) -> str:
    """
        Hand-tuned curriculum for increasingly dense traffic:
            - eps   1- 3000: mostly low traffic (learn basic timing)
            - eps 3001-10000: introduce medium/high but keep low dominant
            - eps 10001-20000: focus on medium with steady high exposure
            - eps 20001+: balanced mix with 25% high traffic

    REINFORCE needs LOTS of episodes for hard scenarios!
    """

    if epi <= 5000:
        return rng.choice(["low", "medium", "high"], p=[0.25, 0.60, 0.15])
    elif epi <= 10000:
        return rng.choice(["low", "medium", "high"], p=[0.15, 0.60, 0.25])
    elif epi <= 20000:
        return rng.choice(["low", "medium", "high"], p=[0.10, 0.55, 0.35])
    else:
        return rng.choice(["low", "medium", "high"], p=[0.10, 0.40, 0.50])
    

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
    wait_count = 0
    wait_low_tta_count = 0

    def is_decision_stage(stage: int) -> bool:
        return stage in (env.STAGE_CURB, env.STAGE_MEDIAN)

    # DON'T reset rewards here - let batched updates accumulate
    # pi.rewards will be reset in onpolicy_reset() after update

    while (not done) and steps < max_steps:
        stage = int(env.ped_stage)

        if is_decision_stage(stage):
            active_lane = 0 if stage == env.STAGE_CURB else 1
            lane_tta = env._tta_lane(active_lane)
            action = pi.act(state)  # stores log_prob + entropy
            if action == env.ACTION_GO:
                go_count += 1
            else:
                wait_count += 1
                if lane_tta < 1.0:
                    wait_low_tta_count += 1

            # start bucket for this decision
            pi.rewards.append(0.0)

            next_state, reward, done, info = env.step(action)
            pi.rewards[-1] += float(reward)

        else:
            # crossing -> forced WAIT
            next_state, reward, done, info = env.step(env.ACTION_WAIT)

            # Add reward to last decision bucket
            # (Should always have at least one decision before crossing)
            if len(pi.rewards) > 0:
                pi.rewards[-1] += float(reward)
            # If no rewards yet, this reward will be lost, but that's fine
            # (first steps before first decision don't matter for learning)

        total_reward += float(reward)
        state = preprocess_state(next_state)
        steps += 1
        terminal_event = info.get("terminal_event", None)

    episode_time = float(env.t)
    return float(total_reward), int(steps), terminal_event, int(go_count), episode_time, int(wait_count), int(wait_low_tta_count)

@torch.no_grad()
def eval_policy(
    env: PedestrianCrossingEnv,
    pi: PolicyNet,
    episodes: int = 5000,     # Default value for robust evaluation; can be reduced for quick checks
    temperature: float = 1.0,
):
    """
    Evaluation with same masking:
    - decision only at CURB/MEDIAN
    - force WAIT during crossing
    """
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

# Plotting
def save_training_plots(log: Dict[str, List], window: int, save_path: str):
    """Save training plots to a file."""
    if len(log["episode"]) < window:
        return
    ep = np.asarray(log["episode"], dtype=np.int32)
    ret_s  = rolling_mean(log["return"], window)
    succ_s = rolling_mean(log["success"], window)
    col_s  = rolling_mean(log["collision"], window)
    tout_s = rolling_mean(log["timeout"], window)
    epi_time_s = rolling_mean(log["episode_time"], window)
    wait_s = rolling_mean(log["wait"], window)
    wait_low_s = rolling_mean(log["wait_low_tta"], window)
    loss_s = rolling_mean(log["loss"], window)
    ep_s = ep[: len(ret_s)]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(f"Training Progress - Episode {log['episode'][-1]}", fontsize=16)
    
    # 1) Return
    axes[0, 0].plot(ep_s, ret_s)
    axes[0, 0].set_xlabel("Episode")
    axes[0, 0].set_ylabel("Average Return")
    axes[0, 0].set_title(f"All Episodes: Return ")
    axes[0, 0].grid(True)
    
    # 2) Outcome rates
    axes[0, 1].plot(ep_s, succ_s, label="Success")
    axes[0, 1].plot(ep_s, col_s, label="Collision")
    axes[0, 1].plot(ep_s, tout_s, label="Timeout")
    axes[0, 1].set_xlabel("Episode")
    axes[0, 1].set_ylabel("Rate (0-1)")
    axes[0, 1].set_title(f"All Episodes: Outcomes ")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 3) Episode duration 
    axes[0, 2].plot(ep_s, epi_time_s)
    axes[0, 2].set_xlabel("Episode")
    axes[0, 2].set_ylabel("Seconds")
    axes[0, 2].set_title("All Episodes: Episode Time")
    axes[0, 2].grid(True)
    
    # 4) Loss
    axes[1, 0].plot(ep_s, loss_s)
    axes[1, 0].set_xlabel("Episode")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title(f"All Episodes: Loss ")
    axes[1, 0].grid(True)
    
    # 5) Entropy
    axes[1, 1].plot(ep, log["entropy_beta"])
    axes[1, 1].set_xlabel("Episode")
    axes[1, 1].set_ylabel("entropy_beta")
    axes[1, 1].set_title("Entropy Schedule")
    axes[1, 1].grid(True)
    
    # 6) WAIT stats (decision behaviour)
    axes[1, 2].plot(ep_s, wait_s, label="WAIT")
    axes[1, 2].plot(ep_s, wait_low_s, label="WAIT (tta<1)")
    axes[1, 2].set_xlabel("Episode")
    axes[1, 2].set_ylabel("Avg count")
    axes[1, 2].set_title("All Episodes: WAIT usage")
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

# Main
def main():
    torch.manual_seed(0)
    np.random.seed(0)
    rng = np.random.default_rng(0)

    #  Scenario presets 
    # - low  : quiet road
    # - med  : normal traffic
    # - high : rush-hour-like
    scenarios = {
        "low":    Scenario(lambda_lane1=0.18, lambda_lane2=0.18), 
        "medium": Scenario(lambda_lane1=0.27, lambda_lane2=0.27), 
        "high":   Scenario(lambda_lane1=0.34, lambda_lane2=0.34), 
    }
                                        # low: λ = 0.18 → Ø 5.4 s
                                        # medium: λ = 0.27 → Ø 4.0 s
                                        # high: λ = 0.34 → Ø 3.5 s

    log_env_diagnostics(scenarios, EnvParams())

    # REINFORCE config 
    cfg = ReinforceConfig()
    cfg.entropy_beta = entropy_beta_at_episode(1)

    pi = PolicyNet(in_dim=6, out_dim=2, hidden=cfg.hidden)
    optimizer = optim.Adam(pi.parameters(), lr=cfg.lr)
    
    # Learning rate stays constant (default 3e-4) unless cfg is changed
    print(f"Learning Rate: {cfg.lr} (constant throughout training)")
    print(f"Policy MLP hidden layer size: {cfg.hidden}\n")

    num_episodes = 50000
    print_every = 100
    save_plot_every = 500  # Update plot image every N episodes
    
    plot_filename = "training_progress.png"  # Always same file, gets overwritten
    print(f"\n Plots will be saved to '{plot_filename}' (updated every {save_plot_every} episodes)")
    print(f"    You can open this file during training to see current progress.\n")
    
    # log everything (ALL episodes)
    log: Dict[str, List] = {
        "episode": [],
        "return": [],
        "success": [],
        "collision": [],
        "timeout": [],
        "go": [],
        "wait": [],
        "wait_low_tta": [],
        "loss": [],
        "pg_loss": [],          #  track policy gradient loss separately
        "entropy": [],          #  track actual policy entropy
        "entropy_beta": [],
        "scenario": [],
        "lr": [],               #  track learning rate
        "episode_time": [],
    }

    # rolling windows for printing
    return_window, go_window = [], []
    success_window, collision_window, timeout_window = [], [], []
    episode_time_window = []
    wait_window, wait_low_window = [], []
    
    # For batched updates
    episodes_in_batch = 0
    batch_losses = []

    #  Main training loop 
    for epi in range(1, num_episodes + 1):
        cfg.entropy_beta = entropy_beta_at_episode(epi)
        scen_name = pick_scenario_name(epi, rng)
        env = make_env(seed=epi, scenario=scenarios[scen_name])

        total_reward, steps, terminal_event, go_count, episode_time, wait_count, wait_low_tta_count = run_episode(env, pi)
        episodes_in_batch += 1

        # Batched update: accumulate gradients over multiple episodes
        if episodes_in_batch >= cfg.episodes_per_update:
            update_info = reinforce_update(pi, optimizer, cfg)
            
            loss = update_info['loss']
            pg_loss = update_info['pg_loss']
            entropy = update_info['entropy']
            
            # Reset after update
            pi.onpolicy_reset()
            episodes_in_batch = 0
        else:
            # Still accumulating, use last values for logging
            loss = batch_losses[-1] if batch_losses else 0.0
            pg_loss = 0.0
            entropy = 0.0

        #  Per-episode logging payload (ALL episodes)
        log["episode"].append(epi)
        log["return"].append(float(total_reward))
        log["success"].append(1 if terminal_event == "success" else 0)
        log["collision"].append(1 if terminal_event == "collision" else 0)
        log["timeout"].append(1 if terminal_event == "timeout" else 0)
        log["go"].append(int(go_count))
        log["wait"].append(int(wait_count))
        log["wait_low_tta"].append(int(wait_low_tta_count))
        log["loss"].append(float(loss))
        log["pg_loss"].append(float(pg_loss))
        log["entropy"].append(float(entropy))
        log["entropy_beta"].append(float(cfg.entropy_beta))
        log["scenario"].append(scen_name)
        log["lr"].append(float(optimizer.param_groups[0]['lr']))
        log["episode_time"].append(float(episode_time))
        
        batch_losses.append(float(loss))

        # rolling stats (for printing)
        return_window.append(total_reward)
        go_window.append(go_count)
        wait_window.append(wait_count)
        wait_low_window.append(wait_low_tta_count)
        success_window.append(1 if terminal_event == "success" else 0)
        collision_window.append(1 if terminal_event == "collision" else 0)
        timeout_window.append(1 if terminal_event == "timeout" else 0)
        episode_time_window.append(episode_time)

        if epi % print_every == 0:
            avg_ret = float(np.mean(return_window[-print_every:]))
            succ = float(np.mean(success_window[-print_every:]))
            col = float(np.mean(collision_window[-print_every:]))
            tout = float(np.mean(timeout_window[-print_every:]))
            avg_go = float(np.mean(go_window[-print_every:]))
            avg_wait = float(np.mean(wait_window[-print_every:]))
            avg_wait_low = float(np.mean(wait_low_window[-print_every:]))
            current_lr = optimizer.param_groups[0]['lr']
            if ENABLE_TIMING_DEBUG:
                avg_epi_time = float(np.mean(episode_time_window[-print_every:]))
            else:
                avg_epi_time = 0.0

            print(
                f"epi {epi:5d} | loss {loss:8.3f} | avg_return {avg_ret:7.3f} "
                f"| success {succ:5.2f} | collision {col:5.2f} | timeout {tout:5.2f} "
                f"| avg_GO {avg_go:6.2f} | avg_WAIT {avg_wait:6.2f} | wait_tta<1 {avg_wait_low:6.2f} "
                f"| lr {current_lr:.6f} | scen {scen_name:<6}"
                + (f" | epi_time {avg_epi_time:5.1f}s" if ENABLE_TIMING_DEBUG else "")
            )
            
        
        #  Save/update plot image periodically (always same filename) 
        if epi % save_plot_every == 0 and epi >= 200:
            save_training_plots(log, window=200, save_path=plot_filename)
            print(f"Plot updated: {plot_filename}")

    # Save final plot with timestamp to keep historical runs
    timestamped_plot = f"training_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    save_training_plots(log, window=200, save_path=timestamped_plot)
    print(f"\n Training complete! Final plot saved: {timestamped_plot}\n")

    # Final evaluation across multiple seeds for robustness
    print("\n" + "="*60)
    print("FINAL EVALUATION (multiple seeds for generalization test)")
    print("="*60)
    for name in ["low", "medium", "high"]:
        total_succ = total_col = total_tout = 0
        num_eval_seeds = 5
        episodes_per_seed = 200
        
        for seed_offset in range(num_eval_seeds):
            eval_env = make_env(seed=123 + seed_offset, scenario=scenarios[name])
            succ, col, tout = eval_policy(eval_env, pi, episodes=episodes_per_seed, temperature=1.0)
            total_succ += succ
            total_col += col
            total_tout += tout
        
        total_episodes = num_eval_seeds * episodes_per_seed
        print(f"\nScenario: {name}")
        print(f"  success  : {total_succ}/{total_episodes} ({100*total_succ/total_episodes:.1f}%)")
        print(f"  collision: {total_col}/{total_episodes} ({100*total_col/total_episodes:.1f}%)")
        print(f"  timeout  : {total_tout}/{total_episodes} ({100*total_tout/total_episodes:.1f}%)")


if __name__ == "__main__":
    main()
