# reinforce.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


@dataclass
class ReinforceConfig:
    """Hyperparameters for the REINFORCE training loop."""
    gamma: float = 0.99              # Discount factor for Monte-Carlo returns
    lr: float = 3e-4                 # Adam learning rate
    hidden: int = 64                # Policy MLP width
    use_baseline: bool = True        # Subtract mean return for variance reduction
    entropy_beta: float = 0.02       # Exploration bonus weight (annealed in train.py)
    grad_clip_norm: float = 1.0      # Global-norm clip to prevent exploding updates

    # Advanced: accumulate gradients over multiple episodes (reduces variance)
    episodes_per_update: int = 1     # Set to 3-5 for more stable updates    

class PolicyNet(nn.Module):
    """Minimal MLP policy with LayerNorm front-end for stable scaling."""
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )
        self.onpolicy_reset()
        self.train()

    def onpolicy_reset(self) -> None:
        self.log_probs: List[torch.Tensor] = []
        self.rewards: List[float] = []
        self.entropies: List[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def act(self, state: np.ndarray) -> int:
        x = torch.as_tensor(state, dtype=torch.float32)
        logits = self.forward(x)
        dist = Categorical(logits=logits)
        a = dist.sample()
        self.log_probs.append(dist.log_prob(a))
        self.entropies.append(dist.entropy())
        return int(a.item())                    # Vorlesung  Seite 180.




def compute_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    T = len(rewards)
    rets = torch.empty(T, dtype=torch.float32)
    future = 0.0
    for t in reversed(range(T)):
        future = float(rewards[t]) + gamma * future
        rets[t] = future
    return rets


def reinforce_update(pi: PolicyNet, optimizer: optim.Optimizer, cfg: ReinforceConfig) -> dict:
    """
    REINFORCE update with entropy regularization.
    Uses mean-reduction so updates are not dependent on episode length.
    
    Returns:
        dict with 'loss', 'pg_loss', 'entropy' for better monitoring
    """
    if len(pi.rewards) == 0:
        return {'loss': 0.0, 'pg_loss': 0.0, 'entropy': 0.0}

    returns = compute_returns(pi.rewards, cfg.gamma)  # Monte-Carlo estimate (one full episode)

    # Baseline reduces variance 
    if cfg.use_baseline:
        returns = returns - returns.mean()  # Centering keeps gradients unbiased but steadier

    log_probs = torch.stack(pi.log_probs)      # [T]
    entropies = torch.stack(pi.entropies)      # [T]

    pg_loss = -(log_probs * returns).mean()
    entropy_bonus = cfg.entropy_beta * entropies.mean()
    loss = pg_loss - entropy_bonus

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(pi.parameters(), cfg.grad_clip_norm)
        # grad_norm is unused later; side effect is the clipped gradients in-place
    optimizer.step()

    return {
        'loss': float(loss.item()),
        'pg_loss': float(pg_loss.item()),
        'entropy': float(entropies.mean().item())
    }
