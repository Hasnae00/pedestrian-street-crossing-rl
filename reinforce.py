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
    gamma: float = 0.99
    lr: float = 3e-4
    hidden: int = 64
    use_return_normalization: bool = False
    use_baseline: bool = True
    entropy_beta: float = 0.02
    grad_clip_norm: float = 1.0



class PolicyNet(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64):
        super().__init__()
        self.model = nn.Sequential(
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
        return int(a.item())

    @torch.no_grad()
    def act_greedy(self, state: np.ndarray) -> int:
        x = torch.as_tensor(state, dtype=torch.float32)
        logits = self.forward(x)
        return int(torch.argmax(logits).item())


def compute_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    T = len(rewards)
    rets = torch.empty(T, dtype=torch.float32)
    future = 0.0
    for t in reversed(range(T)):
        future = float(rewards[t]) + gamma * future
        rets[t] = future
    return rets


def reinforce_update(pi: PolicyNet, optimizer: optim.Optimizer, cfg: ReinforceConfig) -> float:
    """
    REINFORCE update with entropy regularization.
    Uses mean-reduction so updates are not dependent on episode length.
    """
    if len(pi.rewards) == 0:
        return 0.0

    returns = compute_returns(pi.rewards, cfg.gamma)

    # IMPORTANT: don't do baseline + normalization together unless you really want to.
    if cfg.use_return_normalization:
        mean = returns.mean()
        std = returns.std(unbiased=False)
        if float(std) > 1e-8:
            returns = (returns - mean) / (std + 1e-8)
        else:
            returns = returns - mean
    elif cfg.use_baseline:
        returns = returns - returns.mean()

    log_probs = torch.stack(pi.log_probs)      # [T]
    entropies = torch.stack(pi.entropies)      # [T]

    pg_loss = -(log_probs * returns).mean()
    entropy_bonus = cfg.entropy_beta * entropies.mean()
    loss = pg_loss - entropy_bonus

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
        torch.nn.utils.clip_grad_norm_(pi.parameters(), cfg.grad_clip_norm)
    optimizer.step()

    return float(loss.item())
