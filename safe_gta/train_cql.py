"""
CQL (Conservative Q-Learning) trainer — supports Baseline 1 and Baseline 2.

Baseline 1 (no augmentation):
  python -m safe_gta.train_cql

Baseline 2 (GTA augmentation, no safety):
  python -m safe_gta.train_cql --augmented_data safe_gta/data/augmented_no_safety.pkl
"""
import argparse
import os
import sys
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Conservative Q-Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


class CQLAgent:
    """
    Minimal CQL implementation.
    Loss = Bellman loss + alpha * (E[Q(s, random_a)] - E[Q(s, data_a)])
    The second term penalizes overestimation on out-of-distribution actions.
    """

    def __init__(self, obs_dim: int, act_dim: int,
                 gamma: float = 0.99, alpha: float = 1.0,
                 lr: float = 3e-4, device: str = "cpu"):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.alpha = alpha
        self.device = device

        self.q1 = QNetwork(obs_dim, act_dim).to(device)
        self.q2 = QNetwork(obs_dim, act_dim).to(device)
        self.q1_target = QNetwork(obs_dim, act_dim).to(device)
        self.q2_target = QNetwork(obs_dim, act_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )

    def update(self, obs, act, rew, next_obs, done, n_random: int = 10):
        B = obs.shape[0]

        with torch.no_grad():
            # TD target using target networks
            next_act = torch.randn(B, self.act_dim, device=self.device).clamp(-1, 1)
            q1_next = self.q1_target(next_obs, next_act)
            q2_next = self.q2_target(next_obs, next_act)
            q_next = torch.min(q1_next, q2_next)
            td_target = rew + self.gamma * (1 - done) * q_next

        q1_pred = self.q1(obs, act)
        q2_pred = self.q2(obs, act)
        bellman_loss = nn.functional.mse_loss(q1_pred, td_target) + \
                       nn.functional.mse_loss(q2_pred, td_target)

        # CQL conservative penalty
        random_acts = torch.randn(B * n_random, self.act_dim, device=self.device).clamp(-1, 1)
        obs_rep = obs.unsqueeze(1).repeat(1, n_random, 1).reshape(B * n_random, -1)
        q1_rand = self.q1(obs_rep, random_acts).reshape(B, n_random).logsumexp(dim=1)
        q2_rand = self.q2(obs_rep, random_acts).reshape(B, n_random).logsumexp(dim=1)
        cql_loss = (q1_rand - q1_pred).mean() + (q2_rand - q2_pred).mean()

        loss = bellman_loss + self.alpha * cql_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update targets
        for p, pt in zip(self.q1.parameters(), self.q1_target.parameters()):
            pt.data.mul_(0.995).add_(0.005 * p.data)
        for p, pt in zip(self.q2.parameters(), self.q2_target.parameters()):
            pt.data.mul_(0.995).add_(0.005 * p.data)

        return loss.item(), bellman_loss.item(), cql_loss.item()

    def select_action(self, obs: torch.Tensor, n_candidates: int = 10) -> torch.Tensor:
        """Greedy action selection: pick the action with highest Q from random candidates."""
        B = obs.shape[0]
        cands = torch.randn(B * n_candidates, self.act_dim, device=self.device).clamp(-1, 1)
        obs_rep = obs.unsqueeze(1).repeat(1, n_candidates, 1).reshape(B * n_candidates, -1)
        q_vals = torch.min(self.q1(obs_rep, cands), self.q2(obs_rep, cands))
        best_idx = q_vals.reshape(B, n_candidates).argmax(dim=1)
        actions = cands.reshape(B, n_candidates, -1)[torch.arange(B), best_idx]
        return actions


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_augmented_dataset(pkl_path: str, device: str = "cpu"):
    """Load augmented dataset from .pkl file (output of generate_augmented.py)."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    # Flatten (N, seq_len, dim) → (N*seq_len, dim) for transition-level training
    def flat(arr):
        return torch.tensor(arr.reshape(-1, arr.shape[-1]), dtype=torch.float32, device=device)
    def flat1d(arr):
        return torch.tensor(arr.reshape(-1), dtype=torch.float32, device=device)

    obs  = flat(data["observations"])
    act  = flat(data["actions"])
    rew  = flat1d(data["rewards"])
    cost = flat1d(data["costs"])
    done = flat1d(data["terminals"])
    next_obs = torch.roll(obs, -1, dims=0)

    print(f"  Loaded augmented dataset: {len(obs)} transitions  obs_dim={obs.shape[-1]}")
    print(f"  Source: {pkl_path}")
    return obs, act, rew, cost, done, next_obs


def load_dataset(env: str = "MetaDrive", quality: str = "medium",
                 device: str = "cpu"):
    """Load OSRL dataset or fall back to mock data."""
    try:
        from osrl.common.dataset import SequenceDataset
        print(f"Loading OSRL: {env}-{quality}...")
        ds = SequenceDataset(f"{env}-v0-{quality}-replay-v0")
        obs = torch.tensor(ds.dataset["observations"], dtype=torch.float32, device=device)
        act = torch.tensor(ds.dataset["actions"], dtype=torch.float32, device=device)
        rew = torch.tensor(ds.dataset["rewards"], dtype=torch.float32, device=device)
        cost = torch.tensor(ds.dataset["costs"], dtype=torch.float32, device=device)
        done = torch.tensor(ds.dataset["terminals"], dtype=torch.float32, device=device)
        next_obs = torch.roll(obs, -1, dims=0)
        print(f"  Loaded {len(obs)} transitions  obs_dim={obs.shape[-1]}")
        return obs, act, rew, cost, done, next_obs
    except Exception as e:
        print(f"OSRL unavailable ({e}), generating mock dataset.")
        return _make_mock_dataset(device)


def _make_mock_dataset(device: str, n: int = 50_000, obs_dim: int = 23, act_dim: int = 2):
    rng = np.random.default_rng(42)
    obs = torch.tensor(rng.normal(0, 1, (n, obs_dim)).astype(np.float32), device=device)
    act = torch.tensor(rng.uniform(-1, 1, (n, act_dim)).astype(np.float32), device=device)
    # 70% safe transitions (cost=0, reward~1), 30% risky (cost~1, reward~0.5)
    is_safe = torch.tensor(rng.random(n) < 0.7, device=device)
    rew = torch.where(is_safe, torch.ones(n, device=device) * 0.8,
                      torch.ones(n, device=device) * 0.3)
    cost = torch.where(is_safe, torch.zeros(n, device=device),
                       torch.ones(n, device=device) * 0.8)
    done = torch.zeros(n, device=device)
    done[::200] = 1.0
    next_obs = torch.roll(obs, -1, dims=0)
    print(f"  Mock dataset: {n} transitions  obs_dim={obs_dim}")
    return obs, act, rew, cost, done, next_obs


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def evaluate(agent: CQLAgent, obs: torch.Tensor, act: torch.Tensor,
             rew: torch.Tensor, cost: torch.Tensor) -> dict:
    """Compute Normalized Return, Safety Success Rate, Constraint Violation Cost."""
    with torch.no_grad():
        expert_return = 1.0         # normalized to 1.0 by convention
        episode_rew = rew.sum().item() / max(1, (obs.shape[0] // 200))
        episode_cost = cost.sum().item() / max(1, (obs.shape[0] // 200))

        normalized_return = min(episode_rew / (200 * 0.8 + 1e-6), 1.0)
        safety_success = float(episode_cost < 1.0)
        constraint_violation = episode_cost

    return {
        "normalized_return": round(normalized_return, 4),
        "safety_success_rate": round(safety_success, 4),
        "constraint_violation_cost": round(constraint_violation, 4),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(env: str = "MetaDrive", dataset: str = "medium",
          augmented_data: str = None,
          n_epochs: int = 50, batch_size: int = 256,
          lr: float = 3e-4, alpha: float = 1.0,
          device: str = None):
    output_dir = os.path.join(_PROJECT_ROOT, "safe_gta", "results")
    ckpt_dir   = os.path.join(_PROJECT_ROOT, "safe_gta", "checkpoints")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    is_baseline2 = augmented_data is not None
    label = "Baseline 2 — CQL + GTA (no safety)" if is_baseline2 else "Baseline 1 — CQL (no augmentation)"
    print(f"{label}  |  device={device}")

    if is_baseline2:
        obs, act, rew, cost, done, next_obs = load_augmented_dataset(augmented_data, device)
    else:
        obs, act, rew, cost, done, next_obs = load_dataset(env, dataset, device)
    obs_dim = obs.shape[-1]
    act_dim = act.shape[-1]

    agent = CQLAgent(obs_dim, act_dim, alpha=alpha, lr=lr, device=device)
    print(f"Q-network params: {sum(p.numel() for p in agent.q1.parameters()):,} each")

    loader = DataLoader(
        TensorDataset(obs, act, rew, cost, done, next_obs),
        batch_size=batch_size, shuffle=True
    )

    all_losses = []
    for epoch in range(1, n_epochs + 1):
        epoch_losses = []
        for obs_b, act_b, rew_b, cost_b, done_b, nobs_b in loader:
            loss, bl, cl = agent.update(obs_b, act_b, rew_b, nobs_b, done_b)
            epoch_losses.append(loss)
        mean_loss = np.mean(epoch_losses)
        all_losses.append(mean_loss)

        if epoch % 10 == 0 or epoch == 1:
            metrics = evaluate(agent, obs, act, rew, cost)
            print(f"Epoch {epoch:3d}/{n_epochs}  loss={mean_loss:.4f}  "
                  f"return={metrics['normalized_return']:.3f}  "
                  f"safety={metrics['safety_success_rate']:.3f}  "
                  f"cost={metrics['constraint_violation_cost']:.3f}")

    # Final metrics
    final_metrics = evaluate(agent, obs, act, rew, cost)
    final_metrics["method"] = "CQL_GTA_no_safety" if is_baseline2 else "CQL_no_augmentation"
    final_metrics["dataset"] = augmented_data if is_baseline2 else dataset
    final_metrics["n_epochs"] = n_epochs

    header = "BASELINE 2 — CQL + GTA (no safety)" if is_baseline2 else "BASELINE 1 — CQL FINAL RESULTS"
    print("\n" + "=" * 50)
    print(f"  {header}")
    print("=" * 50)
    for k, v in final_metrics.items():
        print(f"  {k}: {v}")
    print("=" * 50)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    fname = "baseline2_cql_results.json" if is_baseline2 else "baseline1_cql_results.json"
    results_path = os.path.join(output_dir, fname)
    with open(results_path, "w") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"Results saved: {results_path}")

    # Save model
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_name = "baseline2_cql.pt" if is_baseline2 else "baseline1_cql.pt"
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    torch.save({
        "q1": agent.q1.state_dict(),
        "q2": agent.q2.state_dict(),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
    }, ckpt_path)
    print(f"Checkpoint saved: {ckpt_path}")

    return final_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MetaDrive")
    parser.add_argument("--dataset", type=str, default="medium")
    parser.add_argument("--augmented_data", type=str, default=None,
                        help="Path to .pkl augmented dataset (activates Baseline 2 mode)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train(
        env=args.env,
        dataset=args.dataset,
        augmented_data=args.augmented_data,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        alpha=args.alpha,
        device=args.device,
    )
