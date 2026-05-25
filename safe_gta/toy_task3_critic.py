"""
Toy Task 3 — Safety Critic Accuracy Test
Run: python -m safe_gta.toy_task3_critic

Goal: Train a simple Safety Critic MLP that outputs:
  - cost < 0.3  for clearly safe state-action pairs
  - cost > 0.7  for clearly dangerous state-action pairs
Success criterion: > 80% accuracy on held-out safe/dangerous pairs.
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# State-action space (simplified MetaDrive-like)
# ---------------------------------------------------------------------------
# State: [x_pos, y_pos, heading, speed, dist_to_left_wall, dist_to_right_wall]
# Action: [steering (-1..1), throttle (0..1)]
STATE_DIM = 6
ACTION_DIM = 2


# ---------------------------------------------------------------------------
# Safety Critic model
# ---------------------------------------------------------------------------

class SafetyCritic(nn.Module):
    """
    Simple MLP: (state, action) → cost ∈ [0, 1]
    Sigmoid output: 0 = safe, 1 = dangerous.
    """

    def __init__(self, state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM,
                 hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Mock dataset construction
# ---------------------------------------------------------------------------

def make_safe_pair(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Safe: driving near center of lane (y ≈ 0), reasonable speed, no aggressive steering.
    dist_to_walls ≈ 1.5~3.0 (well within lane).
    """
    states = np.zeros((n, STATE_DIM), dtype=np.float32)
    states[:, 0] = rng.uniform(0, 20, n)            # x: anywhere on track
    states[:, 1] = rng.uniform(-0.5, 0.5, n)        # y: near center
    states[:, 2] = rng.uniform(-0.15, 0.15, n)      # heading: roughly straight
    states[:, 3] = rng.uniform(0.3, 0.7, n)         # speed: moderate
    states[:, 4] = rng.uniform(1.5, 3.0, n)         # dist_to_left_wall: comfortable
    states[:, 5] = rng.uniform(1.5, 3.0, n)         # dist_to_right_wall: comfortable

    actions = np.zeros((n, ACTION_DIM), dtype=np.float32)
    actions[:, 0] = rng.uniform(-0.2, 0.2, n)       # steering: gentle
    actions[:, 1] = rng.uniform(0.3, 0.7, n)        # throttle: moderate
    return states, actions


def make_dangerous_pair(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Dangerous: near a wall AND moving toward it.
    Small dist_to_wall + steering/heading toward that wall.
    """
    states = np.zeros((n, STATE_DIM), dtype=np.float32)
    states[:, 0] = rng.uniform(0, 20, n)
    # near one of the walls
    near_left = rng.integers(0, 2, n).astype(bool)
    states[:, 1] = np.where(near_left,
                             rng.uniform(-2.5, -1.8, n),
                             rng.uniform(1.8, 2.5, n))
    states[:, 2] = np.where(near_left,
                             rng.uniform(-0.5, -0.15, n),  # heading into left wall
                             rng.uniform(0.15, 0.5, n))    # heading into right wall
    states[:, 3] = rng.uniform(0.5, 1.0, n)                # higher speed = more dangerous
    states[:, 4] = np.where(near_left, rng.uniform(0.05, 0.4, n), rng.uniform(1.5, 3.0, n))
    states[:, 5] = np.where(near_left, rng.uniform(1.5, 3.0, n), rng.uniform(0.05, 0.4, n))

    actions = np.zeros((n, ACTION_DIM), dtype=np.float32)
    # steering toward the wall
    actions[:, 0] = np.where(near_left,
                              rng.uniform(-1.0, -0.5, n),
                              rng.uniform(0.5, 1.0, n))
    actions[:, 1] = rng.uniform(0.5, 1.0, n)  # high throttle = keep going into wall
    return states, actions


def build_dataset(n_per_class: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)
    s_safe, a_safe = make_safe_pair(n_per_class, rng)
    s_dang, a_dang = make_dangerous_pair(n_per_class, rng)

    states = np.concatenate([s_safe, s_dang], axis=0)
    actions = np.concatenate([a_safe, a_dang], axis=0)
    labels = np.concatenate([
        np.zeros(n_per_class, dtype=np.float32),   # safe = 0
        np.ones(n_per_class, dtype=np.float32),    # dangerous = 1
    ])

    # shuffle
    idx = rng.permutation(len(labels))
    return states[idx], actions[idx], labels[idx]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_critic(n_epochs: int = 100, batch_size: int = 128, lr: float = 1e-3,
                 device: str = None) -> SafetyCritic:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    states, actions, labels = build_dataset(n_per_class=2000)
    split = int(0.8 * len(labels))
    s_tr, a_tr, y_tr = states[:split], actions[:split], labels[:split]
    s_val, a_val, y_val = states[split:], actions[split:], labels[split:]

    to_t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
    loader = DataLoader(
        TensorDataset(to_t(s_tr), to_t(a_tr), to_t(y_tr)),
        batch_size=batch_size, shuffle=True
    )

    model = SafetyCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(1, n_epochs + 1):
        model.train()
        for s, a, y in loader:
            optimizer.zero_grad()
            pred = model(s, a)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        if epoch % 20 == 0 or epoch == n_epochs:
            model.eval()
            with torch.no_grad():
                val_pred = model(to_t(s_val), to_t(a_val))
                val_acc = ((val_pred > 0.5).float() == to_t(y_val)).float().mean()
            print(f"Epoch {epoch:3d}/{n_epochs}  val_acc={val_acc.item():.3f}")

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_critic(model: SafetyCritic, n_eval: int = 500, device: str = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rng = np.random.default_rng(999)
    s_safe, a_safe = make_safe_pair(n_eval, rng)
    s_dang, a_dang = make_dangerous_pair(n_eval, rng)

    to_t = lambda x: torch.tensor(x, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        cost_safe = model(to_t(s_safe), to_t(a_safe)).cpu().numpy()
        cost_dang = model(to_t(s_dang), to_t(a_dang)).cpu().numpy()

    safe_correct = (cost_safe < 0.3).mean()
    dang_correct = (cost_dang > 0.7).mean()
    overall_acc = 0.5 * (safe_correct + dang_correct)

    print("\n" + "=" * 45)
    print("  TOY TASK 3 — SAFETY CRITIC EVALUATION")
    print("=" * 45)
    print(f"  Safe actions:      mean cost = {cost_safe.mean():.3f}  "
          f"(< 0.3): {safe_correct * 100:.1f}%")
    print(f"  Dangerous actions: mean cost = {cost_dang.mean():.3f}  "
          f"(> 0.7): {dang_correct * 100:.1f}%")
    print(f"  Overall accuracy:  {overall_acc * 100:.1f}%")
    print("=" * 45)

    if overall_acc >= 0.80:
        print("  SUCCESS CRITERION MET (>80%) ✓")
    else:
        print("  CRITERION NOT MET ✗ — consider more epochs or better features")
    print()

    # Save cost distribution plot
    try:
        import matplotlib.pyplot as plt
        os.makedirs("safe_gta/results", exist_ok=True)
        plt.figure(figsize=(7, 4))
        plt.hist(cost_safe, bins=30, alpha=0.6, color="green", label="safe", density=True)
        plt.hist(cost_dang, bins=30, alpha=0.6, color="red", label="dangerous", density=True)
        plt.axvline(0.3, color="green", linestyle="--", alpha=0.7, label="safe threshold (0.3)")
        plt.axvline(0.7, color="red", linestyle="--", alpha=0.7, label="danger threshold (0.7)")
        plt.xlabel("Predicted cost"); plt.ylabel("Density")
        plt.title("Safety Critic — Cost Distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig("safe_gta/results/safety_critic_costs.png", dpi=100)
        print("Saved: safe_gta/results/safety_critic_costs.png")
    except ImportError:
        pass

    return overall_acc


if __name__ == "__main__":
    print("=" * 50)
    print("Toy Task 3 — Safety Critic Training & Evaluation")
    print("=" * 50)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_critic(n_epochs=100, device=device)

    # Save checkpoint
    os.makedirs("safe_gta/checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "safe_gta/checkpoints/safety_critic.pt")
    print("Checkpoint saved: safe_gta/checkpoints/safety_critic.pt")

    evaluate_critic(model, device=device)
