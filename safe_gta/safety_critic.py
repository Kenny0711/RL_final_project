"""
Train a MetaDrive Safety Critic on the offline HDF5 dataset.

The critic consumes normalized joint features (observation + action) and
predicts per-step safety cost in [0, 1]. Its gradients are used by
DDPM.guided_repair().

Run:
  python -m safe_gta.safety_critic --epochs 20
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)


class StepSafetyCritic(nn.Module):
    """MLP cost model for normalized (obs, action) features."""

    def __init__(self, n_features: int, hidden: int = 256):
        super().__init__()
        self.n_features = n_features
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, joint: torch.Tensor) -> torch.Tensor:
        original_shape = joint.shape[:-1]
        flat = joint.reshape(-1, joint.shape[-1])
        cost = self.net(flat).squeeze(-1)
        return cost.reshape(original_shape)


def find_hdf5(data_dir: str):
    for name in ["metadrive_mediumsparse.hdf5", "metadrive_mediummean.hdf5",
                 "metadrive_mediumdense.hdf5"]:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path
    matches = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    return matches[0] if matches else None


def load_diffusion_norm(diffusion_checkpoint: str):
    if not diffusion_checkpoint or not os.path.exists(diffusion_checkpoint):
        return None
    payload = torch.load(diffusion_checkpoint, map_location="cpu", weights_only=False)
    norm_stats = payload.get("norm_stats", {})
    if "feat_min" not in norm_stats or "feat_max" not in norm_stats:
        return None
    return norm_stats


def normalize_joint(joint: np.ndarray, norm_stats: dict):
    feat_min = np.asarray(norm_stats["feat_min"], dtype=np.float32)
    feat_max = np.asarray(norm_stats["feat_max"], dtype=np.float32)
    feat_range = np.where(feat_max - feat_min > 1e-6, feat_max - feat_min, 1.0)
    return 2.0 * (joint - feat_min[:joint.shape[-1]]) / feat_range[:joint.shape[-1]] - 1.0


def load_balanced_hdf5_sample(hdf5_path: str, max_samples: int = 200_000,
                              seed: int = 42):
    """Read a balanced safe/dangerous transition sample without loading all obs."""
    import h5py

    rng = np.random.default_rng(seed)
    with h5py.File(hdf5_path, "r") as f:
        costs = np.asarray(f["costs"][:], dtype=np.float32)
        safe_idx = np.where(costs <= 0.0)[0]
        danger_idx = np.where(costs > 0.0)[0]

        n_each = min(max_samples // 2, len(safe_idx), len(danger_idx))
        if n_each == 0:
            raise ValueError("Need both safe and dangerous transitions to train critic.")

        picked = np.concatenate([
            rng.choice(safe_idx, n_each, replace=False),
            rng.choice(danger_idx, n_each, replace=False),
        ])
        picked.sort()

        obs = np.asarray(f["observations"][picked], dtype=np.float32)
        act = np.asarray(f["actions"][picked], dtype=np.float32)
        labels = (costs[picked] > 0.0).astype(np.float32)

    joint = np.concatenate([obs, act], axis=-1)
    order = rng.permutation(len(labels))
    return joint[order], labels[order], obs.shape[-1], act.shape[-1]


def train(
    hdf5_path: str = None,
    diffusion_checkpoint: str = None,
    output_path: str = None,
    epochs: int = 20,
    batch_size: int = 1024,
    lr: float = 1e-3,
    hidden: int = 256,
    max_samples: int = 200_000,
    device: str = None,
):
    data_dir = os.path.join(_PROJECT_ROOT, "safe_gta", "data")
    ckpt_dir = os.path.join(_PROJECT_ROOT, "safe_gta", "checkpoints")
    results_dir = os.path.join(_PROJECT_ROOT, "safe_gta", "results", "metrics")
    figures_dir = os.path.join(_PROJECT_ROOT, "safe_gta", "results", "final_figures")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    if hdf5_path is None:
        hdf5_path = find_hdf5(data_dir)
    if hdf5_path is None:
        raise FileNotFoundError("No MetaDrive HDF5 found. Run python download_dataset.py")

    if diffusion_checkpoint is None:
        diffusion_checkpoint = os.path.join(ckpt_dir, "diffusion_metadrive_final.pt")
    if output_path is None:
        output_path = os.path.join(ckpt_dir, "metadrive_safety_critic.pt")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading critic data: {hdf5_path}")
    joint, labels, obs_dim, act_dim = load_balanced_hdf5_sample(
        hdf5_path, max_samples=max_samples
    )
    n_features = joint.shape[-1]

    norm_stats = load_diffusion_norm(diffusion_checkpoint)
    if norm_stats is None:
        print("No diffusion norm_stats found; fitting critic normalization from sample.")
        feat_min = joint.min(axis=0)
        feat_max = joint.max(axis=0)
        norm_stats = {
            "feat_min": feat_min.tolist(),
            "feat_max": feat_max.tolist(),
            "obs_dim": int(obs_dim),
            "act_dim": int(act_dim),
        }
    joint_norm = normalize_joint(joint, norm_stats).astype(np.float32)

    split = int(0.8 * len(labels))
    x_train, y_train = joint_norm[:split], labels[:split]
    x_val, y_val = joint_norm[split:], labels[split:]

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=batch_size, shuffle=True
    )

    model = StepSafetyCritic(n_features=n_features, hidden=hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            metrics = evaluate_model(model, x_val, y_val, device=device)
            print(f"Epoch {epoch:3d}/{epochs}  loss={np.mean(losses):.4f}  "
                  f"acc={metrics['accuracy']:.3f}  "
                  f"safe_cost={metrics['safe_mean_cost']:.3f}  "
                  f"danger_cost={metrics['danger_mean_cost']:.3f}")

    metrics = evaluate_model(model, x_val, y_val, device=device)
    payload = {
        "model_state": model.state_dict(),
        "n_features": int(n_features),
        "obs_dim": int(obs_dim),
        "act_dim": int(act_dim),
        "hidden": int(hidden),
        "norm_stats": norm_stats,
        "metrics": metrics,
        "source_hdf5": hdf5_path,
    }
    torch.save(payload, output_path)
    print(f"Checkpoint saved: {output_path}")

    result_name = "metadrive_safety_critic_results.json"
    if "_debug" in os.path.basename(output_path).lower():
        result_name = "_debug_safety_critic_results.json"
    results_path = os.path.join(results_dir, result_name)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved: {results_path}")

    plot_name = "_debug_safety_critic_costs.png" if "_debug" in os.path.basename(output_path).lower() else "metadrive_safety_critic_costs.png"
    save_cost_plot(model, x_val, y_val, figures_dir, plot_name=plot_name, device=device)
    return model, metrics


def evaluate_model(model, x_val, y_val, device: str):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(x_val, dtype=torch.float32, device=device)
        pred = model(x).cpu().numpy()
    y = np.asarray(y_val)
    pred_label = pred > 0.5
    safe_mask = y <= 0.5
    danger_mask = y > 0.5
    return {
        "accuracy": float((pred_label == danger_mask).mean()),
        "safe_mean_cost": float(pred[safe_mask].mean()) if safe_mask.any() else None,
        "danger_mean_cost": float(pred[danger_mask].mean()) if danger_mask.any() else None,
        "safe_pass_rate": float((pred[safe_mask] < 0.3).mean()) if safe_mask.any() else None,
        "danger_pass_rate": float((pred[danger_mask] > 0.7).mean()) if danger_mask.any() else None,
    }


def save_cost_plot(model, x_val, y_val, results_dir, plot_name: str, device: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    model.eval()
    with torch.no_grad():
        x = torch.tensor(x_val, dtype=torch.float32, device=device)
        pred = model(x).cpu().numpy()
    y = np.asarray(y_val)

    plt.figure(figsize=(7, 4))
    plt.hist(pred[y <= 0.5], bins=40, alpha=0.65, label="safe", density=True)
    plt.hist(pred[y > 0.5], bins=40, alpha=0.65, label="dangerous", density=True)
    plt.axvline(0.3, color="green", linestyle="--", linewidth=1)
    plt.axvline(0.7, color="red", linestyle="--", linewidth=1)
    plt.xlabel("Predicted cost")
    plt.ylabel("Density")
    plt.title("MetaDrive Safety Critic Cost Distribution")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(results_dir, plot_name)
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"Plot saved: {out}")


def load_checkpoint(path: str, device: str = "cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = StepSafetyCritic(
        n_features=payload["n_features"],
        hidden=payload.get("hidden", 256),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=str, default=None)
    parser.add_argument("--diffusion_checkpoint", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=200_000)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    train(
        hdf5_path=args.hdf5,
        diffusion_checkpoint=args.diffusion_checkpoint,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden=args.hidden,
        max_samples=args.max_samples,
        device=args.device,
    )
