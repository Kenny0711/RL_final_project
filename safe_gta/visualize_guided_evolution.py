"""
Visualize the lane-relative evolution of Safe-GTA guided repair.

This is not a MetaDrive simulator render. It is a route-like plot using the
dataset observation feature "lateral position" as the y-axis and timestep as
the x-axis. It shows how the trajectory changes during guided denoising.

Run:
  python -m safe_gta.visualize_guided_evolution --start_t 25 --beta 2.0
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from safe_gta.generate_safe_gta import denormalize, load_ddpm, normalize
from safe_gta.safety_critic import load_checkpoint as load_critic_checkpoint

RESULTS_DIR = os.path.join(_PROJECT_ROOT, "safe_gta", "results")
DATA_DIR = os.path.join(_PROJECT_ROOT, "safe_gta", "data")
CKPT_DIR = os.path.join(_PROJECT_ROOT, "safe_gta", "checkpoints")

LATERAL_POS_IDX = 4
SPEED_IDX = 7


def find_hdf5():
    for name in ["metadrive_mediumsparse.hdf5", "metadrive_mediummean.hdf5",
                 "metadrive_mediumdense.hdf5"]:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            return path
    matches = sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))
    if not matches:
        raise FileNotFoundError("No HDF5 file found in safe_gta/data/")
    return matches[0]


def load_trajectory_chunks(hdf5_path, seq_len):
    import h5py

    print(f"Loading: {hdf5_path}")
    with h5py.File(hdf5_path, "r") as f:
        obs = np.asarray(f["observations"][:], dtype=np.float32)
        act = np.asarray(f["actions"][:], dtype=np.float32)
        cost = np.asarray(f["costs"][:], dtype=np.float32)
        done = np.asarray(f["terminals"][:], dtype=np.float32)
        if "timeouts" in f:
            done = np.logical_or(done, f["timeouts"][:]).astype(np.float32)

    joints, costs = [], []
    start = 0
    ends = list(np.where(done > 0.5)[0]) + [len(obs) - 1]
    for end in ends:
        ep_obs = obs[start:end + 1]
        ep_act = act[start:end + 1]
        ep_cost = cost[start:end + 1]
        for i in range(0, len(ep_obs) - seq_len + 1, seq_len):
            joints.append(np.concatenate([ep_obs[i:i + seq_len],
                                          ep_act[i:i + seq_len]], axis=-1))
            costs.append(ep_cost[i:i + seq_len])
        start = end + 1

    joints = np.stack(joints).astype(np.float32)
    costs = np.stack(costs).astype(np.float32)
    print(f"Loaded trajectory chunks: {joints.shape}")
    return joints, costs


def mean_cost(critic, x_norm, device):
    critic.eval()
    with torch.no_grad():
        x = torch.tensor(x_norm, dtype=torch.float32, device=device)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        return float(critic(x).mean().cpu())


def trace_guided_repair(ddpm, critic, x_bad_norm, start_t, beta,
                        capture_steps, guidance_clip=1.0):
    """Return normalized snapshots from a guided repair pass."""
    x_tensor = torch.tensor(x_bad_norm, dtype=torch.float32, device=ddpm.device)
    if x_tensor.ndim == 2:
        x_tensor = x_tensor.unsqueeze(0)

    batch_size = x_tensor.shape[0]
    t_batch = torch.full((batch_size,), start_t, dtype=torch.long, device=ddpm.device)
    x = ddpm.schedule.q_sample(x_tensor, t_batch)

    snapshots = {"original": x_tensor.detach().cpu().numpy(),
                 f"noisy t={start_t}": x.detach().cpu().numpy()}

    ddpm.model.eval()
    critic.eval()
    for t_val in range(start_t - 1, -1, -1):
        t = torch.full((batch_size,), t_val, dtype=torch.long, device=ddpm.device)
        with torch.no_grad():
            x = ddpm.schedule.p_sample_step(ddpm.model, x, t)

        if beta > 0:
            x_guided = x.detach().requires_grad_(True)
            cost_loss = critic(x_guided).mean()
            grad = torch.autograd.grad(cost_loss, x_guided)[0]
            if guidance_clip is not None and guidance_clip > 0:
                grad = grad.clamp(-guidance_clip, guidance_clip)
            x = (x_guided - beta * grad).detach().clamp(-1.0, 1.0)

        if t_val in capture_steps:
            snapshots[f"guided t={t_val}"] = x.detach().cpu().numpy()

    snapshots["final"] = x.detach().cpu().numpy()
    return snapshots


def plot_evolution(raw_snapshots, costs, source_total_cost,
                   out_path, title_suffix):
    import matplotlib.pyplot as plt

    steps = np.arange(next(iter(raw_snapshots.values())).shape[0])
    labels = list(raw_snapshots.keys())

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), height_ratios=[2.2, 1])
    fig.suptitle(f"Safe-GTA Guided Repair Evolution {title_suffix}",
                 fontsize=14, fontweight="bold")

    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(labels)))
    for i, label in enumerate(labels):
        raw = raw_snapshots[label]
        if label == "original":
            ax.plot(steps, raw[:, LATERAL_POS_IDX], color="#4D4D4D",
                    linestyle="--", linewidth=2.0,
                    label=f"{label} (dataset cost={source_total_cost:.1f})")
        elif label.startswith("noisy"):
            ax.plot(steps, raw[:, LATERAL_POS_IDX], color="#D95F02",
                    linestyle=":", linewidth=1.8,
                    label=f"{label} (critic={costs[label]:.3f})")
        else:
            ax.plot(steps, raw[:, LATERAL_POS_IDX], color=colors[i],
                    linewidth=1.8, label=f"{label} (critic={costs[label]:.3f})")

    ax.set_ylabel("Lane-relative lateral position feature")
    ax.set_xlabel("Timestep")
    ax.set_title("Route-like view: timestep vs lateral position")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = axes[1]
    cost_labels = [label for label in labels if label in costs]
    cost_vals = [costs[label] for label in cost_labels]
    ax2.plot(range(len(cost_vals)), cost_vals, marker="o", color="#1F77B4")
    ax2.set_xticks(range(len(cost_labels)))
    ax2.set_xticklabels(cost_labels, rotation=25, ha="right")
    ax2.set_ylabel("Mean critic cost")
    ax2.set_title("Predicted safety cost during guided denoising")
    ax2.grid(True, alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main(start_t=25, beta=2.0, index=None, search_n=256, seed=7,
         unsafe_min_cost=1.0, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    diffusion_checkpoint = os.path.join(CKPT_DIR, "diffusion_metadrive_final.pt")
    critic_checkpoint = os.path.join(CKPT_DIR, "metadrive_safety_critic.pt")
    hdf5_path = find_hdf5()

    ddpm, meta = load_ddpm(diffusion_checkpoint, device)
    critic, critic_payload = load_critic_checkpoint(critic_checkpoint, device=device)
    if critic_payload["n_features"] != meta["n_features"]:
        raise ValueError("Critic and diffusion feature dimensions do not match.")

    joints, dataset_costs = load_trajectory_chunks(hdf5_path, seq_len=meta["seq_len"])
    joints = joints[:, :, :meta["n_features"]]
    total_cost = dataset_costs.sum(axis=1)

    capture_steps = {20, 15, 10, 5, 0}
    capture_steps = {t for t in capture_steps if t < start_t}

    if index is None:
        rng = np.random.default_rng(seed)
        pool = np.where(total_cost > unsafe_min_cost)[0]
        if len(pool) == 0:
            pool = np.arange(len(joints))
        candidate_count = min(search_n, len(pool))
        candidate_idx = rng.choice(pool, size=candidate_count, replace=False)
        print(f"Searching {candidate_count} candidate trajectories for a clear low-cost example...")
        batch = joints[candidate_idx]
        batch_norm = normalize(batch, meta["norm_stats"], meta["n_features"]).astype(np.float32)
        torch.manual_seed(seed)
        norm_snapshots_batch = trace_guided_repair(
            ddpm, critic, batch_norm, start_t=start_t, beta=beta,
            capture_steps=capture_steps,
        )
        original_costs = critic(
            torch.tensor(norm_snapshots_batch["original"],
                         dtype=torch.float32, device=device)
        ).mean(dim=1).detach().cpu().numpy()
        final_costs = critic(
            torch.tensor(norm_snapshots_batch["final"],
                         dtype=torch.float32, device=device)
        ).mean(dim=1).detach().cpu().numpy()
        improvement = original_costs - final_costs
        if np.any(improvement > 0):
            best_local = int(np.argmax(improvement))
        else:
            best_local = int(np.argmin(final_costs))
        index = int(candidate_idx[best_local])
        print(
            "Selected best candidate: "
            f"index={index}  original_critic={original_costs[best_local]:.4f}  "
            f"final_critic={final_costs[best_local]:.4f}  "
            f"improvement={improvement[best_local]:.4f}"
        )
        norm_snapshots = {
            label: arr[best_local]
            for label, arr in norm_snapshots_batch.items()
        }
    else:
        print(f"Using requested trajectory index: {index}")
        x_bad = joints[index]
        x_bad_norm = normalize(x_bad[None], meta["norm_stats"], meta["n_features"]).astype(np.float32)
        torch.manual_seed(seed)
        norm_snapshots_batch = trace_guided_repair(
            ddpm, critic, x_bad_norm, start_t=start_t, beta=beta,
            capture_steps=capture_steps,
        )
        norm_snapshots = {
            label: arr[0]
            for label, arr in norm_snapshots_batch.items()
        }

    print(f"Dataset total cost for selected source: {total_cost[index]:.3f}")

    raw_snapshots = {
        label: denormalize(arr[None], meta["norm_stats"], meta["n_features"])[0]
        for label, arr in norm_snapshots.items()
    }
    costs = {
        label: mean_cost(critic, arr, device)
        for label, arr in norm_snapshots.items()
    }

    out_path = os.path.join(RESULTS_DIR, "guided_repair_evolution_route.png")
    plot_evolution(
        raw_snapshots,
        costs,
        source_total_cost=float(total_cost[index]),
        out_path=out_path,
        title_suffix=f"(start_t={start_t}, beta={beta})",
    )
    print("\nSnapshot critic costs:")
    for label, value in costs.items():
        print(f"  {label:14s}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_t", type=int, default=25)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--search_n", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--unsafe_min_cost", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    main(start_t=args.start_t, beta=args.beta, index=args.index,
         search_n=args.search_n, seed=args.seed,
         unsafe_min_cost=args.unsafe_min_cost, device=args.device)
