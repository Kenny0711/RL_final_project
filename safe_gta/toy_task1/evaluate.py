"""
Toy Task 1 — Evaluation Script
Run: python -m safe_gta.toy_task1.evaluate --checkpoint safe_gta/checkpoints/toy_task1_final.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from safe_gta.diffusion.noise_schedule import NoiseSchedule
from safe_gta.diffusion.unet1d import TemporalUNet
from safe_gta.diffusion.ddpm import DDPM
from safe_gta.toy_task1.data_gen import (
    generate_good_trajectories,
    generate_bad_trajectories,
    normalize_trajectories,
    denormalize_trajectories,
    load_dataset,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_l2_distance_from_ideal(trajs: np.ndarray) -> np.ndarray:
    """
    Mean L2 distance from ideal straight line y=0 over x ∈ [0, 10].
    trajs: (N, seq_len, 2)  — in original (un-normalized) space
    Returns: (N,) per-trajectory mean L2 distance
    """
    ideal_y = np.zeros(trajs.shape[1])
    return np.mean(np.abs(trajs[:, :, 1] - ideal_y), axis=1)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_repair_comparison(bad: np.ndarray, repaired: np.ndarray,
                           n_show: int = 8,
                           save_path: str = "safe_gta/results/repair_comparison.png"):
    """
    Grid: top row = bad (red), bottom row = repaired (blue), ideal line (green dashed).
    """
    import matplotlib.pyplot as plt

    n_show = min(n_show, len(bad))
    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 2.5, 5))

    for i in range(n_show):
        for row, (traj, color, label) in enumerate([
            (bad[i], "red", "bad"),
            (repaired[i], "dodgerblue", "repaired"),
        ]):
            ax = axes[row, i]
            ax.plot(traj[:, 0], traj[:, 1], color=color, lw=1.5)
            ax.axhline(0, color="green", linestyle="--", lw=1, alpha=0.7,
                       label="ideal" if i == 0 else None)
            ax.set_ylim(-3, 3)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_ylabel(label, fontsize=9)

    axes[0, n_show // 2].set_title("Trajectory Repair Comparison", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Saved: {save_path}")


def plot_denoising_timelapse(ddpm: DDPM, bad_traj: np.ndarray,
                              stats: dict, start_t: int = None,
                              n_frames: int = 10,
                              save_path: str = "safe_gta/results/denoising_timelapse.png"):
    """
    Shows repair process at n_frames intermediate timesteps.
    Colors transition from red (noisy) → blue (clean).
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if start_t is None:
        start_t = min(500, ddpm.schedule.T - 1)

    device = ddpm.device
    bad_norm, _ = normalize_trajectories(bad_traj[np.newaxis], stats=stats)
    x0 = torch.tensor(bad_norm, dtype=torch.float32, device=device)
    t_batch = torch.full((1,), start_t, dtype=torch.long, device=device)
    x_noisy = ddpm.schedule.q_sample(x0, t_batch)

    # collect frames during reverse diffusion
    frames = []
    t_steps = sorted(set(
        [start_t] +
        list(np.linspace(start_t - 1, 0, n_frames - 1).astype(int))
    ), reverse=True)

    ddpm.model.eval()
    x = x_noisy.clone()
    with torch.no_grad():
        for t_val in range(start_t - 1, -1, -1):
            t_b = torch.full((1,), t_val, dtype=torch.long, device=device)
            x = ddpm.schedule.p_sample_step(ddpm.model, x, t_b)
            if t_val in t_steps or t_val == 0:
                x_np = denormalize_trajectories(x.cpu().numpy(), stats)[0]
                frames.append((t_val, x_np))

    # keep at most n_frames frames
    if len(frames) > n_frames:
        idx = np.linspace(0, len(frames) - 1, n_frames).astype(int)
        frames = [frames[i] for i in idx]

    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(n * 2.2, 3))
    colors = cm.coolwarm(np.linspace(1.0, 0.0, n))

    bad_raw = denormalize_trajectories(bad_traj[np.newaxis], stats)[0]
    for i, (t_val, traj) in enumerate(frames):
        ax = axes[i] if n > 1 else axes
        ax.plot(traj[:, 0], traj[:, 1], color=colors[i], lw=1.5)
        ax.plot(bad_raw[:, 0], bad_raw[:, 1], color="salmon", lw=0.8, alpha=0.4)
        ax.axhline(0, color="green", linestyle="--", lw=0.8, alpha=0.6)
        ax.set_ylim(-3, 3)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t={t_val}", fontsize=7)

    plt.suptitle("Denoising Timelapse (red→blue = noisy→clean)", fontsize=10)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Saved: {save_path}")


def plot_start_t_ablation(ddpm: DDPM, bad_trajs: np.ndarray, stats: dict,
                           start_t_values=(200, 400, 500, 700),
                           save_path: str = "safe_gta/results/l2_ablation_start_t.png"):
    import matplotlib.pyplot as plt

    mean_l2s = []
    for st in start_t_values:
        rep = ddpm.repair(bad_trajs, start_t=st)
        rep_denorm = denormalize_trajectories(rep, stats)
        mean_l2s.append(compute_l2_distance_from_ideal(rep_denorm).mean())

    plt.figure(figsize=(6, 4))
    plt.plot(start_t_values, mean_l2s, "o-", color="steelblue")
    bad_denorm = denormalize_trajectories(bad_trajs, stats)
    plt.axhline(compute_l2_distance_from_ideal(bad_denorm).mean(),
                color="red", linestyle="--", label="bad (baseline)")
    plt.xlabel("start_t"); plt.ylabel("Mean L2 from ideal")
    plt.title("Repair quality vs start_t"); plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"Saved: {save_path}")


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def run_full_evaluation(checkpoint_path: str, n_eval: int = 200, start_t: int = 500):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    T = payload.get("T", 200)
    stats = payload.get("norm_stats", None)

    model = TemporalUNet(seq_len=50, n_features=2)
    schedule = NoiseSchedule(T=T, device=device)
    ddpm = DDPM(model, schedule, device=device)
    ddpm.load_checkpoint(checkpoint_path)
    print(f"Loaded checkpoint from {checkpoint_path}  (T={T})")

    # Generate eval data
    good = generate_good_trajectories(n_traj=n_eval, seq_len=50, seed=999)
    bad = generate_bad_trajectories(good, seed=888)

    if stats:
        good_norm, _ = normalize_trajectories(good, stats=stats)
        bad_norm, _ = normalize_trajectories(bad, stats=stats)
    else:
        good_norm, stats = normalize_trajectories(good)
        bad_norm, _ = normalize_trajectories(bad, stats=stats)

    # Repair
    print(f"Repairing {n_eval} trajectories (start_t={start_t})...")
    repaired_norm = ddpm.repair(bad_norm, start_t=start_t)
    repaired_norm = np.array(repaired_norm)

    # Denormalize for metric
    bad_raw = denormalize_trajectories(bad_norm, stats)
    repaired_raw = denormalize_trajectories(repaired_norm, stats)

    l2_bad = compute_l2_distance_from_ideal(bad_raw)
    l2_repaired = compute_l2_distance_from_ideal(repaired_raw)
    improvement = (l2_repaired < l2_bad).mean()

    print("\n" + "=" * 45)
    print("  TOY TASK 1 EVALUATION")
    print("=" * 45)
    print(f"  Bad trajectories:      L2 = {l2_bad.mean():.3f} ± {l2_bad.std():.3f}")
    print(f"  Repaired trajectories: L2 = {l2_repaired.mean():.3f} ± {l2_repaired.std():.3f}")
    print(f"  Improvement rate: {improvement * 100:.1f}%")
    print("=" * 45)

    if l2_repaired.mean() < l2_bad.mean():
        print("  SUCCESS CRITERION MET ✓")
    else:
        print("  CRITERION NOT MET ✗  — consider increasing start_t or training longer")
    print()

    # Plots
    try:
        plot_repair_comparison(bad_raw, repaired_raw)
        plot_denoising_timelapse(ddpm, bad_norm[0], stats, start_t=start_t)
        plot_start_t_ablation(ddpm, bad_norm[:50], stats)
    except ImportError:
        print("matplotlib not found — skipping plots")

    return l2_bad, l2_repaired


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="safe_gta/checkpoints/toy_task1_final.pt")
    parser.add_argument("--n_eval", type=int, default=200)
    parser.add_argument("--start_t", type=int, default=500)
    args = parser.parse_args()

    run_full_evaluation(args.checkpoint, n_eval=args.n_eval, start_t=args.start_t)
