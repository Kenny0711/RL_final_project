"""
Create a schematic corner-view figure for explaining Safe-GTA turn repair.

This is a presentation-oriented visualization. It uses the real selected
MetaDrive offline trajectory, real HDF5 velocity_cost labels, and real Safety
Critic scores, but places the route on a stylized corner background so the
turning scenario is visually obvious.

Run:
  python -m safe_gta.visualize_corner_repair --index 4211 --start_t 25 --beta 2.0
"""
import argparse
import os
import sys

import numpy as np
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from safe_gta.generate_safe_gta import denormalize, load_ddpm, normalize
from safe_gta.safety_critic import load_checkpoint as load_critic_checkpoint
from safe_gta.visualize_danger_reason import hdf5_chunk_bounds
from safe_gta.visualize_guided_evolution import (
    find_hdf5,
    load_trajectory_chunks,
    trace_guided_repair,
)

RESULTS_DIR = os.path.join(_PROJECT_ROOT, "safe_gta", "results")
CKPT_DIR = os.path.join(_PROJECT_ROOT, "safe_gta", "checkpoints")

LATERAL_POS_IDX = 4


def load_costs(hdf5_path, index, seq_len):
    import h5py

    bounds = hdf5_chunk_bounds(hdf5_path, seq_len)
    start, end = bounds[index]
    with h5py.File(hdf5_path, "r") as f:
        costs = {
            "costs": np.asarray(f["costs"][start:end], dtype=np.float32),
            "velocity_costs": np.asarray(f["velocity_costs"][start:end], dtype=np.float32),
            "crash_costs": np.asarray(f["crash_costs"][start:end], dtype=np.float32),
            "out_of_road_costs": np.asarray(f["out_of_road_costs"][start:end], dtype=np.float32),
            "proximity_costs": np.asarray(f["proximity_costs"][start:end], dtype=np.float32),
        }
    return costs, bounds[index]


def critic_steps(critic, x_norm, device):
    with torch.no_grad():
        x = torch.tensor(x_norm[None], dtype=torch.float32, device=device)
        return critic(x).squeeze(0).detach().cpu().numpy()


def corner_centerline(n_steps, radius=20.0, angle_deg=85.0):
    theta = np.linspace(0.0, np.deg2rad(angle_deg), n_steps)
    x = radius * np.sin(theta)
    y = radius * (1.0 - np.cos(theta))
    tangent = theta
    normal_x = -np.sin(tangent)
    normal_y = np.cos(tangent)
    return x, y, normal_x, normal_y


def route_on_corner(raw, center):
    cx, cy, nx, ny = center
    lateral = (raw[:, LATERAL_POS_IDX] - 0.5) * 16.0
    return cx + lateral * nx, cy + lateral * ny


def draw_corner_road(ax, center, lane_width=5.0):
    cx, cy, nx, ny = center
    outer_x = cx + lane_width * nx
    outer_y = cy + lane_width * ny
    inner_x = cx - lane_width * nx
    inner_y = cy - lane_width * ny

    road_x = np.concatenate([outer_x, inner_x[::-1]])
    road_y = np.concatenate([outer_y, inner_y[::-1]])
    ax.fill(road_x, road_y, color="#E7E7E7", edgecolor="#B6B6B6", linewidth=1.2, zorder=0)
    ax.plot(cx, cy, color="#777777", linestyle=(0, (7, 5)), linewidth=1.2, alpha=0.8, zorder=1)
    ax.plot(outer_x, outer_y, color="#A8A8A8", linewidth=1.2, zorder=1)
    ax.plot(inner_x, inner_y, color="#A8A8A8", linewidth=1.2, zorder=1)


def plot_corner(original_raw, repaired_raw, costs, original_critic,
                repaired_critic, out_path, index, bounds, start_t, beta):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    n_steps = original_raw.shape[0]
    center = corner_centerline(n_steps)
    ox, oy = route_on_corner(original_raw, center)
    rx, ry = route_on_corner(repaired_raw, center)
    cx, cy, _, _ = center
    amplify = 8.0
    ax_rx = ox + amplify * (rx - ox)
    ax_ry = oy + amplify * (ry - oy)

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.25, 1], hspace=0.32, wspace=0.22)
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_rep = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])
    fig.suptitle(
        f"Schematic Corner Repair View (index={index}, start_t={start_t}, beta={beta})",
        fontsize=15,
        fontweight="bold",
    )

    draw_corner_road(ax_orig, center)
    ax_orig.plot(ox, oy, color="#333333", linestyle="--", linewidth=2.8,
                 label=f"original path (true cost={costs['costs'].sum():.3f})", zorder=4)

    vel_steps = np.where(costs["velocity_costs"] > 0)[0]
    if len(vel_steps):
        sizes = 60 + 1300 * costs["velocity_costs"][vel_steps]
        ax_orig.scatter(ox[vel_steps], oy[vel_steps], s=sizes, color="#9467BD",
                        marker="^", edgecolor="white", linewidth=0.8,
                        alpha=0.9, label="true velocity cost", zorder=6)

    ax_orig.scatter([cx[0]], [cy[0]], color="#222222", s=70, zorder=8)
    ax_orig.scatter([cx[-1]], [cy[-1]], color="#222222", s=75, marker="s", zorder=8)
    ax_orig.text(cx[0] - 1.0, cy[0] - 1.8, "start", fontsize=10, color="#333333")
    ax_orig.text(cx[-1] - 2.0, cy[-1] + 1.4, "end", fontsize=10, color="#333333")
    ax_orig.axis("equal")
    ax_orig.set_title("Original corner: over-speed cost labels")
    ax_orig.set_xlabel("Schematic road x")
    ax_orig.set_ylabel("Schematic road y")
    ax_orig.grid(True, alpha=0.18)
    ax_orig.legend(loc="upper left", fontsize=8)
    ax_orig.spines["top"].set_visible(False)
    ax_orig.spines["right"].set_visible(False)

    draw_corner_road(ax_rep, center)
    ax_rep.plot(ox, oy, color="#333333", linestyle="--", linewidth=1.5,
                alpha=0.35, label="original reference", zorder=3)
    ax_rep.plot(rx, ry, color="#E6C700", linewidth=2.5,
                label=f"actual repaired path (critic mean={repaired_critic.mean():.3f})", zorder=5)
    ax_rep.plot(ax_rx, ax_ry, color="#009E73", linewidth=2.2,
                linestyle="-.", label=f"route delta amplified {amplify:.0f}x", zorder=6)

    # Show repaired route local risk as small colored dots.
    sc = ax_rep.scatter(rx, ry, c=repaired_critic, cmap="viridis_r", vmin=0, vmax=1,
                    s=24, alpha=0.9, zorder=7)
    ax_rep.scatter([cx[0]], [cy[0]], color="#222222", s=70, zorder=8)
    ax_rep.scatter([cx[-1]], [cy[-1]], color="#222222", s=75, marker="s", zorder=8)
    ax_rep.text(cx[0] - 1.0, cy[0] - 1.8, "start", fontsize=10, color="#333333")
    ax_rep.text(cx[-1] - 2.0, cy[-1] + 1.4, "end", fontsize=10, color="#333333")

    cbar = fig.colorbar(sc, ax=ax_rep, fraction=0.046, pad=0.02)
    cbar.set_label("Repaired step critic cost")
    ax_rep.axis("equal")
    ax_rep.set_title("Safe-GTA repair: path change is small, risk is lower")
    ax_rep.set_xlabel("Schematic road x")
    ax_rep.set_ylabel("Schematic road y")
    ax_rep.grid(True, alpha=0.18)
    ax_rep.legend(loc="upper left", fontsize=8)
    ax_rep.spines["top"].set_visible(False)
    ax_rep.spines["right"].set_visible(False)

    steps = np.arange(n_steps)
    ax2.bar(steps, costs["velocity_costs"], color="#9467BD", alpha=0.8,
            label=f"true original velocity_cost sum={costs['velocity_costs'].sum():.3f}")
    ax2.plot(steps, original_critic, color="#333333", linewidth=1.8,
             label=f"original critic mean={original_critic.mean():.3f}")
    ax2.plot(steps, repaired_critic, color="#2CA25F", linewidth=1.8,
             label=f"repaired critic mean={repaired_critic.mean():.3f}")
    ax2.set_title(f"Safety evidence from HDF5 chunk {bounds[0]}:{bounds[1]}")
    ax2.set_xlabel("Trajectory step through the corner")
    ax2.set_ylabel("Cost / critic score")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    note = (
        "Schematic corner background for presentation. Route/cost values come from "
        "offline MetaDrive observations, true HDF5 velocity_cost labels, and the "
        "learned Safety Critic; this is not simulator replay. The green line is an "
        "8x amplified route delta for visibility, not the actual repaired path."
    )
    fig.text(0.5, 0.012, note, ha="center", fontsize=9, color="#555555")

    plt.tight_layout(rect=(0, 0.035, 1, 0.96))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main(index=4211, start_t=25, beta=2.0, seed=11, output=None, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    hdf5_path = find_hdf5()
    ddpm, meta = load_ddpm(os.path.join(CKPT_DIR, "diffusion_metadrive_final.pt"), device)
    critic, payload = load_critic_checkpoint(
        os.path.join(CKPT_DIR, "metadrive_safety_critic.pt"), device=device
    )
    if payload["n_features"] != meta["n_features"]:
        raise ValueError("Critic and diffusion feature dimensions do not match.")

    joints, _ = load_trajectory_chunks(hdf5_path, seq_len=meta["seq_len"])
    joints = joints[:, :, :meta["n_features"]]
    costs, bounds = load_costs(hdf5_path, index, meta["seq_len"])

    original_norm = normalize(joints[index:index + 1], meta["norm_stats"], meta["n_features"]).astype(np.float32)
    torch.manual_seed(seed)
    snapshots = trace_guided_repair(
        ddpm,
        critic,
        original_norm,
        start_t=start_t,
        beta=beta,
        capture_steps={0},
    )
    original_norm_single = snapshots["original"][0]
    repaired_norm = snapshots["final"][0]
    original_raw = denormalize(original_norm_single[None], meta["norm_stats"], meta["n_features"])[0]
    repaired_raw = denormalize(repaired_norm[None], meta["norm_stats"], meta["n_features"])[0]
    original_critic = critic_steps(critic, original_norm_single, device)
    repaired_critic = critic_steps(critic, repaired_norm, device)

    print(f"Selected corner trajectory index: {index}")
    print(f"  true dataset cost: {costs['costs'].sum():.4f}")
    print(f"  true velocity_cost: {costs['velocity_costs'].sum():.4f}")
    print(f"  original critic mean: {original_critic.mean():.4f}")
    print(f"  repaired critic mean: {repaired_critic.mean():.4f}")

    out_path = output or os.path.join(RESULTS_DIR, "metadrive_corner_repair.png")
    plot_corner(
        original_raw,
        repaired_raw,
        costs,
        original_critic,
        repaired_critic,
        out_path,
        index,
        bounds,
        start_t,
        beta,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=4211)
    parser.add_argument("--start_t", type=int, default=25)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    main(
        index=args.index,
        start_t=args.start_t,
        beta=args.beta,
        seed=args.seed,
        output=args.output,
        device=args.device,
    )
