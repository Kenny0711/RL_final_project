"""
Visualize a turn/corner-like trajectory before and after Safe-GTA repair.

The offline MetaDrive HDF5 dataset does not provide full simulator replay
coordinates. This script reconstructs an approximate bird-view curve from the
heading cosine/sine observation features, then overlays the real HDF5
velocity_cost labels that make the original turn unsafe.

Run:
  python -m safe_gta.visualize_turn_repair --index 4211 --start_t 25 --beta 2.0
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

HEADING_COS_IDX = 2
HEADING_SIN_IDX = 3
LATERAL_POS_IDX = 4
SPEED_IDX = 7


def load_original_costs(hdf5_path, index, seq_len):
    import h5py

    bounds = hdf5_chunk_bounds(hdf5_path, seq_len)
    start, end = bounds[index]
    with h5py.File(hdf5_path, "r") as f:
        costs = {name: np.asarray(f[name][start:end], dtype=np.float32)
                 for name in ["costs", "velocity_costs", "crash_costs",
                              "out_of_road_costs", "proximity_costs"]}
    return costs, (start, end)


def critic_step_scores(critic, x_norm, device):
    with torch.no_grad():
        x = torch.tensor(x_norm[None], dtype=torch.float32, device=device)
        return critic(x).squeeze(0).detach().cpu().numpy()


def approximate_turn_path(raw):
    angle = np.unwrap(np.arctan2(raw[:, HEADING_SIN_IDX], raw[:, HEADING_COS_IDX]))
    rel_angle = angle - angle[0]

    # Use a constant step length so the plot highlights the shape of the turn
    # rather than the unknown simulator replay timing.
    dx = np.cos(rel_angle)
    dy = np.sin(rel_angle)
    x = np.concatenate([[0.0], np.cumsum(dx[:-1])])
    y = np.concatenate([[0.0], np.cumsum(dy[:-1])])

    # Add scaled lane-relative offset perpendicular to the heading. This makes
    # the original and repaired route differences visible in the bird-view proxy.
    lateral = (raw[:, LATERAL_POS_IDX] - 0.5) * 8.0
    normal_x = -np.sin(rel_angle)
    normal_y = np.cos(rel_angle)
    x = x + lateral * normal_x
    y = y + lateral * normal_y

    return x, y, rel_angle


def plot_turn(original_raw, repaired_raw, costs, original_critic,
              repaired_critic, out_path, index, bounds, start_t, beta):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    ox, oy, o_angle = approximate_turn_path(original_raw)
    rx, ry, r_angle = approximate_turn_path(repaired_raw)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9),
                             gridspec_kw={"height_ratios": [2.2, 1]})
    fig.suptitle(
        f"Turn/Cornor Repair Evidence (index={index}, start_t={start_t}, beta={beta})",
        fontsize=15,
        fontweight="bold",
    )

    ax = axes[0, 0]
    ax.set_title("Approximate bird-view turn route")
    ax.plot(ox, oy, color="#333333", linestyle="--", linewidth=2.6,
            label=f"original route (true cost={costs['costs'].sum():.3f})")
    ax.plot(rx, ry, color="#E6C700", linewidth=2.6,
            label=f"repaired route (critic mean={repaired_critic.mean():.3f})")

    velocity_steps = np.where(costs["velocity_costs"] > 0)[0]
    if len(velocity_steps):
        sizes = 55 + 900 * costs["velocity_costs"][velocity_steps]
        ax.scatter(ox[velocity_steps], oy[velocity_steps],
                   s=sizes, color="#9467BD", marker="^", alpha=0.85,
                   edgecolor="white", linewidth=0.8,
                   label="true velocity cost on original")

    ax.scatter([ox[0]], [oy[0]], color="#333333", s=60, label="start")
    ax.scatter([ox[-1]], [oy[-1]], color="#333333", s=65, marker="s", label="end")
    ax.axis("equal")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("Approx. x from heading")
    ax.set_ylabel("Approx. y from heading")
    ax.legend(fontsize=8, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax_route = axes[0, 1]
    ax_route.set_title("Repaired route colored by step critic cost")
    points = np.array([rx, ry]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="viridis_r", norm=plt.Normalize(0.0, 1.0))
    lc.set_array(repaired_critic[:-1])
    lc.set_linewidth(3.4)
    ax_route.add_collection(lc)
    ax_route.plot(ox, oy, color="#333333", linestyle="--", linewidth=1.9,
                  alpha=0.75, label="original")
    ax_route.scatter([rx[0]], [ry[0]], color="#222222", s=55)
    ax_route.scatter([rx[-1]], [ry[-1]], color="#222222", s=60, marker="s")
    ax_route.axis("equal")
    ax_route.grid(True, alpha=0.25)
    ax_route.set_xlabel("Approx. x from heading")
    ax_route.set_ylabel("Approx. y from heading")
    fig.colorbar(lc, ax=ax_route, fraction=0.046, pad=0.03,
                 label="Repaired step critic cost")
    ax_route.legend(fontsize=8, loc="best")
    ax_route.spines["top"].set_visible(False)
    ax_route.spines["right"].set_visible(False)

    steps = np.arange(len(costs["costs"]))
    ax_cost = axes[1, 0]
    ax_cost.bar(steps, costs["velocity_costs"], color="#9467BD",
                alpha=0.8, label="true original velocity_cost")
    ax_cost.plot(steps, original_critic, color="#333333", linewidth=1.8,
                 label=f"original critic mean={original_critic.mean():.3f}")
    ax_cost.plot(steps, repaired_critic, color="#2CA25F", linewidth=1.8,
                 label=f"repaired critic mean={repaired_critic.mean():.3f}")
    ax_cost.set_title(f"Safety evidence from HDF5 chunk {bounds[0]}:{bounds[1]}")
    ax_cost.set_xlabel("Trajectory step")
    ax_cost.set_ylabel("Cost / critic score")
    ax_cost.grid(True, axis="y", alpha=0.25)
    ax_cost.legend(fontsize=8)
    ax_cost.spines["top"].set_visible(False)
    ax_cost.spines["right"].set_visible(False)

    ax_heading = axes[1, 1]
    ax_heading.plot(steps, np.rad2deg(o_angle), color="#333333", linewidth=1.8,
                    label="original heading change")
    ax_heading.plot(steps, np.rad2deg(r_angle), color="#E6C700", linewidth=1.8,
                    label="repaired heading change")
    ax_heading.set_title("Heading changes show this is a turning trajectory")
    ax_heading.set_xlabel("Trajectory step")
    ax_heading.set_ylabel("Relative heading angle (deg)")
    ax_heading.grid(True, alpha=0.25)
    ax_heading.legend(fontsize=8)
    ax_heading.spines["top"].set_visible(False)
    ax_heading.spines["right"].set_visible(False)

    note = (
        "This is a heading-based turn proxy from offline observations, not a true "
        "MetaDrive world-coordinate replay. The original danger label is true "
        "HDF5 velocity_cost; repaired safety is learned Safety Critic cost."
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
    costs, bounds = load_original_costs(hdf5_path, index, meta["seq_len"])

    original = joints[index]
    original_norm = normalize(original[None], meta["norm_stats"], meta["n_features"]).astype(np.float32)

    torch.manual_seed(seed)
    snapshots = trace_guided_repair(
        ddpm,
        critic,
        original_norm,
        start_t=start_t,
        beta=beta,
        capture_steps={0},
    )
    repaired_norm = snapshots["final"][0]
    original_norm_single = snapshots["original"][0]

    original_raw = denormalize(original_norm_single[None], meta["norm_stats"], meta["n_features"])[0]
    repaired_raw = denormalize(repaired_norm[None], meta["norm_stats"], meta["n_features"])[0]
    original_critic = critic_step_scores(critic, original_norm_single, device)
    repaired_critic = critic_step_scores(critic, repaired_norm, device)

    angle = np.unwrap(np.arctan2(original_raw[:, HEADING_SIN_IDX], original_raw[:, HEADING_COS_IDX]))
    print(f"Selected turn trajectory index: {index}")
    print(f"  heading change range: {np.rad2deg(angle.max() - angle.min()):.2f} deg")
    print(f"  true dataset cost: {costs['costs'].sum():.4f}")
    print(f"  true velocity_cost: {costs['velocity_costs'].sum():.4f}")
    print(f"  original critic mean: {original_critic.mean():.4f}")
    print(f"  repaired critic mean: {repaired_critic.mean():.4f}")

    out_path = output or os.path.join(RESULTS_DIR, "metadrive_turn_repair.png")
    plot_turn(
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
