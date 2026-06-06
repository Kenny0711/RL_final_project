"""
Visualize why an original MetaDrive offline trajectory is dangerous, then
compare it with a Safe-GTA repaired trajectory.

The offline HDF5 file does not contain full simulator replay state, so this is
a lane-relative bird-view proxy. The danger labels, however, are real HDF5
cost components from MetaDrive: crash_costs, proximity_costs, velocity_costs,
and out_of_road_costs.

Run:
  python -m safe_gta.visualize_danger_reason --index 6118 --start_t 25 --beta 2.0
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
from safe_gta.visualize_guided_evolution import (
    find_hdf5,
    load_trajectory_chunks,
    trace_guided_repair,
)

RESULTS_DIR = os.path.join(_PROJECT_ROOT, "safe_gta", "results")
CKPT_DIR = os.path.join(_PROJECT_ROOT, "safe_gta", "checkpoints")

LATERAL_POS_IDX = 4
COST_KEYS = ["crash_costs", "out_of_road_costs", "proximity_costs", "velocity_costs"]


def hdf5_chunk_bounds(hdf5_path, seq_len):
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        done = np.asarray(f["terminals"][:], dtype=bool)
        if "timeouts" in f:
            done = np.logical_or(done, np.asarray(f["timeouts"][:], dtype=bool))

    bounds = []
    start = 0
    ends = list(np.where(done)[0]) + [len(done) - 1]
    for end in ends:
        ep_len = end - start + 1
        for i in range(0, ep_len - seq_len + 1, seq_len):
            bounds.append((start + i, start + i + seq_len))
        start = end + 1
    return bounds


def load_cost_components(hdf5_path, chunk_index, seq_len):
    import h5py

    bounds = hdf5_chunk_bounds(hdf5_path, seq_len)
    start, end = bounds[chunk_index]
    components = {}
    with h5py.File(hdf5_path, "r") as f:
        for key in COST_KEYS:
            if key in f:
                components[key] = np.asarray(f[key][start:end], dtype=np.float32)
            else:
                components[key] = np.zeros(seq_len, dtype=np.float32)
        components["costs"] = np.asarray(f["costs"][start:end], dtype=np.float32)
    return components, (start, end)


def route_xy(raw):
    steps = raw.shape[0]
    x = np.linspace(0.0, steps - 1, steps)
    y = (raw[:, LATERAL_POS_IDX] - 0.5) * 100.0
    return x, y


def critic_steps(critic, x_norm, device):
    critic.eval()
    with torch.no_grad():
        x = torch.tensor(x_norm[None], dtype=torch.float32, device=device)
        return critic(x).squeeze(0).detach().cpu().numpy()


def plot_danger_reason(original_raw, repaired_raw, components,
                       original_critic_steps, repaired_critic_steps,
                       out_path, index, source_bounds, start_t, beta):
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.patches import Rectangle

    x_orig, y_orig = route_xy(original_raw)
    x_rep, y_rep = route_xy(repaired_raw)
    all_y = np.concatenate([y_orig, y_rep])
    y_abs = max(1.2, float(np.max(np.abs(all_y))) * 1.3)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2.45, 1.05])
    fig.suptitle(
        f"Danger Evidence and Safe-GTA Repair (index={index}, start_t={start_t}, beta={beta})",
        fontsize=15,
        fontweight="bold",
    )

    ax = axes[0]
    ax.add_patch(
        Rectangle(
            (-1.0, -y_abs),
            51.0,
            2.0 * y_abs,
            facecolor="#ECECEC",
            edgecolor="#B8B8B8",
            linewidth=1.0,
            zorder=0,
        )
    )
    ax.axhline(0.0, color="#666666", linestyle=(0, (6, 5)), linewidth=1.2, alpha=0.7)
    ax.axhline(y_abs * 0.72, color="#A0A0A0", linewidth=1.0, alpha=0.65)
    ax.axhline(-y_abs * 0.72, color="#A0A0A0", linewidth=1.0, alpha=0.65)

    points = np.array([x_rep, y_rep]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    rep_cost = repaired_critic_steps[:-1]
    lc = LineCollection(segments, cmap="viridis_r", norm=plt.Normalize(0.0, 1.0))
    lc.set_array(rep_cost)
    lc.set_linewidth(3.2)
    lc.set_alpha(0.95)
    ax.add_collection(lc)

    ax.plot(
        x_orig,
        y_orig,
        color="#333333",
        linestyle="--",
        linewidth=2.6,
        label=f"original route (true dataset cost={components['costs'].sum():.3f})",
        zorder=3,
    )
    ax.plot(
        x_rep,
        y_rep,
        color="#E6C700",
        linewidth=2.0,
        alpha=0.75,
        label=f"repaired route (critic mean={repaired_critic_steps.mean():.3f})",
        zorder=4,
    )

    crash_steps = np.where(components["crash_costs"] > 0)[0]
    if len(crash_steps):
        ax.scatter(
            x_orig[crash_steps],
            y_orig[crash_steps],
            s=220,
            marker="X",
            color="#D62728",
            edgecolor="white",
            linewidth=1.2,
            label="true crash cost in dataset",
            zorder=6,
        )

    proximity_steps = np.where(components["proximity_costs"] > 0)[0]
    if len(proximity_steps):
        sizes = 80 + 1400 * components["proximity_costs"][proximity_steps]
        ax.scatter(
            x_orig[proximity_steps],
            y_orig[proximity_steps],
            s=sizes,
            marker="o",
            color="#FF7F0E",
            edgecolor="white",
            linewidth=0.8,
            alpha=0.85,
            label="true proximity cost",
            zorder=5,
        )

    velocity_steps = np.where(components["velocity_costs"] > 0)[0]
    if len(velocity_steps):
        ax.scatter(
            x_orig[velocity_steps],
            y_orig[velocity_steps],
            s=28,
            marker="^",
            color="#9467BD",
            alpha=0.75,
            label="true velocity cost",
            zorder=5,
        )

    ax.scatter([x_orig[0]], [y_orig[0]], color="#333333", s=48, zorder=7)
    ax.scatter([x_orig[-1]], [y_orig[-1]], color="#333333", s=52, marker="s", zorder=7)
    ax.text(0.01, 0.94, "start", transform=ax.transAxes, fontsize=9, color="#333333")
    ax.text(0.94, 0.08, "end", transform=ax.transAxes, fontsize=9, color="#333333")
    cbar = fig.colorbar(lc, ax=ax, fraction=0.026, pad=0.015)
    cbar.set_label("Repaired step critic cost")

    ax.set_xlim(-1, 50)
    ax.set_ylim(-y_abs, y_abs)
    ax.set_xlabel("Trajectory progress step")
    ax.set_ylabel("Scaled lane-relative lateral offset")
    ax.set_title("Original danger labels overlaid with repaired route")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = axes[1]
    steps = np.arange(len(components["costs"]))
    bottom = np.zeros_like(steps, dtype=np.float32)
    comp_style = [
        ("velocity_costs", "#9467BD", "velocity"),
        ("proximity_costs", "#FF7F0E", "proximity"),
        ("out_of_road_costs", "#8C564B", "out of road"),
        ("crash_costs", "#D62728", "crash"),
    ]
    for key, color, label in comp_style:
        vals = components[key]
        ax2.bar(steps, vals, bottom=bottom, color=color, alpha=0.8, label=f"true {label} cost")
        bottom += vals
    ax2.plot(
        steps,
        original_critic_steps,
        color="#333333",
        linewidth=1.6,
        label=f"original critic mean={original_critic_steps.mean():.3f}",
    )
    ax2.plot(
        steps,
        repaired_critic_steps,
        color="#2CA25F",
        linewidth=1.8,
        label=f"repaired critic mean={repaired_critic_steps.mean():.3f}",
    )
    ax2.set_xlim(-1, 50)
    ax2.set_xlabel("Trajectory progress step")
    ax2.set_ylabel("Cost / critic score")
    ax2.set_title(
        f"True original cost components from HDF5 chunk {source_bounds[0]}:{source_bounds[1]}"
    )
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(loc="upper left", fontsize=8, ncol=3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    note = (
        "This explains the original dataset danger using true HDF5 cost labels. "
        "The repaired route safety is evaluated by the learned Safety Critic, not simulator replay."
    )
    fig.text(0.5, 0.012, note, ha="center", fontsize=9, color="#555555")

    plt.tight_layout(rect=(0, 0.035, 1, 0.96))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main(index=6118, start_t=25, beta=2.0, seed=7, output=None, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    diffusion_checkpoint = os.path.join(CKPT_DIR, "diffusion_metadrive_final.pt")
    critic_checkpoint = os.path.join(CKPT_DIR, "metadrive_safety_critic.pt")

    ddpm, meta = load_ddpm(diffusion_checkpoint, device)
    critic, critic_payload = load_critic_checkpoint(critic_checkpoint, device=device)
    if critic_payload["n_features"] != meta["n_features"]:
        raise ValueError("Critic and diffusion feature dimensions do not match.")

    hdf5_path = find_hdf5()
    joints, dataset_costs = load_trajectory_chunks(hdf5_path, seq_len=meta["seq_len"])
    joints = joints[:, :, :meta["n_features"]]
    components, source_bounds = load_cost_components(hdf5_path, index, meta["seq_len"])

    x = joints[index]
    x_norm = normalize(x[None], meta["norm_stats"], meta["n_features"]).astype(np.float32)
    capture_steps = {0}
    torch.manual_seed(seed)
    snapshots_batch = trace_guided_repair(
        ddpm,
        critic,
        x_norm,
        start_t=start_t,
        beta=beta,
        capture_steps=capture_steps,
    )
    original_norm = snapshots_batch["original"][0]
    repaired_norm = snapshots_batch["final"][0]
    original_raw = denormalize(original_norm[None], meta["norm_stats"], meta["n_features"])[0]
    repaired_raw = denormalize(repaired_norm[None], meta["norm_stats"], meta["n_features"])[0]

    original_critic = critic_steps(critic, original_norm, device)
    repaired_critic = critic_steps(critic, repaired_norm, device)

    print(f"Selected dataset trajectory index: {index}")
    print(f"HDF5 chunk rows: {source_bounds[0]}:{source_bounds[1]}")
    for key in ["costs"] + COST_KEYS:
        vals = components[key]
        print(f"  {key:18s} sum={vals.sum():.6f} nonzero={np.count_nonzero(vals)} max={vals.max():.6f}")
    print(f"  original critic mean={original_critic.mean():.4f}")
    print(f"  repaired critic mean={repaired_critic.mean():.4f}")

    out_path = output or os.path.join(RESULTS_DIR, "metadrive_danger_reason_topdown.png")
    plot_danger_reason(
        original_raw,
        repaired_raw,
        components,
        original_critic,
        repaired_critic,
        out_path,
        index,
        source_bounds,
        start_t,
        beta,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=6118)
    parser.add_argument("--start_t", type=int, default=25)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=7)
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
