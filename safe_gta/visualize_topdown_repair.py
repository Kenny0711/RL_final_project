"""
Create a bird-view style comparison of original vs Safe-GTA repaired routes.

Important: the offline MetaDrive HDF5 dataset used here does not contain the
full simulator world coordinates or map replay state. This plot is therefore a
lane-relative top-down proxy: x-axis is trajectory progress, y-axis is the
scaled lane-relative lateral-position observation feature.

Run:
  python -m safe_gta.visualize_topdown_repair --start_t 25 --beta 2.0
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
    mean_cost,
    trace_guided_repair,
)

RESULTS_DIR = os.path.join(_PROJECT_ROOT, "safe_gta", "results")
CKPT_DIR = os.path.join(_PROJECT_ROOT, "safe_gta", "checkpoints")

LATERAL_POS_IDX = 4


def select_repair_example(joints, dataset_costs, meta, ddpm, critic, start_t,
                          beta, search_n, unsafe_min_cost, seed, device):
    total_cost = dataset_costs.sum(axis=1)
    rng = np.random.default_rng(seed)
    pool = np.where(total_cost > unsafe_min_cost)[0]
    if len(pool) == 0:
        pool = np.arange(len(joints))

    candidate_count = min(search_n, len(pool))
    candidate_idx = rng.choice(pool, size=candidate_count, replace=False)
    print(f"Searching {candidate_count} candidate trajectories...")

    batch = joints[candidate_idx]
    batch_norm = normalize(batch, meta["norm_stats"], meta["n_features"]).astype(np.float32)
    capture_steps = {20, 15, 10, 5, 0}
    capture_steps = {t for t in capture_steps if t < start_t}

    torch.manual_seed(seed)
    snapshots_batch = trace_guided_repair(
        ddpm,
        critic,
        batch_norm,
        start_t=start_t,
        beta=beta,
        capture_steps=capture_steps,
    )

    original_costs = critic(
        torch.tensor(snapshots_batch["original"], dtype=torch.float32, device=device)
    ).mean(dim=1).detach().cpu().numpy()
    final_costs = critic(
        torch.tensor(snapshots_batch["final"], dtype=torch.float32, device=device)
    ).mean(dim=1).detach().cpu().numpy()
    improvement = original_costs - final_costs
    best_local = int(np.argmax(improvement)) if np.any(improvement > 0) else int(np.argmin(final_costs))
    index = int(candidate_idx[best_local])

    print(
        "Selected trajectory: "
        f"index={index}  dataset_total_cost={total_cost[index]:.3f}  "
        f"original_critic={original_costs[best_local]:.4f}  "
        f"final_critic={final_costs[best_local]:.4f}  "
        f"improvement={improvement[best_local]:.4f}"
    )

    snapshots = {label: arr[best_local] for label, arr in snapshots_batch.items()}
    return index, total_cost, snapshots


def route_xy(raw):
    steps = raw.shape[0]
    x = np.linspace(0.0, steps - 1, steps)

    # The raw observation lateral feature is near 0.5. Center and scale it so
    # small lane-relative changes become visible in a top-down view.
    y = (raw[:, LATERAL_POS_IDX] - 0.5) * 100.0
    return x, y


def plot_topdown(raw_snapshots, costs, source_total_cost, out_path,
                 title_suffix):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    preferred_labels = [
        "original",
        "noisy t=25",
        "guided t=20",
        "guided t=15",
        "guided t=10",
        "guided t=5",
        "guided t=0",
        "final",
    ]
    labels = [label for label in preferred_labels if label in raw_snapshots]

    routes = {label: route_xy(raw_snapshots[label]) for label in labels}
    all_y = np.concatenate([xy[1] for xy in routes.values()])
    y_abs = max(1.2, float(np.max(np.abs(all_y))) * 1.25)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2.5, 1])
    fig.suptitle(
        f"MetaDrive Bird-view Route Proxy {title_suffix}",
        fontsize=15,
        fontweight="bold",
    )

    ax = axes[0]
    road_x0, road_x1 = -1.0, 50.0
    road_y0, road_h = -y_abs, 2.0 * y_abs
    ax.add_patch(
        Rectangle(
            (road_x0, road_y0),
            road_x1 - road_x0,
            road_h,
            facecolor="#ECECEC",
            edgecolor="#B8B8B8",
            linewidth=1.0,
            zorder=0,
        )
    )
    ax.axhline(0.0, color="#666666", linestyle=(0, (6, 5)), linewidth=1.2, alpha=0.7)
    ax.axhline(y_abs * 0.72, color="#A0A0A0", linewidth=1.0, alpha=0.65)
    ax.axhline(-y_abs * 0.72, color="#A0A0A0", linewidth=1.0, alpha=0.65)

    style = {
        "original": dict(color="#333333", linestyle="--", linewidth=2.6, alpha=0.95),
        "noisy t=25": dict(color="#D95F02", linestyle=":", linewidth=2.0, alpha=0.8),
        "guided t=20": dict(color="#6BAED6", linestyle="-", linewidth=1.6, alpha=0.55),
        "guided t=15": dict(color="#4292C6", linestyle="-", linewidth=1.6, alpha=0.6),
        "guided t=10": dict(color="#2CA25F", linestyle="-", linewidth=1.8, alpha=0.7),
        "guided t=5": dict(color="#31A354", linestyle="-", linewidth=2.0, alpha=0.8),
        "guided t=0": dict(color="#74C476", linestyle="-", linewidth=2.0, alpha=0.85),
        "final": dict(color="#E6C700", linestyle="-", linewidth=3.0, alpha=0.95),
    }

    for label in labels:
        x, y = routes[label]
        critic_text = f"{costs[label]:.3f}" if label in costs else "n/a"
        if label == "original":
            legend = f"original (dataset cost={source_total_cost:.1f}, critic={critic_text})"
        else:
            legend = f"{label} (critic={critic_text})"
        ax.plot(x, y, label=legend, zorder=2, **style.get(label, {}))
        ax.scatter([x[0]], [y[0]], s=38, color=style.get(label, {}).get("color", "#333333"), zorder=3)
        ax.scatter([x[-1]], [y[-1]], s=44, marker="s",
                   color=style.get(label, {}).get("color", "#333333"), zorder=3)

    ax.text(0.01, 0.95, "start", transform=ax.transAxes, fontsize=9, color="#333333")
    ax.text(0.94, 0.08, "end", transform=ax.transAxes, fontsize=9, color="#333333")
    ax.set_xlim(road_x0, road_x1)
    ax.set_ylim(road_y0, -road_y0)
    ax.set_xlabel("Trajectory progress step")
    ax.set_ylabel("Scaled lane-relative lateral offset")
    ax.set_title("Original vs repaired trajectory over a lane-relative road view")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax2 = axes[1]
    cost_labels = [label for label in labels if label in costs]
    cost_vals = [costs[label] for label in cost_labels]
    bar_colors = [style.get(label, {}).get("color", "#1F77B4") for label in cost_labels]
    ax2.bar(range(len(cost_vals)), cost_vals, color=bar_colors, alpha=0.9)
    ax2.plot(range(len(cost_vals)), cost_vals, color="#333333", marker="o", linewidth=1.2)
    ax2.set_xticks(range(len(cost_labels)))
    ax2.set_xticklabels(cost_labels, rotation=25, ha="right")
    ax2.set_ylabel("Mean critic cost")
    ax2.set_title("Safety Critic score drops during guided repair")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    note = (
        "Note: this is a lane-relative bird-view proxy from offline observations, "
        "not a replay of MetaDrive world coordinates."
    )
    fig.text(0.5, 0.01, note, ha="center", fontsize=9, color="#555555")

    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main(start_t=25, beta=2.0, index=None, search_n=2000,
         unsafe_min_cost=1.0, seed=7, device=None, output=None):
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

    if index is None:
        index, total_cost, norm_snapshots = select_repair_example(
            joints,
            dataset_costs,
            meta,
            ddpm,
            critic,
            start_t,
            beta,
            search_n,
            unsafe_min_cost,
            seed,
            device,
        )
    else:
        total_cost = dataset_costs.sum(axis=1)
        print(f"Using requested trajectory index: {index}")
        x = joints[index]
        x_norm = normalize(x[None], meta["norm_stats"], meta["n_features"]).astype(np.float32)
        capture_steps = {20, 15, 10, 5, 0}
        capture_steps = {t for t in capture_steps if t < start_t}
        torch.manual_seed(seed)
        snapshots_batch = trace_guided_repair(
            ddpm,
            critic,
            x_norm,
            start_t=start_t,
            beta=beta,
            capture_steps=capture_steps,
        )
        norm_snapshots = {label: arr[0] for label, arr in snapshots_batch.items()}

    raw_snapshots = {
        label: denormalize(arr[None], meta["norm_stats"], meta["n_features"])[0]
        for label, arr in norm_snapshots.items()
    }
    costs = {label: mean_cost(critic, arr, device) for label, arr in norm_snapshots.items()}

    out_path = output or os.path.join(RESULTS_DIR, "guided_repair_topdown_route.png")
    plot_topdown(
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
    parser.add_argument("--search_n", type=int, default=2000)
    parser.add_argument("--unsafe_min_cost", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    main(
        start_t=args.start_t,
        beta=args.beta,
        index=args.index,
        search_n=args.search_n,
        unsafe_min_cost=args.unsafe_min_cost,
        seed=args.seed,
        device=args.device,
        output=args.output,
    )
