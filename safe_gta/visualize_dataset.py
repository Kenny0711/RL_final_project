"""
Generate the two dataset figures kept for the final report.

Outputs:
  safe_gta/results/figure/dataset_safe_vs_dangerous.png
  safe_gta/results/figure/dataset_actions.png
"""
from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "safe_gta", "data")
FIGURE_DIR = os.path.join(PROJECT_ROOT, "safe_gta", "results", "figure")


def find_hdf5() -> str:
    for name in [
        "metadrive_mediumsparse.hdf5",
        "metadrive_mediummean.hdf5",
        "metadrive_mediumdense.hdf5",
    ]:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            return path
    matches = sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))
    if not matches:
        raise FileNotFoundError("No HDF5 file found in safe_gta/data/")
    return matches[0]


def load_hdf5(seq_len: int = 50):
    import h5py

    path = find_hdf5()
    print(f"Loading: {path}")
    with h5py.File(path, "r") as f:
        obs = np.array(f["observations"][:], dtype=np.float32)
        act = np.array(f["actions"][:], dtype=np.float32)
        cost = np.array(f["costs"][:], dtype=np.float32)
        rew = np.array(f["rewards"][:], dtype=np.float32)
        done = np.array(f["terminals"][:], dtype=np.float32)
        if "timeouts" in f:
            done = np.logical_or(done, f["timeouts"][:]).astype(np.float32)

    traj_obs, traj_act, traj_cost, traj_rew = [], [], [], []
    ends = list(np.where(done > 0.5)[0]) + [len(obs) - 1]
    start = 0
    for end in ends:
        eo = obs[start:end + 1]
        ea = act[start:end + 1]
        ec = cost[start:end + 1]
        er = rew[start:end + 1]
        for i in range(0, len(eo) - seq_len + 1, seq_len):
            traj_obs.append(eo[i:i + seq_len])
            traj_act.append(ea[i:i + seq_len])
            traj_cost.append(ec[i:i + seq_len])
            traj_rew.append(er[i:i + seq_len])
        start = end + 1

    return (
        np.stack(traj_obs),
        np.stack(traj_act),
        np.stack(traj_cost),
        np.stack(traj_rew),
    )


def style_axis(ax) -> None:
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    os.makedirs(FIGURE_DIR, exist_ok=True)
    traj_obs, traj_act, traj_cost, traj_rew = load_hdf5()
    total_cost = traj_cost.sum(axis=1)

    print(f"Total trajectories : {len(traj_obs)}")
    print(f"Cost range         : {total_cost.min():.2f} ~ {total_cost.max():.2f}")
    print(f"Safe (cost=0)      : {(total_cost == 0).sum()}")
    print(f"Dangerous (cost>0) : {(total_cost > 0).sum()}")

    steps = np.arange(50)
    n_show = 5
    rng = np.random.default_rng(42)
    safe_idx = rng.choice(np.where(total_cost == 0)[0], n_show, replace=False)
    danger_idx = np.argsort(total_cost)[::-1][:n_show]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("MetaDrive Dataset - Safe vs Dangerous Trajectories",
                 fontsize=14, fontweight="bold")
    features = [(7, "Speed"), (4, "Lateral Position")]
    for row, (feature_idx, feature_name) in enumerate(features):
        for idx in safe_idx:
            axes[row][0].plot(steps, traj_obs[idx, :, feature_idx],
                              color="#55A868", alpha=0.65, lw=1.2)
        axes[row][0].set_title(f"Safe Trajectories (cost=0), n={n_show}", fontsize=11)
        axes[row][0].set_ylabel(feature_name)
        axes[row][0].set_xlabel("Step")
        style_axis(axes[row][0])

        for idx in danger_idx:
            axes[row][1].plot(steps, traj_obs[idx, :, feature_idx],
                              color="#C44E52", alpha=0.65, lw=1.2)
        axes[row][1].set_title(f"Dangerous Trajectories (highest cost), n={n_show}", fontsize=11)
        axes[row][1].set_ylabel(feature_name)
        axes[row][1].set_xlabel("Step")
        style_axis(axes[row][1])

    plt.tight_layout()
    out1 = os.path.join(FIGURE_DIR, "dataset_safe_vs_dangerous.png")
    plt.savefig(out1, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out1}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 6))
    fig.suptitle("MetaDrive Dataset - Actions: Steering & Throttle",
                 fontsize=14, fontweight="bold")
    actions = [(0, "Steering"), (1, "Throttle/Brake")]
    for row, (action_idx, action_name) in enumerate(actions):
        for idx in safe_idx:
            axes[row][0].plot(steps, traj_act[idx, :, action_idx],
                              color="#55A868", alpha=0.55, lw=1.2)
        axes[row][0].set_title(f"Safe - {action_name}", fontsize=11)
        axes[row][0].set_ylabel(action_name)
        axes[row][0].set_xlabel("Step")
        axes[row][0].axhline(0, color="k", lw=0.8, ls="--")
        style_axis(axes[row][0])

        for idx in danger_idx:
            axes[row][1].plot(steps, traj_act[idx, :, action_idx],
                              color="#C44E52", alpha=0.55, lw=1.2)
        axes[row][1].set_title(f"Dangerous - {action_name}", fontsize=11)
        axes[row][1].set_ylabel(action_name)
        axes[row][1].set_xlabel("Step")
        axes[row][1].axhline(0, color="k", lw=0.8, ls="--")
        style_axis(axes[row][1])

    plt.tight_layout()
    out2 = os.path.join(FIGURE_DIR, "dataset_actions.png")
    plt.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")
    print("\nDataset figures saved to safe_gta/results/figure/")


if __name__ == "__main__":
    main()
