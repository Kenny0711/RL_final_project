"""
Visualize raw MetaDrive dataset trajectories.
Shows safe vs dangerous trajectories side by side.

Run: python -m safe_gta.visualize_dataset
"""
import os, sys, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR   = os.path.join(_PROJECT_ROOT, "safe_gta", "results")
DATA_DIR      = os.path.join(_PROJECT_ROOT, "safe_gta", "data")


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


def load_hdf5(seq_len=50):
    import h5py
    path = find_hdf5()
    print(f"Loading: {path}")
    with h5py.File(path, "r") as f:
        obs  = np.array(f["observations"][:], dtype=np.float32)
        act  = np.array(f["actions"][:],      dtype=np.float32)
        cost = np.array(f["costs"][:],        dtype=np.float32)
        rew  = np.array(f["rewards"][:],      dtype=np.float32)
        done = np.array(f["terminals"][:],    dtype=np.float32)
        if "timeouts" in f:
            done = np.logical_or(done, f["timeouts"][:]).astype(np.float32)

    traj_obs, traj_act, traj_cost, traj_rew = [], [], [], []
    ends  = list(np.where(done > 0.5)[0]) + [len(obs) - 1]
    start = 0
    for end in ends:
        eo, ea, ec, er = obs[start:end+1], act[start:end+1], cost[start:end+1], rew[start:end+1]
        for i in range(0, len(eo) - seq_len + 1, seq_len):
            traj_obs.append(eo[i:i+seq_len])
            traj_act.append(ea[i:i+seq_len])
            traj_cost.append(ec[i:i+seq_len])
            traj_rew.append(er[i:i+seq_len])
        start = end + 1

    return (np.stack(traj_obs), np.stack(traj_act),
            np.stack(traj_cost), np.stack(traj_rew))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    traj_obs, traj_act, traj_cost, traj_rew = load_hdf5()
    total_cost = traj_cost.sum(axis=1)
    total_rew  = traj_rew.sum(axis=1)

    print(f"Total trajectories : {len(traj_obs)}")
    print(f"Cost range         : {total_cost.min():.2f} ~ {total_cost.max():.2f}")
    print(f"Safe (cost=0)      : {(total_cost == 0).sum()}")
    print(f"Dangerous (cost>0) : {(total_cost > 0).sum()}")

    steps = np.arange(50)
    N_SHOW = 5  # trajectories per group

    # Pick safe (cost=0) and dangerous (cost highest)
    safe_idx = np.where(total_cost == 0)[0]
    rng = np.random.default_rng(42)
    safe_idx = rng.choice(safe_idx, N_SHOW, replace=False)
    dang_idx = np.argsort(total_cost)[::-1][:N_SHOW]

    # ── Figure 1: Speed & Lateral Position ──────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("MetaDrive Dataset — Safe vs Dangerous Trajectories", fontsize=14, fontweight="bold")

    FEAT = [(7, "Speed"), (4, "Lateral Position")]
    for row, (fi, fname) in enumerate(FEAT):
        ax_safe = axes[row][0]
        ax_dang = axes[row][1]

        for i, idx in enumerate(safe_idx):
            alpha = 0.5 + 0.1 * i
            ax_safe.plot(steps, traj_obs[idx, :, fi],
                         color="#55A868", alpha=alpha, lw=1.2)
        ax_safe.set_title(f"Safe Trajectories (cost=0)  n={N_SHOW}", fontsize=11)
        ax_safe.set_ylabel(fname, fontsize=11)
        ax_safe.set_xlabel("Step")
        ax_safe.grid(True, alpha=0.3)
        ax_safe.spines["top"].set_visible(False)
        ax_safe.spines["right"].set_visible(False)

        for i, idx in enumerate(dang_idx):
            alpha = 0.5 + 0.1 * i
            ax_dang.plot(steps, traj_obs[idx, :, fi],
                         color="#C44E52", alpha=alpha, lw=1.2)
        ax_dang.set_title(f"Dangerous Trajectories (highest cost)  n={N_SHOW}", fontsize=11)
        ax_dang.set_ylabel(fname, fontsize=11)
        ax_dang.set_xlabel("Step")
        ax_dang.grid(True, alpha=0.3)
        ax_dang.spines["top"].set_visible(False)
        ax_dang.spines["right"].set_visible(False)

    plt.tight_layout()
    out1 = os.path.join(RESULTS_DIR, "dataset_safe_vs_dangerous.png")
    plt.savefig(out1, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out1}")

    # ── Figure 2: Actions (steering & throttle) ──────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 6))
    fig.suptitle("MetaDrive Dataset — Actions: Steering & Throttle", fontsize=14, fontweight="bold")
    ACT = [(0, "Steering"), (1, "Throttle/Brake")]
    for row, (ai, aname) in enumerate(ACT):
        for i, idx in enumerate(safe_idx):
            axes[row][0].plot(steps, traj_act[idx, :, ai], color="#55A868", alpha=0.55, lw=1.2)
        axes[row][0].set_title(f"Safe — {aname}", fontsize=11)
        axes[row][0].set_ylabel(aname); axes[row][0].set_xlabel("Step")
        axes[row][0].axhline(0, color="k", lw=0.8, ls="--")
        axes[row][0].grid(True, alpha=0.3)
        axes[row][0].spines["top"].set_visible(False)
        axes[row][0].spines["right"].set_visible(False)

        for i, idx in enumerate(dang_idx):
            axes[row][1].plot(steps, traj_act[idx, :, ai], color="#C44E52", alpha=0.55, lw=1.2)
        axes[row][1].set_title(f"Dangerous — {aname}", fontsize=11)
        axes[row][1].set_ylabel(aname); axes[row][1].set_xlabel("Step")
        axes[row][1].axhline(0, color="k", lw=0.8, ls="--")
        axes[row][1].grid(True, alpha=0.3)
        axes[row][1].spines["top"].set_visible(False)
        axes[row][1].spines["right"].set_visible(False)

    plt.tight_layout()
    out2 = os.path.join(RESULTS_DIR, "dataset_actions.png")
    plt.savefig(out2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")

    # ── Figure 3: Cost distribution histogram ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("MetaDrive Dataset — Cost & Reward Distribution", fontsize=13, fontweight="bold")

    axes[0].hist(total_cost, bins=40, color="#C44E52", alpha=0.8, edgecolor="white")
    axes[0].set_xlabel("Total Cost per Trajectory", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("Cost Distribution\n(0 = safe, >0 = has violations)", fontsize=11)
    axes[0].axvline(0, color="k", lw=1.5, ls="--", label="Safe boundary")
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].hist(total_rew, bins=40, color="#4C72B0", alpha=0.8, edgecolor="white")
    axes[1].set_xlabel("Total Reward per Trajectory", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title("Reward Distribution", fontsize=11)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    out3 = os.path.join(RESULTS_DIR, "dataset_distribution.png")
    plt.savefig(out3, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out3}")

    print("\nAll figures saved to safe_gta/results/")


if __name__ == "__main__":
    main()
