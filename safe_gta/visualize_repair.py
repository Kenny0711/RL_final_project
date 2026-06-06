"""
Visualize MetaDrive trajectory repair by the diffusion model.
Picks high-cost (dangerous) trajectories, repairs them, and plots before/after.

Run: python -m safe_gta.visualize_repair
"""
import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from safe_gta.diffusion.noise_schedule import NoiseSchedule
from safe_gta.diffusion.unet1d import TemporalUNet
from safe_gta.diffusion.ddpm import DDPM

RESULTS_DIR = os.path.join(_PROJECT_ROOT, "safe_gta", "results")
DATA_DIR    = os.path.join(_PROJECT_ROOT, "safe_gta", "data")
CKPT_DIR    = os.path.join(_PROJECT_ROOT, "safe_gta", "checkpoints")

# ── MetaDrive obs feature names (23D) ──────────────────────────────────────
# Index reference for SafeMetaDrive observations:
# 0-1:  ego velocity (vx, vy)
# 2:    ego heading (cos)
# 3:    ego heading (sin)
# 4:    lateral position in lane (positive = right of center)
# 5:    distance to left lane boundary
# 6:    distance to right lane boundary
# 7:    speed (scalar)
# 8-22: lidar / surrounding vehicle readings

OBS_NAMES = [
    "vel_x", "vel_y", "heading_cos", "heading_sin",
    "lateral_pos", "dist_left", "dist_right", "speed",
    "lidar_0", "lidar_1", "lidar_2", "lidar_3", "lidar_4",
    "lidar_5", "lidar_6", "lidar_7", "lidar_8", "lidar_9",
    "lidar_10", "lidar_11", "lidar_12", "lidar_13", "lidar_14",
]

# Features we'll plot (index, name, color)
PLOT_FEATS = [
    (7,  "Speed",            "#4C72B0"),
    (4,  "Lateral Position", "#C44E52"),
    (5,  "Dist to Left",     "#55A868"),
    (6,  "Dist to Right",    "#DD8452"),
]


# ── Load data ───────────────────────────────────────────────────────────────

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
        obs  = f["observations"][:]
        act  = f["actions"][:]
        cost = f["costs"][:]
        done = f["terminals"][:]
        if "timeouts" in f:
            done = np.logical_or(done, f["timeouts"][:]).astype(np.float32)

    obs  = np.array(obs,  dtype=np.float32)
    act  = np.array(act,  dtype=np.float32)
    cost = np.array(cost, dtype=np.float32)
    done = np.array(done, dtype=np.float32)

    # Segment into (N, seq_len, dim) episodes
    traj_obs, traj_act, traj_cost = [], [], []
    ends = list(np.where(done > 0.5)[0]) + [len(obs) - 1]
    start = 0
    for end in ends:
        eo, ea, ec = obs[start:end+1], act[start:end+1], cost[start:end+1]
        for i in range(0, len(eo) - seq_len + 1, seq_len):
            traj_obs.append(eo[i:i+seq_len])
            traj_act.append(ea[i:i+seq_len])
            traj_cost.append(ec[i:i+seq_len])
        start = end + 1

    traj_obs  = np.stack(traj_obs)
    traj_act  = np.stack(traj_act)
    traj_cost = np.stack(traj_cost)
    print(f"  Trajectories: {traj_obs.shape}  (N, seq_len, obs_dim)")
    return traj_obs, traj_act, traj_cost


# ── Load diffusion model ────────────────────────────────────────────────────

def load_ddpm(device):
    ckpt_path = os.path.join(CKPT_DIR, "diffusion_metadrive_final.pt")
    payload   = torch.load(ckpt_path, map_location=device, weights_only=False)
    T          = payload.get("T", 200)
    n_features = payload.get("n_features", 25)
    base_ch    = payload.get("base_ch", 64)
    norm_stats = payload.get("norm_stats", {})
    seq_len    = payload.get("seq_len", 50)

    model    = TemporalUNet(seq_len=seq_len, n_features=n_features, base_ch=base_ch)
    schedule = NoiseSchedule(T=T, device=device)
    ddpm     = DDPM(model, schedule, device=device)
    ddpm.load_checkpoint(ckpt_path)
    model.eval()
    print(f"Loaded diffusion model  n_features={n_features}  T={T}")
    return ddpm, norm_stats, n_features


# ── Normalize / Denormalize ─────────────────────────────────────────────────

def normalize(trajs, norm_stats, n_features):
    feat_min   = np.array(norm_stats["feat_min"][:n_features], dtype=np.float32)
    feat_max   = np.array(norm_stats["feat_max"][:n_features], dtype=np.float32)
    feat_range = np.where(feat_max - feat_min > 1e-6, feat_max - feat_min, 1.0)
    return 2.0 * (trajs - feat_min) / feat_range - 1.0

def denormalize(trajs, norm_stats, n_features):
    feat_min   = np.array(norm_stats["feat_min"][:n_features], dtype=np.float32)
    feat_max   = np.array(norm_stats["feat_max"][:n_features], dtype=np.float32)
    feat_range = np.where(feat_max - feat_min > 1e-6, feat_max - feat_min, 1.0)
    return (trajs + 1.0) / 2.0 * feat_range + feat_min


# ── Main ────────────────────────────────────────────────────────────────────

def main(n_examples=5, start_t=150):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    traj_obs, traj_act, traj_cost = load_hdf5(seq_len=50)
    ddpm, norm_stats, n_features  = load_ddpm(device)

    # Pick the n_examples trajectories with the highest total cost (most dangerous)
    total_cost = traj_cost.sum(axis=1)          # (N,)
    dangerous_idx = np.argsort(total_cost)[::-1][:n_examples]
    print(f"\nPicked {n_examples} most dangerous trajectories")
    print(f"  Total costs: {total_cost[dangerous_idx].tolist()}")

    # Build joint obs+act array, normalize, repair
    trajs_joint = np.concatenate([traj_obs, traj_act], axis=-1)  # (N, 50, 25)
    bad_batch   = trajs_joint[dangerous_idx, :, :n_features]      # (n_examples, 50, n_features)
    bad_norm    = normalize(bad_batch, norm_stats, n_features)

    print(f"Repairing with SDEdit (start_t={start_t})...")
    repaired_norm = ddpm.repair(bad_norm, start_t=start_t)        # (n_examples, 50, n_features)
    repaired_raw  = denormalize(repaired_norm, norm_stats, n_features)

    bad_raw = bad_batch  # already in original scale

    # ── Plot ────────────────────────────────────────────────────────────────
    n_feats = len(PLOT_FEATS)
    fig = plt.figure(figsize=(5 * n_examples, 4 * n_feats))
    fig.suptitle(
        f"MetaDrive Trajectory Repair (SDEdit start_t={start_t})\n"
        "Red = original dangerous trajectory   Blue = repaired",
        fontsize=13, fontweight="bold", y=1.01
    )

    gs = gridspec.GridSpec(n_feats, n_examples, hspace=0.5, wspace=0.35)
    steps = np.arange(50)

    for row, (feat_idx, feat_name, color) in enumerate(PLOT_FEATS):
        for col, traj_idx in enumerate(dangerous_idx):
            ax = fig.add_subplot(gs[row, col])

            bad_vals      = bad_raw[col, :, feat_idx]
            repaired_vals = repaired_raw[col, :, feat_idx]

            ax.plot(steps, bad_vals,      color="#C44E52", lw=1.5,
                    label="Original", alpha=0.85)
            ax.plot(steps, repaired_vals, color="#4C72B0", lw=1.5,
                    label="Repaired", alpha=0.85, linestyle="--")

            if row == 0:
                ax.set_title(f"Traj #{col+1}\n(cost={total_cost[traj_idx]:.1f})",
                             fontsize=9)
            if col == 0:
                ax.set_ylabel(feat_name, fontsize=9)
            if row == n_feats - 1:
                ax.set_xlabel("Step", fontsize=8)
            if row == 0 and col == n_examples - 1:
                ax.legend(fontsize=7, loc="upper right")

            ax.grid(True, alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "metadrive_repair_comparison.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out_path}")

    # ── Summary stats ───────────────────────────────────────────────────────
    print("\n=== Repair Summary ===")
    for fi, (feat_idx, feat_name, _) in enumerate(PLOT_FEATS):
        bad_std  = bad_raw[:, :, feat_idx].std(axis=1).mean()
        rep_std  = repaired_raw[:, :, feat_idx].std(axis=1).mean()
        print(f"  {feat_name:20s}  std before={bad_std:.4f}  after={rep_std:.4f}  "
              f"change={rep_std/bad_std:.2f}x")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of example trajectories")
    parser.add_argument("--start_t", type=int, default=150)
    args = parser.parse_args()
    main(n_examples=args.n, start_t=args.start_t)
