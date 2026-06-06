"""
Baseline 2 — GTA without safety constraints (MetaDrive 25D version).
Generates an augmented dataset using the trained MetaDrive diffusion model,
with cost labels forced to 0.0 (no safety filtering).

Run:
  python -m safe_gta.baseline2.generate_augmented
  python -m safe_gta.baseline2.generate_augmented \
      --checkpoint safe_gta/checkpoints/diffusion_metadrive_final.pt \
      --n_augmented 5000 \
      --output safe_gta/data/augmented_no_safety.pkl
"""
import argparse
import glob
import os
import sys
import pickle

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from safe_gta.diffusion.noise_schedule import NoiseSchedule
from safe_gta.diffusion.unet1d import TemporalUNet
from safe_gta.diffusion.ddpm import DDPM

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_source_trajectories(hdf5_path: str, seq_len: int = 50):
    """Load MetaDrive HDF5 and return (N, seq_len, obs_dim+act_dim) array."""
    import h5py
    print(f"Loading source data: {hdf5_path}")
    with h5py.File(hdf5_path, "r") as f:
        obs  = f["observations"][:]
        act  = f["actions"][:]
        done = f["terminals"][:]
        if "timeouts" in f:
            done = np.logical_or(done, f["timeouts"][:]).astype(np.float32)

    obs  = np.array(obs,  dtype=np.float32)
    act  = np.array(act,  dtype=np.float32)
    done = np.array(done, dtype=np.float32)

    if obs.ndim == 3:
        trajs = np.concatenate([obs, act], axis=-1)
    else:
        traj_obs_list, traj_act_list = [], []
        N = len(obs)
        ends = np.where(done > 0.5)[0].tolist()
        ends.append(N - 1)
        start = 0
        for end in ends:
            eo, ea = obs[start:end+1], act[start:end+1]
            for i in range(0, len(eo) - seq_len + 1, seq_len):
                traj_obs_list.append(eo[i:i+seq_len])
                traj_act_list.append(ea[i:i+seq_len])
            start = end + 1
        if not traj_obs_list:
            for i in range(0, N - seq_len + 1, seq_len):
                traj_obs_list.append(obs[i:i+seq_len])
                traj_act_list.append(act[i:i+seq_len])
        trajs = np.concatenate([
            np.stack(traj_obs_list),
            np.stack(traj_act_list)
        ], axis=-1)

    print(f"  Source trajectories: {trajs.shape}")
    return trajs, obs.shape[-1], act.shape[-1]


def find_hdf5(data_dir: str):
    for name in ["metadrive_mediumsparse.hdf5", "metadrive_mediummean.hdf5",
                 "metadrive_mediumdense.hdf5"]:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return p
    matches = glob.glob(os.path.join(data_dir, "*.hdf5"))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def generate_augmented_dataset(
    ddpm_checkpoint: str = None,
    n_augmented: int = 5000,
    start_t: int = 150,
    output_path: str = None,
    seq_len: int = 50,
    batch_size: int = 64,
    device: str = None,
):
    """
    Repair MetaDrive trajectories with SDEdit.
    Cost labels forced to 0.0 (Baseline 2: no safety filtering).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = os.path.join(_PROJECT_ROOT, "safe_gta", "data")
    ckpt_dir = os.path.join(_PROJECT_ROOT, "safe_gta", "checkpoints")

    # Auto-find checkpoint
    if ddpm_checkpoint is None:
        candidates = [
            os.path.join(ckpt_dir, "diffusion_metadrive_final.pt"),
        ]
        for c in candidates:
            if os.path.exists(c):
                ddpm_checkpoint = c
                break
        if ddpm_checkpoint is None:
            raise FileNotFoundError(
                "No diffusion checkpoint found. Run:\n"
                "  python -m safe_gta.train_diffusion"
            )

    if output_path is None:
        output_path = os.path.join(data_dir, "augmented_no_safety.pkl")

    # Load checkpoint metadata
    payload = torch.load(ddpm_checkpoint, map_location=device, weights_only=False)
    T          = payload.get("T", 200)
    n_features = payload.get("n_features", 2)
    obs_dim    = payload.get("obs_dim", 2)
    act_dim    = payload.get("act_dim", 2)
    base_ch    = payload.get("base_ch", 32)
    norm_stats = payload.get("norm_stats", None)
    print(f"Checkpoint: {ddpm_checkpoint}")
    print(f"  n_features={n_features}  obs_dim={obs_dim}  act_dim={act_dim}  T={T}")

    # Load model
    model    = TemporalUNet(seq_len=seq_len, n_features=n_features, base_ch=base_ch)
    schedule = NoiseSchedule(T=T, device=device)
    ddpm     = DDPM(model, schedule, device=device)
    ddpm.load_checkpoint(ddpm_checkpoint)
    model.eval()

    # Load source trajectories
    hdf5_path = find_hdf5(data_dir)
    if hdf5_path:
        source_trajs, src_obs_dim, src_act_dim = load_source_trajectories(hdf5_path, seq_len)
    else:
        print("No HDF5 found, generating random source trajectories.")
        rng = np.random.default_rng(0)
        source_trajs = rng.normal(0, 0.3, (3000, seq_len, n_features)).astype(np.float32)
        src_obs_dim, src_act_dim = obs_dim, act_dim

    # Normalize source to [-1, 1] using checkpoint's norm_stats
    if norm_stats and "feat_min" in norm_stats:
        feat_min   = np.array(norm_stats["feat_min"], dtype=np.float32)
        feat_max   = np.array(norm_stats["feat_max"], dtype=np.float32)
        feat_range = np.where(feat_max - feat_min > 1e-6, feat_max - feat_min, 1.0)
        src = source_trajs[:, :, :n_features]
        src_norm = 2.0 * (src - feat_min[:n_features]) / feat_range[:n_features] - 1.0
    else:
        src_norm = source_trajs[:, :, :n_features].copy()

    # Repair in batches
    print(f"Repairing {n_augmented} trajectories (SDEdit start_t={start_t})...")
    rng = np.random.default_rng(0)
    repaired_list = []
    n_done = 0
    while n_done < n_augmented:
        bs = min(batch_size, n_augmented - n_done)
        idx = rng.integers(0, len(src_norm), bs)
        batch = src_norm[idx]
        repaired_norm = ddpm.repair(batch, start_t=start_t)
        repaired_list.append(repaired_norm)
        n_done += bs
        if n_done % 500 == 0 or n_done == n_augmented:
            print(f"  {n_done}/{n_augmented}")

    repaired = np.concatenate(repaired_list, axis=0)[:n_augmented]

    # Denormalize back to original scale
    if norm_stats and "feat_min" in norm_stats:
        feat_min   = np.array(norm_stats["feat_min"][:n_features], dtype=np.float32)
        feat_max   = np.array(norm_stats["feat_max"][:n_features], dtype=np.float32)
        feat_range = np.where(feat_max - feat_min > 1e-6, feat_max - feat_min, 1.0)
        repaired = (repaired + 1.0) / 2.0 * feat_range + feat_min

    # Split back into obs / act
    aug_obs = repaired[:, :, :obs_dim]
    aug_act = repaired[:, :, obs_dim:obs_dim + act_dim]
    if aug_act.shape[-1] == 0:
        aug_act = np.zeros((len(repaired), seq_len, act_dim), dtype=np.float32)

    # Build dataset — cost FORCED TO 0.0 (Baseline 2 intentional)
    terminals = np.zeros((len(repaired), seq_len), dtype=np.float32)
    terminals[:, -1] = 1.0

    dataset = {
        "observations": aug_obs.astype(np.float32),
        "actions":      aug_act.astype(np.float32),
        "rewards":      np.ones( (len(repaired), seq_len), dtype=np.float32),
        "costs":        np.zeros((len(repaired), seq_len), dtype=np.float32),
        "terminals":    terminals,
        "n_augmented":  n_augmented,
        "obs_dim":      obs_dim,
        "act_dim":      act_dim,
        "source_checkpoint": ddpm_checkpoint,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"\nAugmented dataset saved: {output_path}")
    print(f"  obs shape: {aug_obs.shape}  act shape: {aug_act.shape}")
    print("  Cost labels: 0.0 (no safety filtering -- Baseline 2)")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--n_augmented", type=int, default=5000)
    parser.add_argument("--start_t", type=int, default=150)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    generate_augmented_dataset(
        ddpm_checkpoint=args.checkpoint,
        n_augmented=args.n_augmented,
        start_t=args.start_t,
        output_path=args.output,
        device=args.device,
    )
