"""
Train Diffusion Model on real MetaDrive HDF5 dataset.
Supports obs_dim=23 + act_dim=2 = 25 features.

Run:
  python -m safe_gta.train_diffusion
  python -m safe_gta.train_diffusion --epochs 200 --seq_len 50
"""
import argparse
import os
import sys
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from safe_gta.diffusion.noise_schedule import NoiseSchedule
from safe_gta.diffusion.unet1d import TemporalUNet
from safe_gta.diffusion.ddpm import DDPM


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_hdf5_trajectories(hdf5_path: str, seq_len: int = 50):
    """
    Load MetaDrive HDF5 dataset and segment into (N, seq_len, obs_dim+act_dim) trajectories.
    Returns: (trajs, obs_dim, act_dim, norm_stats)
    """
    import h5py
    print(f"Loading: {hdf5_path}")
    with h5py.File(hdf5_path, "r") as f:
        keys = list(f.keys())
        print(f"  HDF5 keys: {keys}")
        obs  = f["observations"][:]   # (N, obs_dim) or (N, seq_len, obs_dim)
        act  = f["actions"][:]
        done = f["terminals"][:]
        if "timeouts" in f:
            done = np.logical_or(done, f["timeouts"][:]).astype(np.float32)

    obs = np.array(obs, dtype=np.float32)
    act = np.array(act, dtype=np.float32)
    done = np.array(done, dtype=np.float32)

    print(f"  Raw shapes  obs={obs.shape}  act={act.shape}  done={done.shape}")

    obs_dim = obs.shape[-1]
    act_dim = act.shape[-1]

    # If already in (N_episodes, T, dim) format, use directly
    if obs.ndim == 3:
        traj_obs = obs
        traj_act = act
    else:
        # Flat (N, dim) → segment into episodes using 'done' flags
        traj_obs, traj_act = _segment_episodes(obs, act, done, seq_len)

    # Concatenate obs + act into joint feature vector
    trajs = np.concatenate([traj_obs, traj_act], axis=-1)  # (N, seq_len, obs_dim+act_dim)
    print(f"  Trajectories: {trajs.shape}  n_features={trajs.shape[-1]}")

    # Normalize per-feature to [-1, 1]
    flat = trajs.reshape(-1, trajs.shape[-1])
    feat_min = flat.min(axis=0)
    feat_max = flat.max(axis=0)
    feat_range = np.where(feat_max - feat_min > 1e-6, feat_max - feat_min, 1.0)
    trajs_norm = 2.0 * (trajs - feat_min) / feat_range - 1.0

    norm_stats = {
        "feat_min": feat_min.tolist(),
        "feat_max": feat_max.tolist(),
        "obs_dim": int(obs_dim),
        "act_dim": int(act_dim),
    }
    return trajs_norm, obs_dim, act_dim, norm_stats


def _segment_episodes(obs, act, done, seq_len):
    """Segment flat transitions into fixed-length trajectories."""
    N = len(obs)
    # Find episode boundaries
    episode_ends = np.where(done > 0.5)[0].tolist()
    episode_ends.append(N - 1)

    traj_obs_list, traj_act_list = [], []
    start = 0
    for end in episode_ends:
        ep_obs = obs[start:end + 1]
        ep_act = act[start:end + 1]
        # Slice into seq_len chunks
        for i in range(0, len(ep_obs) - seq_len + 1, seq_len):
            traj_obs_list.append(ep_obs[i:i + seq_len])
            traj_act_list.append(ep_act[i:i + seq_len])
        start = end + 1

    if not traj_obs_list:
        # Fallback: just split sequentially without episode boundaries
        print("  Warning: no done flags found, splitting sequentially.")
        for i in range(0, N - seq_len + 1, seq_len):
            traj_obs_list.append(obs[i:i + seq_len])
            traj_act_list.append(act[i:i + seq_len])

    traj_obs = np.stack(traj_obs_list).astype(np.float32)
    traj_act = np.stack(traj_act_list).astype(np.float32)
    return traj_obs, traj_act


def find_hdf5(data_dir: str):
    """Find first available MetaDrive HDF5 in data_dir."""
    patterns = ["metadrive_mediumsparse.hdf5", "metadrive_mediummean.hdf5",
                "metadrive_mediumdense.hdf5", "*.hdf5"]
    for pat in patterns:
        matches = glob.glob(os.path.join(data_dir, pat))
        if matches:
            return matches[0]
    return None


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    hdf5_path: str = None,
    n_epochs: int = 200,
    batch_size: int = 64,
    lr: float = 2e-4,
    T: int = 200,
    seq_len: int = 50,
    base_ch: int = 64,
    checkpoint_dir: str = None,
    save_every: int = 50,
    device: str = None,
):
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(_PROJECT_ROOT, "safe_gta", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  T={T}  epochs={n_epochs}  batch={batch_size}")

    # Auto-find HDF5 if not specified
    if hdf5_path is None:
        data_dir = os.path.join(_PROJECT_ROOT, "safe_gta", "data")
        hdf5_path = find_hdf5(data_dir)
        if hdf5_path is None:
            raise FileNotFoundError(
                f"No HDF5 dataset found in {data_dir}.\n"
                "Run: python download_dataset.py"
            )

    # Load data
    trajs_norm, obs_dim, act_dim, norm_stats = load_hdf5_trajectories(hdf5_path, seq_len)
    n_features = obs_dim + act_dim
    print(f"n_features={n_features}  (obs={obs_dim} + act={act_dim})")
    print(f"Training trajectories: {len(trajs_norm)}")

    x_train = torch.tensor(trajs_norm, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_train), batch_size=batch_size,
                        shuffle=True, drop_last=False)
    if len(loader) == 0:
        raise ValueError("No training batches were created; lower --seq_len or check the dataset.")

    # Model — larger base_ch for 25D
    model = TemporalUNet(seq_len=seq_len, n_features=n_features, base_ch=base_ch)
    schedule = NoiseSchedule(T=T, device=device)
    ddpm = DDPM(model, schedule, device=device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(loader), epochs=n_epochs,
        pct_start=min(0.3, max(1.0 / n_epochs, 10.0 / n_epochs)),
    )

    # Training loop
    losses = []
    model.train()
    for epoch in range(1, n_epochs + 1):
        epoch_losses = []
        for (x0,) in loader:
            x0 = x0.to(device)
            optimizer.zero_grad()
            loss = ddpm.training_loss(x0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_losses.append(loss.item())

        mean_loss = np.mean(epoch_losses)
        losses.append(mean_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{n_epochs}  loss={mean_loss:.5f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if epoch % save_every == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"diffusion_metadrive_epoch{epoch}.pt")
            ddpm.save_checkpoint(ckpt_path, norm_stats=norm_stats,
                                 extra={"epoch": epoch, "losses": losses,
                                        "n_features": n_features,
                                        "obs_dim": obs_dim, "act_dim": act_dim,
                                        "seq_len": seq_len, "base_ch": base_ch})
            print(f"  Checkpoint saved: {ckpt_path}")

    # Final checkpoint
    final_path = os.path.join(checkpoint_dir, "diffusion_metadrive_final.pt")
    ddpm.save_checkpoint(final_path, norm_stats=norm_stats,
                         extra={"epoch": n_epochs, "losses": losses,
                                "n_features": n_features,
                                "obs_dim": obs_dim, "act_dim": act_dim,
                                "seq_len": seq_len, "base_ch": base_ch})
    print(f"\nFinal checkpoint: {final_path}")

    # Save loss curve
    try:
        import matplotlib.pyplot as plt
        results_dir = os.path.join(_PROJECT_ROOT, "safe_gta", "results")
        os.makedirs(results_dir, exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot(losses)
        plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
        plt.title("MetaDrive Diffusion — Training Loss")
        plt.yscale("log")
        plt.tight_layout()
        loss_path = os.path.join(results_dir, "diffusion_metadrive_loss.png")
        plt.savefig(loss_path, dpi=100)
        plt.close()
        print(f"Loss curve: {loss_path}")
    except ImportError:
        pass

    print(f"Final loss: {losses[-1]:.5f}")
    return ddpm, norm_stats, losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", type=str, default=None,
                        help="Path to MetaDrive HDF5 file (auto-detected if omitted)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--base_ch", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    print("=" * 55)
    print("MetaDrive Diffusion Model Training (25D)")
    print("=" * 55)
    train(
        hdf5_path=args.hdf5,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        T=args.T,
        seq_len=args.seq_len,
        base_ch=args.base_ch,
        device=args.device,
    )
