"""
Toy Task 1 — Training Script
Run: python -m safe_gta.toy_task1.train
"""
import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)

from safe_gta.diffusion.noise_schedule import NoiseSchedule
from safe_gta.diffusion.unet1d import TemporalUNet
from safe_gta.diffusion.ddpm import DDPM
from safe_gta.toy_task1.data_gen import (
    generate_good_trajectories,
    generate_bad_trajectories,
    normalize_trajectories,
    save_dataset,
)


def train(
    n_epochs: int = 200,
    batch_size: int = 64,
    lr: float = 2e-4,
    T: int = 200,
    n_good_trajs: int = 2000,
    seq_len: int = 50,
    checkpoint_dir: str = None,
    data_path: str = None,
    save_every: int = 50,
    device: str = None,
):
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(_PROJECT_ROOT, "safe_gta", "checkpoints")
    if data_path is None:
        data_path = os.path.join(_PROJECT_ROOT, "safe_gta", "data", "toy_task1.npz")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}  |  T={T}  epochs={n_epochs}  batch={batch_size}")

    # --- Data ---
    good = generate_good_trajectories(n_traj=n_good_trajs, seq_len=seq_len)
    bad = generate_bad_trajectories(good)
    good_norm, stats = normalize_trajectories(good)
    bad_norm, _ = normalize_trajectories(bad, stats=stats)
    save_dataset(good_norm, bad_norm, stats, data_path)

    x_train = torch.tensor(good_norm, dtype=torch.float32)
    loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True)

    # --- Model ---
    model = TemporalUNet(seq_len=seq_len, n_features=2)
    schedule = NoiseSchedule(T=T, device=device)
    ddpm = DDPM(model, schedule, device=device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # cosine LR warmup over first 10 epochs, then decay
    warmup_epochs = 10
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(loader), epochs=n_epochs,
        pct_start=warmup_epochs / n_epochs,
    )

    # --- Training loop ---
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
            ckpt_path = os.path.join(checkpoint_dir, f"toy_task1_epoch{epoch}.pt")
            ddpm.save_checkpoint(ckpt_path, norm_stats=stats,
                                  extra={"epoch": epoch, "losses": losses})
            print(f"  Checkpoint saved: {ckpt_path}")

    # --- Final checkpoint ---
    final_path = os.path.join(checkpoint_dir, "toy_task1_final.pt")
    ddpm.save_checkpoint(final_path, norm_stats=stats,
                          extra={"epoch": n_epochs, "losses": losses})
    print(f"\nFinal checkpoint saved: {final_path}")

    # --- Save loss curve ---
    try:
        import matplotlib.pyplot as plt
        results_dir = os.path.join(_PROJECT_ROOT, "safe_gta", "results")
        os.makedirs(results_dir, exist_ok=True)
        plt.figure(figsize=(8, 4))
        plt.plot(losses)
        plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
        plt.title("Toy Task 1 — Training Loss")
        plt.yscale("log")
        plt.tight_layout()
        loss_path = os.path.join(results_dir, "training_loss.png")
        plt.savefig(loss_path, dpi=100)
        print(f"Loss curve saved: {loss_path}")
    except ImportError:
        pass

    print(f"\nFinal loss: {losses[-1]:.5f}")
    if losses[-1] < 0.05:
        print("SUCCESS: Final loss < 0.05 ✓")
    else:
        print("WARNING: Final loss > 0.05 — consider more epochs or lower lr")

    return ddpm, stats, losses


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--T", type=int, default=200)
    parser.add_argument("--n_trajs", type=int, default=2000)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    print("=" * 50)
    print("Toy Task 1 — DDPM Training")
    print("=" * 50)
    print("⚠️  Starting training loop. This may take several minutes.")
    train(
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        T=args.T,
        n_good_trajs=args.n_trajs,
        device=args.device,
    )
