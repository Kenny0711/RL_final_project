"""
Baseline 2 — GTA without safety constraints
Generates an augmented dataset using the trained Toy Task 1 diffusion model,
with cost labels forced to 0.0 (no safety filtering).

This creates Baseline 2: high return but high constraint violations,
demonstrating the importance of the Safety Critic.

Run: python -m safe_gta.baseline2.generate_augmented \
         --checkpoint safe_gta/checkpoints/toy_task1_final.pt \
         --n_augmented 5000 \
         --output safe_gta/data/augmented_no_safety.pkl
"""
import argparse
import os
import sys
import pickle

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from safe_gta.diffusion.noise_schedule import NoiseSchedule
from safe_gta.diffusion.unet1d import TemporalUNet
from safe_gta.diffusion.ddpm import DDPM
from safe_gta.toy_task1.data_gen import (
    normalize_trajectories,
    denormalize_trajectories,
)


# ---------------------------------------------------------------------------
# OSRL dataset loading (with fallback to mock data)
# ---------------------------------------------------------------------------

def load_osrl_trajectories(env: str = "MetaDrive-v0",
                            quality: str = "medium",
                            seq_len: int = 50) -> tuple[np.ndarray, dict]:
    """
    Tries to load real OSRL dataset.
    Falls back to synthetic mock data if osrl-lib is not installed or env not found.
    Returns: (trajectories: (N, seq_len, obs_dim+act_dim), meta: dict)
    """
    try:
        from osrl.common.dataset import SequenceDataset
        print(f"Loading OSRL dataset: {env}-{quality}...")
        ds = SequenceDataset(f"{env}-{quality}-replay-v0", seq_len=seq_len)
        observations = ds.dataset["observations"]
        actions = ds.dataset["actions"]
        trajs = np.concatenate([observations, actions], axis=-1)
        print(f"  Loaded {len(trajs)} transitions, obs+act dim={trajs.shape[-1]}")
        meta = {
            "obs_dim": observations.shape[-1],
            "act_dim": actions.shape[-1],
            "source": "osrl",
        }
        return trajs, meta
    except Exception as e:
        print(f"OSRL not available ({e}), using mock MetaDrive-like data.")
        return _make_mock_metadrive_trajs(n=3000, seq_len=seq_len)


def _make_mock_metadrive_trajs(n: int = 3000, seq_len: int = 50,
                                obs_dim: int = 23, act_dim: int = 2) -> tuple[np.ndarray, dict]:
    """
    Mock MetaDrive-like dataset with obs_dim=23, act_dim=2.
    Contains a mix of 'good' (low cost) and 'bad' (high cost) trajectories.
    """
    rng = np.random.default_rng(0)
    n_good = int(n * 0.4)
    n_bad = n - n_good

    # good: smoother, centered
    good_obs = rng.normal(0, 0.3, (n_good, seq_len, obs_dim)).astype(np.float32)
    good_act = rng.normal(0, 0.1, (n_good, seq_len, act_dim)).astype(np.float32)

    # bad: noisier, drifting
    bad_obs = rng.normal(0, 1.0, (n_bad, seq_len, obs_dim)).astype(np.float32)
    bad_act = rng.normal(0, 0.5, (n_bad, seq_len, act_dim)).astype(np.float32)

    obs = np.concatenate([good_obs, bad_obs], axis=0)
    act = np.concatenate([good_act, bad_act], axis=0)
    trajs = np.concatenate([obs, act], axis=-1)
    meta = {"obs_dim": obs_dim, "act_dim": act_dim, "source": "mock"}
    return trajs, meta


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------

def generate_augmented_dataset(
    ddpm_checkpoint: str,
    n_augmented: int = 5000,
    mode: str = "repair",   # "repair": SDEdit on existing trajs | "generate": from noise
    start_t: int = 400,
    output_path: str = "safe_gta/data/augmented_no_safety.pkl",
    seq_len: int = 50,
    batch_size: int = 64,
    device: str = None,
):
    """
    Generates augmented trajectories using the trained diffusion model.
    cost labels are FORCED TO 0.0 (Baseline 2: no safety filtering).

    mode='repair':   Applies SDEdit to existing OSRL trajectories.
                     Preserves coarse structure, removes noise. Mirrors the
                     final Safe-GTA pipeline but WITHOUT safety guidance.
    mode='generate': Samples unconditionally from the learned distribution.
                     More diverse but less grounded in real data.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    payload = torch.load(ddpm_checkpoint, map_location=device, weights_only=False)
    T = payload.get("T", 200)
    norm_stats = payload.get("norm_stats", None)

    model = TemporalUNet(seq_len=seq_len, n_features=2)
    schedule = NoiseSchedule(T=T, device=device)
    ddpm = DDPM(model, schedule, device=device)
    ddpm.load_checkpoint(ddpm_checkpoint)
    print(f"Loaded checkpoint from {ddpm_checkpoint}")

    augmented_trajs = []

    if mode == "repair":
        source_trajs, meta = load_osrl_trajectories(seq_len=seq_len)
        print(f"Generating {n_augmented} repaired trajectories (SDEdit, start_t={start_t})...")

        # Use only obs+act first 2 dims for 2D diffusion (toy model is 2D)
        # In production this would use the full obs_dim model
        obs_act = source_trajs[:, :, :2]
        obs_act_norm, repair_stats = normalize_trajectories(obs_act)

        # Sample source trajectories with replacement
        rng = np.random.default_rng(0)
        n_batches = (n_augmented + batch_size - 1) // batch_size
        for i in range(n_batches):
            remaining = min(batch_size, n_augmented - len(augmented_trajs))
            idx = rng.integers(0, len(obs_act_norm), remaining)
            batch = obs_act_norm[idx]
            repaired = ddpm.repair(batch, start_t=start_t)
            repaired_raw = denormalize_trajectories(np.array(repaired), repair_stats)
            augmented_trajs.append(repaired_raw)
            if (i + 1) % 10 == 0:
                print(f"  {len(augmented_trajs) * batch_size}/{n_augmented}")

    elif mode == "generate":
        print(f"Generating {n_augmented} trajectories unconditionally...")
        n_batches = (n_augmented + batch_size - 1) // batch_size
        for i in range(n_batches):
            remaining = min(batch_size, n_augmented - sum(len(t) for t in augmented_trajs))
            generated = ddpm.generate(remaining, seq_len=seq_len)
            if norm_stats:
                generated = denormalize_trajectories(generated, norm_stats)
            augmented_trajs.append(generated)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    augmented = np.concatenate(augmented_trajs, axis=0)[:n_augmented]

    # Build dataset dict — cost forced to 0.0 intentionally
    dataset = {
        "observations": augmented[:, :, :2].astype(np.float32),
        "actions": np.zeros((len(augmented), seq_len, 2), dtype=np.float32),
        "rewards": np.ones((len(augmented), seq_len), dtype=np.float32),
        "costs": np.zeros((len(augmented), seq_len), dtype=np.float32),   # FORCED 0.0
        "terminals": np.zeros((len(augmented), seq_len), dtype=np.float32),
        "mode": mode,
        "n_augmented": n_augmented,
        "source_checkpoint": ddpm_checkpoint,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"\nAugmented dataset saved: {output_path}")
    print(f"  Trajectories: {len(augmented)}  shape: {augmented.shape}")
    print("  Cost labels: 0.0 (no safety filtering — Baseline 2 intentional)")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="safe_gta/toy_task1/safe_gta/checkpoints/toy_task1_final.pt")
    parser.add_argument("--n_augmented", type=int, default=5000)
    parser.add_argument("--mode", type=str, default="repair",
                        choices=["repair", "generate"])
    parser.add_argument("--start_t", type=int, default=400)
    parser.add_argument("--output", type=str,
                        default="safe_gta/data/augmented_no_safety.pkl")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    generate_augmented_dataset(
        ddpm_checkpoint=args.checkpoint,
        n_augmented=args.n_augmented,
        mode=args.mode,
        start_t=args.start_t,
        output_path=args.output,
        device=args.device,
    )
