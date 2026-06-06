"""
Generate Safe-GTA augmented data with Safety Critic guided diffusion repair.

Run:
  python -m safe_gta.generate_safe_gta --n_augmented 5000 --beta 0.5
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from safe_gta.baseline2.generate_augmented import find_hdf5, load_source_trajectories
from safe_gta.diffusion.ddpm import DDPM
from safe_gta.diffusion.noise_schedule import NoiseSchedule
from safe_gta.diffusion.unet1d import TemporalUNet
from safe_gta.safety_critic import load_checkpoint as load_critic_checkpoint


def normalize(trajs, norm_stats, n_features):
    feat_min = np.asarray(norm_stats["feat_min"][:n_features], dtype=np.float32)
    feat_max = np.asarray(norm_stats["feat_max"][:n_features], dtype=np.float32)
    feat_range = np.where(feat_max - feat_min > 1e-6, feat_max - feat_min, 1.0)
    return 2.0 * (trajs - feat_min) / feat_range - 1.0


def denormalize(trajs, norm_stats, n_features):
    feat_min = np.asarray(norm_stats["feat_min"][:n_features], dtype=np.float32)
    feat_max = np.asarray(norm_stats["feat_max"][:n_features], dtype=np.float32)
    feat_range = np.where(feat_max - feat_min > 1e-6, feat_max - feat_min, 1.0)
    return (trajs + 1.0) / 2.0 * feat_range + feat_min


def load_ddpm(checkpoint_path: str, device: str):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    T = payload.get("T", 200)
    n_features = payload.get("n_features", 2)
    obs_dim = payload.get("obs_dim", 2)
    act_dim = payload.get("act_dim", 2)
    base_ch = payload.get("base_ch", 32)
    seq_len = payload.get("seq_len", 50)
    norm_stats = payload.get("norm_stats", {})
    if "feat_min" not in norm_stats:
        raise ValueError("Diffusion checkpoint must contain norm_stats for guided repair.")

    model = TemporalUNet(seq_len=seq_len, n_features=n_features, base_ch=base_ch)
    schedule = NoiseSchedule(T=T, device=device)
    ddpm = DDPM(model, schedule, device=device)
    ddpm.load_checkpoint(checkpoint_path)
    ddpm.model.eval()
    return ddpm, {
        "T": T,
        "n_features": n_features,
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "base_ch": base_ch,
        "seq_len": seq_len,
        "norm_stats": norm_stats,
    }


def predict_costs(critic, trajs_norm: np.ndarray, device: str, batch_size: int = 256):
    costs = []
    critic.eval()
    with torch.no_grad():
        for start in range(0, len(trajs_norm), batch_size):
            batch = torch.tensor(trajs_norm[start:start + batch_size],
                                 dtype=torch.float32, device=device)
            pred = critic(batch).cpu().numpy()
            costs.append(pred)
    return np.concatenate(costs, axis=0)


def generate_safe_gta_dataset(
    diffusion_checkpoint: str = None,
    critic_checkpoint: str = None,
    hdf5_path: str = None,
    output_path: str = None,
    n_augmented: int = 5000,
    start_t: int = 150,
    beta: float = 0.5,
    oversample: float = 2.0,
    batch_size: int = 32,
    guidance_every: int = 1,
    accept_threshold: float = 0.35,
    device: str = None,
):
    data_dir = os.path.join(_PROJECT_ROOT, "safe_gta", "data")
    ckpt_dir = os.path.join(_PROJECT_ROOT, "safe_gta", "checkpoints")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if diffusion_checkpoint is None:
        diffusion_checkpoint = os.path.join(ckpt_dir, "diffusion_metadrive_final.pt")
    if critic_checkpoint is None:
        critic_checkpoint = os.path.join(ckpt_dir, "metadrive_safety_critic.pt")
    if output_path is None:
        output_path = os.path.join(data_dir, "augmented_safe_gta.pkl")
    if hdf5_path is None:
        hdf5_path = find_hdf5(data_dir)
    if hdf5_path is None:
        raise FileNotFoundError("No MetaDrive HDF5 found. Run python download_dataset.py")

    print(f"Diffusion checkpoint: {diffusion_checkpoint}")
    ddpm, meta = load_ddpm(diffusion_checkpoint, device)
    critic, critic_payload = load_critic_checkpoint(critic_checkpoint, device=device)
    if critic_payload["n_features"] != meta["n_features"]:
        raise ValueError(
            f"Critic n_features={critic_payload['n_features']} does not match "
            f"diffusion n_features={meta['n_features']}"
        )

    source_trajs, _, _ = load_source_trajectories(hdf5_path, seq_len=meta["seq_len"])
    source_trajs = source_trajs[:, :, :meta["n_features"]]
    source_norm = normalize(source_trajs, meta["norm_stats"], meta["n_features"]).astype(np.float32)

    n_candidates = max(n_augmented, int(np.ceil(n_augmented * oversample)))
    rng = np.random.default_rng(0)
    repaired_chunks = []
    cost_chunks = []
    n_done = 0

    print(f"Generating {n_candidates} guided candidates "
          f"(target={n_augmented}, beta={beta}, start_t={start_t})...")
    while n_done < n_candidates:
        bs = min(batch_size, n_candidates - n_done)
        idx = rng.integers(0, len(source_norm), size=bs)
        batch = source_norm[idx]
        repaired_norm = ddpm.guided_repair(
            batch,
            critic,
            beta=beta,
            start_t=start_t,
            guidance_every=guidance_every,
        )
        pred_cost = predict_costs(critic, repaired_norm, device=device, batch_size=batch_size)
        repaired_chunks.append(repaired_norm.astype(np.float32))
        cost_chunks.append(pred_cost.astype(np.float32))
        n_done += bs
        if n_done % 500 == 0 or n_done == n_candidates:
            print(f"  {n_done}/{n_candidates}")

    repaired_norm_all = np.concatenate(repaired_chunks, axis=0)
    costs_all = np.concatenate(cost_chunks, axis=0)
    mean_cost = costs_all.mean(axis=1)
    order = np.argsort(mean_cost)
    chosen = order[:n_augmented]

    repaired_norm = repaired_norm_all[chosen]
    pred_costs = costs_all[chosen]
    repaired = denormalize(repaired_norm, meta["norm_stats"], meta["n_features"])

    obs_dim = meta["obs_dim"]
    act_dim = meta["act_dim"]
    aug_obs = repaired[:, :, :obs_dim]
    aug_act = repaired[:, :, obs_dim:obs_dim + act_dim]
    terminals = np.zeros((len(repaired), meta["seq_len"]), dtype=np.float32)
    terminals[:, -1] = 1.0

    dataset = {
        "observations": aug_obs.astype(np.float32),
        "actions": aug_act.astype(np.float32),
        "rewards": np.ones((len(repaired), meta["seq_len"]), dtype=np.float32),
        "costs": pred_costs.astype(np.float32),
        "terminals": terminals,
        "n_augmented": int(n_augmented),
        "n_candidates": int(n_candidates),
        "obs_dim": int(obs_dim),
        "act_dim": int(act_dim),
        "beta": float(beta),
        "start_t": int(start_t),
        "mean_predicted_cost": float(pred_costs.mean()),
        "max_trajectory_mean_cost": float(mean_cost[chosen].max()),
        "pass_rate_under_threshold": float((mean_cost[chosen] <= accept_threshold).mean()),
        "source_diffusion_checkpoint": diffusion_checkpoint,
        "source_critic_checkpoint": critic_checkpoint,
        "source_hdf5": hdf5_path,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)

    print(f"\nSafe-GTA dataset saved: {output_path}")
    print(f"  obs shape: {aug_obs.shape}  act shape: {aug_act.shape}")
    print(f"  mean predicted cost: {dataset['mean_predicted_cost']:.4f}")
    print(f"  max selected trajectory mean cost: {dataset['max_trajectory_mean_cost']:.4f}")
    print(f"  pass rate under {accept_threshold}: "
          f"{dataset['pass_rate_under_threshold'] * 100:.1f}%")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diffusion_checkpoint", type=str, default=None)
    parser.add_argument("--critic_checkpoint", type=str, default=None)
    parser.add_argument("--hdf5", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--n_augmented", type=int, default=5000)
    parser.add_argument("--start_t", type=int, default=150)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--oversample", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--guidance_every", type=int, default=1)
    parser.add_argument("--accept_threshold", type=float, default=0.35)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    generate_safe_gta_dataset(
        diffusion_checkpoint=args.diffusion_checkpoint,
        critic_checkpoint=args.critic_checkpoint,
        hdf5_path=args.hdf5,
        output_path=args.output,
        n_augmented=args.n_augmented,
        start_t=args.start_t,
        beta=args.beta,
        oversample=args.oversample,
        batch_size=args.batch_size,
        guidance_every=args.guidance_every,
        accept_threshold=args.accept_threshold,
        device=args.device,
    )
