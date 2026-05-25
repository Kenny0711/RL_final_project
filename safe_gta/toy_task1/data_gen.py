import numpy as np
import os


def generate_good_trajectories(n_traj: int = 2000, seq_len: int = 50,
                                 seed: int = 42) -> np.ndarray:
    """
    Good trajectories: straight line along x-axis (y ≈ 0).
    Tiny y-noise (std=0.02) prevents the model from collapsing on a perfectly flat dataset.
    Returns: (n_traj, seq_len, 2)
    """
    rng = np.random.default_rng(seed)
    x = np.tile(np.linspace(0, 10, seq_len), (n_traj, 1))
    y = rng.normal(0, 0.02, (n_traj, seq_len))
    return np.stack([x, y], axis=-1).astype(np.float32)


def generate_bad_trajectories(good_trajs: np.ndarray, noise_std: float = 0.5,
                               drift_amplitude: float = 1.0,
                               seed: int = 123) -> np.ndarray:
    """
    Bad trajectories: good + Gaussian noise + sinusoidal drift on the y-axis.
    This mimics real-world imperfect driving (jitter + lane drift).
    Returns: (n_traj, seq_len, 2)
    """
    rng = np.random.default_rng(seed)
    n, seq_len, _ = good_trajs.shape
    noise = rng.normal(0, noise_std, good_trajs.shape).astype(np.float32)
    drift = (np.sin(np.linspace(0, 2 * np.pi, seq_len)) * drift_amplitude).astype(np.float32)
    bad = good_trajs.copy()
    bad[:, :, 1] += noise[:, :, 1] + drift
    bad[:, :, 0] += noise[:, :, 0] * 0.1  # small x-jitter too
    return bad.astype(np.float32)


def normalize_trajectories(trajs: np.ndarray,
                            stats: dict = None) -> tuple[np.ndarray, dict]:
    """
    Normalize trajectories to [-1, 1] per feature dimension.
    DDPM training is sensitive to scale: unnormalized data leads to poor convergence.
    If stats=None, computes stats from the data (use on training set).
    If stats provided, applies existing stats (use on test/eval set).
    Returns: (normalized_trajs, stats_dict)
    """
    if stats is None:
        min_val = trajs.min(axis=(0, 1), keepdims=True)  # (1, 1, 2)
        max_val = trajs.max(axis=(0, 1), keepdims=True)
        stats = {"min": min_val, "max": max_val}
    min_val = stats["min"]
    max_val = stats["max"]
    scale = (max_val - min_val).clip(1e-6)
    normalized = 2 * (trajs - min_val) / scale - 1.0
    return normalized.astype(np.float32), stats


def denormalize_trajectories(trajs: np.ndarray, stats: dict) -> np.ndarray:
    """Invert normalization for visualization and L2 metric computation."""
    min_val = stats["min"]
    max_val = stats["max"]
    scale = (max_val - min_val).clip(1e-6)
    return ((trajs + 1.0) / 2.0 * scale + min_val).astype(np.float32)


def save_dataset(good: np.ndarray, bad: np.ndarray, stats: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, good=good, bad=bad,
             stats_min=stats["min"], stats_max=stats["max"])
    print(f"Dataset saved to {path}  |  good: {good.shape}  bad: {bad.shape}")


def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    data = np.load(path)
    stats = {"min": data["stats_min"], "max": data["stats_max"]}
    return data["good"], data["bad"], stats


if __name__ == "__main__":
    good = generate_good_trajectories(n_traj=2000, seq_len=50)
    bad = generate_bad_trajectories(good)
    good_norm, stats = normalize_trajectories(good)
    bad_norm, _ = normalize_trajectories(bad, stats=stats)
    save_dataset(good_norm, bad_norm, stats,
                 path="safe_gta/data/toy_task1.npz")

    # Quick sanity check with a plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        good_raw = denormalize_trajectories(good_norm[:5], stats)
        bad_raw = denormalize_trajectories(bad_norm[:5], stats)
        for traj in good_raw:
            axes[0].plot(traj[:, 0], traj[:, 1], color="green", alpha=0.7)
        for traj in bad_raw:
            axes[1].plot(traj[:, 0], traj[:, 1], color="red", alpha=0.7)
        axes[0].set_title("Good trajectories (straight)")
        axes[1].set_title("Bad trajectories (noisy + drift)")
        for ax in axes:
            ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_ylim(-3, 3)
        plt.tight_layout()
        plt.savefig("safe_gta/results/data_preview.png", dpi=100)
        print("Preview saved to safe_gta/results/data_preview.png")
    except ImportError:
        pass
