"""
Generate a 2D poster figure for Safe-GTA trajectory repair.

This script does not modify the 3D validation pipeline. It only reads action
sequences and creates a high-resolution 2D comparison figure.

Preferred input:
    safe_gta/results/repair_3d_actions.npz

Fallback inputs:
    data/processed/metadrive_bad.npz
    data/processed/repaired_actions.npy

Run:
    python generate_final_poster.py

If you want a demo figure without any saved actions:
    python generate_final_poster.py --allow-mock

Extend the repaired route for a longer poster line:
    python generate_final_poster.py --extend-good 100
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ACTIONS_NPZ = PROJECT_ROOT / "safe_gta" / "results" / "repair_3d_actions.npz"
DEFAULT_BAD_NPZ = PROJECT_ROOT / "data" / "processed" / "metadrive_bad.npz"
DEFAULT_REPAIRED_NPY = PROJECT_ROOT / "data" / "processed" / "repaired_actions.npy"
DEFAULT_OUTPUT = PROJECT_ROOT / "safe_gta" / "results" / "safe_gta_final_poster.png"


def apply_safety_guidance(states, actions, critic_model, guidance_scale=0.05):
    """
    Teammate integration hook for the real Diffusion denoising loop.

    Expected future usage:
        states:  torch.Tensor, shape (batch, seq_len, 259)
        actions: torch.Tensor, shape (batch, seq_len, 2)

    This helper assumes critic_model(states, actions) returns step-wise danger
    scores. If your critic accepts concatenated (obs, action), adapt this
    wrapper before using it in the real model.
    """
    import torch

    actions_with_grad = actions.detach().clone()
    actions_with_grad.requires_grad_(True)

    danger_scores = critic_model(states, actions_with_grad)
    total_danger = danger_scores.sum()

    grad = torch.autograd.grad(
        outputs=total_danger,
        inputs=actions_with_grad,
        retain_graph=False,
        create_graph=False,
    )[0]

    repaired_actions = actions - guidance_scale * grad
    return repaired_actions.detach()


def smooth_actions(actions: np.ndarray, steering_scale: float = 0.2) -> np.ndarray:
    """Simple poster fallback when real repaired actions are not available."""
    repaired = np.asarray(actions, dtype=np.float32).copy()
    if len(repaired) == 0:
        return repaired

    steering = repaired[:, 0]
    smoothed = steering.copy()
    for i in range(1, len(steering) - 1):
        smoothed[i] = 0.25 * steering[i - 1] + 0.5 * steering[i] + 0.25 * steering[i + 1]
    repaired[:, 0] = steering_scale * smoothed
    return repaired


def load_from_repair_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    if "bad_actions" not in payload or "safe_actions" not in payload:
        raise ValueError(f"{path} must contain bad_actions and safe_actions.")
    return np.asarray(payload["bad_actions"], dtype=np.float32), np.asarray(payload["safe_actions"], dtype=np.float32)


def load_from_processed_data(bad_path: Path, repaired_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    bad_data = np.load(bad_path, allow_pickle=True)["data"]
    target_traj = None

    for traj in bad_data:
        item = traj.item() if hasattr(traj, "item") else traj
        if not isinstance(item, dict) or "actions" not in item:
            continue
        actions = np.asarray(item["actions"], dtype=np.float32)
        if actions.ndim == 2 and actions.shape[1] >= 2 and np.sum(np.abs(actions[:, 0]) > 0.1) > 20:
            target_traj = item
            break

    if target_traj is None:
        raise ValueError(f"No suitable turning trajectory found in {bad_path}.")

    bad_actions = np.asarray(target_traj["actions"], dtype=np.float32)[:, :2]
    repaired_actions = np.asarray(np.load(repaired_path), dtype=np.float32)[:, :2]
    return bad_actions, repaired_actions


def mock_actions(n_steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    steps = np.arange(n_steps, dtype=np.float32)
    bad_steering = 0.55 * np.sin(steps / 10.0) + 0.25 * (steps > 45)
    bad_throttle = np.full(n_steps, 0.65, dtype=np.float32)
    bad = np.stack([bad_steering, bad_throttle], axis=1)
    good = smooth_actions(bad, steering_scale=0.15)
    return bad, good


def load_actions(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, str]:
    actions_npz = Path(args.actions_npz)
    bad_npz = Path(args.bad_npz)
    repaired_npy = Path(args.repaired_npy)

    if actions_npz.exists():
        bad, good = load_from_repair_npz(actions_npz)
        return bad, good, str(actions_npz)

    if bad_npz.exists() and repaired_npy.exists():
        bad, good = load_from_processed_data(bad_npz, repaired_npy)
        return bad, good, f"{bad_npz} + {repaired_npy}"

    if args.allow_mock:
        bad, good = mock_actions()
        return bad, good, "mock actions"

    raise FileNotFoundError(
        "No action data found. Run evaluate_repair_3d.py first, or provide "
        "--allow-mock for a demo figure."
    )


def calculate_trajectory(
    actions: np.ndarray,
    dt: float = 0.1,
    initial_speed: float = 5.0,
    accel_gain: float = 3.0,
    steer_gain: float = 0.5,
    max_speed: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert action sequence to approximate 2D kinematic trajectory."""
    x, y, heading, speed = 0.0, 0.0, 0.0, initial_speed
    xs, ys, headings, speeds = [x], [y], [heading], [speed]

    for steering, throttle in np.asarray(actions, dtype=np.float32):
        speed += float(throttle) * accel_gain * dt
        speed = float(np.clip(speed, 0.0, max_speed))
        heading += float(steering) * steer_gain * dt * max(speed / max(initial_speed, 1e-6), 0.2)
        x += speed * np.cos(heading) * dt
        y += speed * np.sin(heading) * dt
        xs.append(x)
        ys.append(y)
        headings.append(heading)
        speeds.append(speed)

    return np.asarray(xs), np.asarray(ys), np.asarray(headings), np.asarray(speeds)


def smooth_curve(values: np.ndarray, window: int = 21) -> np.ndarray:
    if len(values) < 3:
        return values
    window = min(window, len(values))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def build_track_from_repaired(xs_good: np.ndarray, ys_good: np.ndarray) -> Dict[str, np.ndarray]:
    """Use repaired trajectory as the poster track centerline.

    This avoids the misleading case where the repaired route is straight but
    the decorative track bends away from it.
    """
    x = np.asarray(xs_good, dtype=np.float32)
    y = smooth_curve(np.asarray(ys_good, dtype=np.float32), window=25)

    # Extend the visual track slightly beyond the repaired endpoint.
    if len(x) >= 2:
        dx = x[-1] - x[-2]
        dy = y[-1] - y[-2]
        extra = np.arange(1, 31, dtype=np.float32)
        x = np.concatenate([x, x[-1] + extra * dx])
        y = np.concatenate([y, y[-1] + extra * dy])

    return {"x": x, "y": y}


def extend_repaired_actions(actions: np.ndarray, extra_steps: int, mode: str) -> np.ndarray:
    """Extend repaired actions so the poster route continues after repair."""
    actions = np.asarray(actions, dtype=np.float32)
    if extra_steps <= 0 or len(actions) == 0:
        return actions

    tail = np.zeros((extra_steps, 2), dtype=np.float32)
    if mode == "straight":
        tail[:, 0] = 0.0
        tail[:, 1] = actions[-1, 1]
    elif mode == "coast":
        tail[:, 0] = 0.0
        tail[:, 1] = 0.0
    else:
        tail[:, 0] = actions[-1, 0]
        tail[:, 1] = actions[-1, 1]
    return np.concatenate([actions, tail], axis=0)


def plot_poster(
    bad_actions: np.ndarray,
    good_actions: np.ndarray,
    source: str,
    output: Path,
    extend_good: int,
    extend_mode: str,
) -> None:
    bad_actions = np.asarray(bad_actions, dtype=np.float32)
    good_actions = extend_repaired_actions(good_actions, extend_good, extend_mode)

    xs_bad, ys_bad, heading_bad, speed_bad = calculate_trajectory(bad_actions)
    xs_good, ys_good, heading_good, speed_good = calculate_trajectory(good_actions)
    track = build_track_from_repaired(xs_good, ys_good)

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.7, 1.0], hspace=0.28, wspace=0.18)
    ax = fig.add_subplot(gs[0, :])
    ax_steer = fig.add_subplot(gs[1, 0])
    ax_speed = fig.add_subplot(gs[1, 1])

    fig.suptitle("Safe-GTA: Zero-shot Trajectory Repair", fontsize=28, fontweight="bold", y=0.97)

    ax.plot(track["x"], track["y"], color="#D3D3D3", linewidth=80, alpha=0.6, solid_capstyle="round", zorder=1)
    ax.plot(track["x"], track["y"], color="white", linewidth=3, linestyle="--", zorder=2, label="Track centerline")

    ax.plot(xs_good, ys_good, color="#1F77B4", linewidth=6, zorder=4, label="Safe-GTA Repaired")
    ax.scatter(xs_good[-1], ys_good[-1], color="#1F77B4", marker="o", edgecolor="white", linewidth=1.5, s=190, zorder=7)

    # Draw the baseline above the repaired path so it remains visible even when
    # both routes overlap early in the episode.
    ax.plot(xs_bad, ys_bad, color="#FF4B4B", linewidth=5, linestyle="--", zorder=8, label="Original Baseline")
    ax.scatter(xs_bad[-1], ys_bad[-1], color="black", marker="X", s=360, zorder=9, label="Baseline endpoint")
    ax.scatter(0, 0, color="#2CA02C", marker="s", s=230, zorder=7, label="Start")

    ax.annotate(
        "Risky baseline route",
        xy=(xs_bad[-1], ys_bad[-1]),
        xytext=(xs_bad[-1] - 45, ys_bad[-1] + 35),
        arrowprops={"arrowstyle": "->", "linewidth": 2, "color": "#333333"},
        fontsize=15,
        color="#333333",
    )
    ax.annotate(
        "Repaired action sequence",
        xy=(xs_good[-1], ys_good[-1]),
        xytext=(xs_good[-1] - 70, ys_good[-1] - 45),
        arrowprops={"arrowstyle": "->", "linewidth": 2, "color": "#1F77B4"},
        fontsize=15,
        color="#1F77B4",
    )

    ax.set_xlabel("X Position (meters)", fontsize=17)
    ax.set_ylabel("Y Position (meters)", fontsize=17)
    ax.tick_params(labelsize=13)
    ax.legend(loc="lower left", fontsize=14, framealpha=0.94, edgecolor="black")
    ax.grid(True, linestyle=":", alpha=0.55)
    ax.axis("equal")

    all_x = np.concatenate([xs_bad, xs_good, track["x"]])
    all_y = np.concatenate([ys_bad, ys_good, track["y"]])
    ax.set_xlim(float(all_x.min()) - 15, float(all_x.max()) + 15)
    ax.set_ylim(float(all_y.min()) - 35, float(all_y.max()) + 25)

    # Add a zoomed inset around the failure/repair point so the original short
    # baseline does not disappear after the repaired route is extended.
    axins = inset_axes(ax, width="34%", height="34%", loc="upper right", borderpad=1.0)
    axins.plot(track["x"], track["y"], color="#D3D3D3", linewidth=34, alpha=0.45, solid_capstyle="round", zorder=1)
    axins.plot(track["x"], track["y"], color="white", linewidth=1.6, linestyle="--", zorder=2)
    axins.plot(xs_good, ys_good, color="#1F77B4", linewidth=3.0, zorder=3)
    axins.plot(xs_bad, ys_bad, color="#FF4B4B", linewidth=3.2, linestyle="--", zorder=4)
    axins.scatter(xs_bad[-1], ys_bad[-1], color="black", marker="X", s=120, zorder=5)
    axins.scatter(xs_good[min(len(xs_good) - 1, len(xs_bad) - 1)],
                  ys_good[min(len(ys_good) - 1, len(ys_bad) - 1)],
                  color="#1F77B4", marker="o", edgecolor="white", linewidth=0.8, s=80, zorder=5)
    fail_x = float(xs_bad[-1])
    fail_y = float(ys_bad[-1])
    axins.set_xlim(fail_x - 35, fail_x + 35)
    axins.set_ylim(fail_y - 25, fail_y + 25)
    axins.set_title("Failure zone zoom", fontsize=10)
    axins.grid(True, linestyle=":", alpha=0.35)
    axins.tick_params(labelsize=8)
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="#555555", alpha=0.65)

    bad_steps = np.arange(len(bad_actions))
    good_steps = np.arange(len(good_actions))
    ax_steer.plot(bad_steps, bad_actions[:, 0], color="#FF4B4B", linewidth=2.4, linestyle="--", label="Original steering")
    ax_steer.plot(good_steps, good_actions[:, 0], color="#1F77B4", linewidth=2.6, label="Repaired steering")
    ax_steer.axhline(0.0, color="#333333", linewidth=1, alpha=0.7)
    ax_steer.set_title("Steering Action", fontsize=15, fontweight="bold")
    ax_steer.set_xlabel("Step", fontsize=13)
    ax_steer.set_ylabel("Steering", fontsize=13)
    ax_steer.grid(True, alpha=0.3)
    ax_steer.legend(fontsize=11)

    ax_speed.plot(np.arange(len(speed_bad)), speed_bad, color="#FF4B4B", linewidth=2.4, linestyle="--", label="Original speed")
    ax_speed.plot(np.arange(len(speed_good)), speed_good, color="#1F77B4", linewidth=2.6, label="Repaired speed")
    ax_speed.set_title("Approximate Kinematic Speed", fontsize=15, fontweight="bold")
    ax_speed.set_xlabel("Step", fontsize=13)
    ax_speed.set_ylabel("Speed", fontsize=13)
    ax_speed.grid(True, alpha=0.3)
    ax_speed.legend(fontsize=11)

    extension_note = f" Repaired route extended by {extend_good} steps ({extend_mode})." if extend_good > 0 else ""
    fig.text(
        0.5,
        0.015,
        f"Source: {source}. 2D kinematic visualization for poster use; not a MetaDrive world-coordinate replay.{extension_note}",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved poster figure: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Safe-GTA final poster 2D repair graphic.")
    parser.add_argument("--actions-npz", type=str, default=str(DEFAULT_ACTIONS_NPZ))
    parser.add_argument("--bad-npz", type=str, default=str(DEFAULT_BAD_NPZ))
    parser.add_argument("--repaired-npy", type=str, default=str(DEFAULT_REPAIRED_NPY))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--allow-mock", action="store_true")
    parser.add_argument("--extend-good", type=int, default=100, help="Extra steps appended to repaired actions for poster continuity.")
    parser.add_argument("--extend-mode", choices=["straight", "coast", "hold"], default="straight")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Generating Safe-GTA final poster graphic...")
    bad_actions, good_actions, source = load_actions(args)
    print(f"Loaded actions from: {source}")
    print(f"Original actions shape: {bad_actions.shape}")
    print(f"Repaired actions shape: {good_actions.shape}")
    plot_poster(
        bad_actions,
        good_actions,
        source,
        Path(args.output),
        extend_good=args.extend_good,
        extend_mode=args.extend_mode,
    )


if __name__ == "__main__":
    main()
