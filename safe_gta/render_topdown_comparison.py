"""
Render a true MetaDrive top-down comparison on a fixed map.

This uses MetaDrive's native top-down renderer, not a hand-drawn proxy.
It runs the same fixed map twice:

1. baseline bad policy
2. repaired actions from the Safe-GTA mock repair hook

Outputs:
  safe_gta/results/final_figures/topdown_baseline.png
  safe_gta/results/final_figures/topdown_repaired.png
  safe_gta/results/final_figures/topdown_comparison.png
  safe_gta/results/metrics/topdown_comparison_summary.json
  safe_gta/results/artifacts/topdown_comparison_actions.npz

Run:
  python -m safe_gta.render_topdown_comparison

For a stronger failure:
  python -m safe_gta.render_topdown_comparison --steering-scale 0.6 --warmup-steps 35

The default repaired replay is slowed down and extended so the top-down image
shows a longer safe continuation after the repaired action segment.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from evaluate_repair_3d import (  # noqa: E402
    bad_policy,
    reset_env,
    run_safe_gta_repair,
    step_env,
)

RESULTS_DIR = _PROJECT_ROOT / "safe_gta" / "results"
FINAL_FIGURES_DIR = RESULTS_DIR / "final_figures"
METRICS_DIR = RESULTS_DIR / "metrics"
ARTIFACTS_DIR = RESULTS_DIR / "artifacts"


def make_env(seed: int, traffic_density: float):
    from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv

    config = {
        "use_render": False,
        "num_scenarios": 1,
        "start_seed": seed,
        "traffic_density": traffic_density,
        "window_size": (1000, 1000),
        "horizon": 1000,
    }
    return SafeMetaDriveEnv(config)


def render_topdown(env: Any):
    """Render a MetaDrive top-down frame, supporting naming differences."""
    render_kwargs = {
        "semantic_map": True,
        "film_size": (1200, 1200),
        "screen_size": (1200, 1200),
    }
    try:
        return env.render(mode="top_down", track_target_vehicle=True, **render_kwargs)
    except TypeError:
        try:
            return env.render(mode="topdown", track_target_vehicle=True, **render_kwargs)
        except TypeError:
            return env.render(mode="topdown", **render_kwargs)


def frame_to_array(frame: Any) -> np.ndarray:
    try:
        import pygame

        if isinstance(frame, pygame.Surface):
            arr = pygame.surfarray.array3d(frame)
            return np.transpose(arr, (1, 0, 2))
    except Exception:
        pass

    arr = np.asarray(frame)
    if arr.ndim == 3:
        return arr.astype(np.uint8)
    raise RuntimeError(f"Unsupported top-down frame type: {type(frame)}")


def save_image(frame: Any, path: Path) -> None:
    from PIL import Image

    arr = frame_to_array(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def slow_and_extend_actions(
    actions: np.ndarray,
    throttle_scale: float,
    extension_steps: int,
    extension_throttle: float,
    extension_steering: float,
) -> np.ndarray:
    """Slow the repaired replay and append a gentle continuation segment."""
    slowed = np.asarray(actions, dtype=np.float32).copy()
    if len(slowed) == 0:
        return slowed

    slowed[:, 1] = np.clip(slowed[:, 1] * throttle_scale, -1.0, 1.0)
    if extension_steps <= 0:
        return slowed

    extension = np.zeros((extension_steps, slowed.shape[1]), dtype=np.float32)
    extension[:, 0] = extension_steering
    extension[:, 1] = extension_throttle
    return np.concatenate([slowed, extension], axis=0)


def rollout(
    env: Any,
    seed: int,
    name: str,
    actions: np.ndarray | None,
    steps: int,
    bad_policy_name: str,
    throttle: float,
    steering_scale: float,
    warmup_steps: int,
    render_every: int,
) -> Tuple[Dict[str, Any], np.ndarray, List[np.ndarray]]:
    reset_env(env, seed)

    recorded_actions: List[np.ndarray] = []
    frames: List[np.ndarray] = []
    total_reward = 0.0
    total_cost = 0.0
    last_info: Dict[str, Any] = {}
    terminated = False
    truncated = False

    rollout_len = len(actions) if actions is not None else steps
    for step in range(rollout_len):
        if actions is None:
            action = bad_policy(
                step,
                bad_policy_name,
                throttle,
                steering_scale,
                warmup_steps,
            )
        else:
            action = np.asarray(actions[step], dtype=np.float32)

        _, reward, terminated, truncated, info = step_env(env, action)
        recorded_actions.append(action)
        total_reward += reward
        total_cost += float(info.get("cost", 0.0))
        last_info = info

        if step % render_every == 0 or terminated or truncated or step == rollout_len - 1:
            frames.append(frame_to_array(render_topdown(env)))

        if terminated or truncated:
            break

    summary = {
        "name": name,
        "steps": len(recorded_actions),
        "total_reward": float(total_reward),
        "total_cost": float(total_cost),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "crash": bool(last_info.get("crash", False)),
        "out_of_road": bool(last_info.get("out_of_road", False)),
        "arrive_dest": bool(last_info.get("arrive_dest", False)),
    }
    return summary, np.asarray(recorded_actions, dtype=np.float32), frames


def save_side_by_side(baseline_frame: np.ndarray, repaired_frame: np.ndarray, output: Path,
                      baseline_summary: Dict[str, Any], repaired_summary: Dict[str, Any]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("MetaDrive Native Top-Down Replay: Baseline vs Repaired", fontsize=16, fontweight="bold")

    panels = [
        ("Baseline risky replay", baseline_frame, baseline_summary),
        ("Safe-GTA repaired replay", repaired_frame, repaired_summary),
    ]
    for ax, (title, frame, summary) in zip(axes, panels):
        ax.imshow(frame)
        ax.set_title(
            f"{title}\nsteps={summary['steps']} cost={summary['total_cost']:.2f} "
            f"out_of_road={summary['out_of_road']}",
            fontsize=11,
        )
        ax.axis("off")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--traffic-density", type=float, default=0.0)
    parser.add_argument("--bad-policy", choices=["late_left", "late_right", "sinusoidal", "zigzag", "hard_left", "hard_right"], default="late_left")
    parser.add_argument("--throttle", type=float, default=0.65)
    parser.add_argument("--steering-scale", type=float, default=0.45)
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--repair-mode", choices=["scale", "clip", "smooth_steering", "lane_keep", "straighten"], default="straighten")
    parser.add_argument("--repaired-throttle-scale", type=float, default=0.45)
    parser.add_argument("--extend-repaired-steps", type=int, default=70)
    parser.add_argument("--extend-throttle", type=float, default=0.10)
    parser.add_argument("--extend-steering", type=float, default=0.0)
    parser.add_argument("--render-every", type=int, default=2)
    args = parser.parse_args()

    print("MetaDrive native top-down comparison")
    print(f"Fixed seed: {args.seed}")

    env = make_env(args.seed, args.traffic_density)
    try:
        baseline_summary, bad_actions, baseline_frames = rollout(
            env=env,
            seed=args.seed,
            name="baseline",
            actions=None,
            steps=args.steps,
            bad_policy_name=args.bad_policy,
            throttle=args.throttle,
            steering_scale=args.steering_scale,
            warmup_steps=args.warmup_steps,
            render_every=args.render_every,
        )
        print(f"Baseline: {baseline_summary}")

        repaired_actions = run_safe_gta_repair(bad_actions, critic_model=None, mode=args.repair_mode)
        repaired_actions = slow_and_extend_actions(
            repaired_actions,
            throttle_scale=args.repaired_throttle_scale,
            extension_steps=args.extend_repaired_steps,
            extension_throttle=args.extend_throttle,
            extension_steering=args.extend_steering,
        )
        repaired_summary, safe_actions, repaired_frames = rollout(
            env=env,
            seed=args.seed,
            name="repaired",
            actions=repaired_actions,
            steps=args.steps,
            bad_policy_name=args.bad_policy,
            throttle=args.throttle,
            steering_scale=args.steering_scale,
            warmup_steps=args.warmup_steps,
            render_every=args.render_every,
        )
        print(f"Repaired: {repaired_summary}")
    finally:
        env.close()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    baseline_path = FINAL_FIGURES_DIR / "topdown_baseline.png"
    repaired_path = FINAL_FIGURES_DIR / "topdown_repaired.png"
    comparison_path = FINAL_FIGURES_DIR / "topdown_comparison.png"
    summary_path = METRICS_DIR / "topdown_comparison_summary.json"
    actions_path = ARTIFACTS_DIR / "topdown_comparison_actions.npz"

    save_image(baseline_frames[-1], baseline_path)
    save_image(repaired_frames[-1], repaired_path)
    save_side_by_side(baseline_frames[-1], repaired_frames[-1], comparison_path, baseline_summary, repaired_summary)
    actions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(actions_path, bad_actions=bad_actions, safe_actions=safe_actions)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "baseline": baseline_summary,
                "repaired": repaired_summary,
                "note": "Native MetaDrive top-down render. Repair hook is currently mock steering repair.",
            },
            f,
            indent=2,
        )

    print(f"Saved baseline top-down: {baseline_path}")
    print(f"Saved repaired top-down: {repaired_path}")
    print(f"Saved comparison: {comparison_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
