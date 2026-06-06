"""
3D validation pipeline skeleton for Safe-GTA repair.

Goal
----
Run the same fixed MetaDrive map twice:

1. Generate a bad baseline trajectory on-site and record its actions.
2. Repair the recorded bad actions with a Safe-GTA repair hook.
3. Reset the exact same map and replay the repaired actions in 3D.

This file intentionally keeps the repair function as a mock so the pipeline can
be demonstrated before the final Diffusion + Safety Critic integration is
finished.

Examples
--------
3D render demo:
    python evaluate_repair_3d.py --render

Fast dry run without a 3D window:
    python evaluate_repair_3d.py --no-render

Use a bad policy that drives forward first, then fails in a turn:
    python evaluate_repair_3d.py --render --bad-policy late_left --throttle 0.65

Use a visually obvious mock repair that straightens steering:
    python evaluate_repair_3d.py --render --repair-mode straighten
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    from metadrive.engine.asset_loader import AssetLoader
    from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
except Exception as exc:  # pragma: no cover - import environment dependent
    AssetLoader = None
    SafeMetaDriveEnv = None
    METADRIVE_IMPORT_ERROR = exc
else:
    METADRIVE_IMPORT_ERROR = None


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "safe_gta" / "results"


@dataclass
class EpisodeSummary:
    name: str
    steps: int
    total_reward: float
    total_cost: float
    terminated: bool
    truncated: bool
    crashed: bool
    out_of_road: bool
    arrive_dest: bool


def metadrive_assets_ready() -> Tuple[bool, str]:
    """Return whether MetaDrive simulator assets are installed."""
    if AssetLoader is None:
        return False, "MetaDrive is not importable"
    asset_path = str(AssetLoader.asset_path)
    version_path = os.path.join(asset_path, "version.txt")
    return os.path.exists(version_path), asset_path


def reset_env(env: Any, seed: int):
    """Handle MetaDrive versions that expose different reset signatures."""
    try:
        out = env.reset(seed=seed)
    except TypeError:
        try:
            out = env.reset(force_seed=seed)
        except TypeError:
            out = env.reset()

    if isinstance(out, tuple) and len(out) == 2:
        return out
    return out, {}


def step_env(env: Any, action: np.ndarray):
    """Handle gym-style 4-return and gymnasium-style 5-return APIs."""
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
    else:
        obs, reward, done, info = out
        terminated, truncated = bool(done), False
    return obs, float(reward), bool(terminated), bool(truncated), info


def make_env(seed: int, render: bool, traffic_density: float):
    """Create a fixed-map SafeMetaDrive environment."""
    if SafeMetaDriveEnv is None:
        raise RuntimeError(f"Cannot import MetaDrive: {METADRIVE_IMPORT_ERROR}")

    config = {
        "use_render": render,
        "num_scenarios": 1,
        "start_seed": seed,
        "traffic_density": traffic_density,
        "window_size": (1200, 800),
        "horizon": 1000,
    }
    return SafeMetaDriveEnv(config)


def bad_policy(
    step: int,
    policy: str,
    throttle: float,
    steering_scale: float,
    warmup_steps: int,
) -> np.ndarray:
    """Baseline policy used to generate a risky trajectory on the fixed map."""
    if policy == "late_left":
        ramp = min(1.0, max(0.0, (step - warmup_steps) / 45.0))
        steering = steering_scale * ramp
    elif policy == "late_right":
        ramp = min(1.0, max(0.0, (step - warmup_steps) / 45.0))
        steering = -steering_scale * ramp
    elif policy == "hard_left":
        steering = steering_scale
    elif policy == "hard_right":
        steering = -steering_scale
    elif policy == "zigzag":
        steering = steering_scale if (step // 18) % 2 == 0 else -steering_scale
    else:
        steering = steering_scale * np.sin(step / 14.0)
    return np.array([steering, throttle], dtype=np.float32)


def run_safe_gta_repair(
    bad_actions: np.ndarray,
    critic_model: Any = None,
    mode: str = "smooth_steering",
) -> np.ndarray:
    """
    Placeholder hook for the future Safe-GTA repair module.

    Future integration point:
        bad_actions
            -> build trajectory tensor with observations/actions
            -> normalize with diffusion checkpoint stats
            -> DDPM.guided_repair(..., safety_critic=critic_model)
            -> extract repaired actions

    Current mock:
        reduce/smooth/straighten over-aggressive steering while preserving
        throttle.
    """
    print("[Safe-GTA] Receive risky action trajectory.")
    print("[Safe-GTA] Safety Critic gradient step: placeholder.")
    print("[Safe-GTA] Diffusion denoising repair: placeholder.")

    safe_actions = np.asarray(bad_actions, dtype=np.float32).copy()
    if len(safe_actions) == 0:
        return safe_actions

    if mode == "straighten":
        safe_actions[:, 0] = 0.0
    elif mode == "lane_keep":
        decay = np.linspace(1.0, 0.0, len(safe_actions), dtype=np.float32)
        safe_actions[:, 0] = 0.15 * safe_actions[:, 0] * decay
    elif mode == "clip":
        safe_actions[:, 0] = np.clip(safe_actions[:, 0], -0.35, 0.35)
    elif mode == "smooth_steering":
        safe_actions[:, 0] = np.clip(safe_actions[:, 0], -0.75, 0.75)
        smoothed = safe_actions[:, 0].copy()
        for i in range(1, len(smoothed) - 1):
            smoothed[i] = 0.25 * safe_actions[i - 1, 0] + 0.5 * safe_actions[i, 0] + 0.25 * safe_actions[i + 1, 0]
        safe_actions[:, 0] = 0.5 * smoothed
    else:
        safe_actions[:, 0] *= 0.5

    print("[Safe-GTA] Repair finished. Safe action sequence is ready.")
    return safe_actions


def rollout_actions(
    env: Any,
    seed: int,
    name: str,
    actions: np.ndarray | None,
    max_steps: int,
    render: bool,
    policy_name: str,
    throttle: float,
    sleep: float,
    steering_scale: float,
    warmup_steps: int,
) -> Tuple[EpisodeSummary, np.ndarray]:
    """Run either policy-generated actions or a provided action sequence."""
    print(f"\n[{name}] Reset fixed map with seed={seed}.")
    obs, info = reset_env(env, seed)

    recorded_actions: List[np.ndarray] = []
    total_reward = 0.0
    total_cost = 0.0
    last_info: Dict[str, Any] = {}
    terminated = False
    truncated = False

    replay_len = len(actions) if actions is not None else max_steps
    for step in range(replay_len):
        if actions is None:
            action = bad_policy(
                step,
                policy_name,
                throttle,
                steering_scale,
                warmup_steps,
            )
        else:
            action = np.asarray(actions[step], dtype=np.float32)

        obs, reward, terminated, truncated, info = step_env(env, action)
        recorded_actions.append(action)
        total_reward += reward
        total_cost += float(info.get("cost", 0.0))
        last_info = info

        if render:
            env.render()
            if sleep > 0:
                time.sleep(sleep)

        if terminated or truncated:
            break

    summary = EpisodeSummary(
        name=name,
        steps=len(recorded_actions),
        total_reward=float(total_reward),
        total_cost=float(total_cost),
        terminated=terminated,
        truncated=truncated,
        crashed=bool(last_info.get("crash", False)),
        out_of_road=bool(last_info.get("out_of_road", False)),
        arrive_dest=bool(last_info.get("arrive_dest", False)),
    )
    return summary, np.asarray(recorded_actions, dtype=np.float32)


def save_outputs(
    bad_actions: np.ndarray,
    safe_actions: np.ndarray,
    summaries: List[EpisodeSummary],
    output_dir: Path,
) -> None:
    metrics_dir = output_dir / "metrics"
    artifacts_dir = output_dir / "artifacts"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        artifacts_dir / "repair_3d_actions.npz",
        bad_actions=bad_actions,
        safe_actions=safe_actions,
    )
    with open(metrics_dir / "repair_3d_summary.json", "w", encoding="utf-8") as f:
        json.dump([asdict(s) for s in summaries], f, indent=2)
    print(f"\nSaved actions: {artifacts_dir / 'repair_3d_actions.npz'}")
    print(f"Saved summary: {metrics_dir / 'repair_3d_summary.json'}")


def evaluate_and_render_3d(args: argparse.Namespace) -> None:
    print("Start Safe-GTA 3D validation pipeline.")

    ready, asset_path = metadrive_assets_ready()
    if args.render and not ready:
        print("\nMetaDrive simulator assets are not installed yet.")
        print(f"Expected assets path: {asset_path}")
        print("Run MetaDrive once with network access so it can download assets,")
        print("or use --no-render for a fast pipeline dry run.")
        if not args.allow_missing_assets:
            return

    env = make_env(args.seed, args.render, args.traffic_density)
    summaries: List[EpisodeSummary] = []
    try:
        print("\nStage 1: generate and record risky baseline actions.")
        bad_summary, bad_actions = rollout_actions(
            env=env,
            seed=args.seed,
            name="baseline_bad",
            actions=None,
            max_steps=args.steps,
            render=args.render,
            policy_name=args.bad_policy,
            throttle=args.throttle,
            sleep=args.sleep,
            steering_scale=args.steering_scale,
            warmup_steps=args.warmup_steps,
        )
        summaries.append(bad_summary)
        print(f"Baseline summary: {bad_summary}")

        print("\nStage 2: repair the risky action sequence with Safe-GTA hook.")
        safe_actions = run_safe_gta_repair(
            bad_actions,
            critic_model=None,
            mode=args.repair_mode,
        )

        if args.pause_before_replay > 0:
            print(f"\nPause {args.pause_before_replay:.1f}s before repaired replay.")
            time.sleep(args.pause_before_replay)

        print("\nStage 3: replay repaired actions on the same fixed map.")
        safe_summary, replayed_safe_actions = rollout_actions(
            env=env,
            seed=args.seed,
            name="safe_gta_repaired",
            actions=safe_actions,
            max_steps=args.steps,
            render=args.render,
            policy_name=args.bad_policy,
            throttle=args.throttle,
            sleep=args.sleep,
            steering_scale=args.steering_scale,
            warmup_steps=args.warmup_steps,
        )
        summaries.append(safe_summary)
        print(f"Repaired summary: {safe_summary}")

        save_outputs(
            bad_actions=bad_actions,
            safe_actions=replayed_safe_actions,
            summaries=summaries,
            output_dir=Path(args.output_dir),
        )
    finally:
        env.close()
        print("\nFinished 3D validation pipeline.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safe-GTA fixed-map 3D repair validation skeleton.")
    parser.add_argument("--seed", type=int, default=42, help="Fixed MetaDrive map seed.")
    parser.add_argument("--steps", type=int, default=300, help="Maximum rollout steps.")
    parser.add_argument("--traffic-density", type=float, default=0.1)
    parser.add_argument(
        "--bad-policy",
        choices=["sinusoidal", "zigzag", "hard_left", "hard_right", "late_left", "late_right"],
        default="late_left",
    )
    parser.add_argument("--throttle", type=float, default=0.65)
    parser.add_argument("--steering-scale", type=float, default=0.45)
    parser.add_argument("--warmup-steps", type=int, default=60)
    parser.add_argument(
        "--repair-mode",
        choices=["scale", "clip", "smooth_steering", "lane_keep", "straighten"],
        default="straighten",
    )
    parser.add_argument("--output-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--sleep", type=float, default=0.0, help="Delay between rendered frames.")
    parser.add_argument("--pause-before-replay", type=float, default=2.0)
    parser.add_argument("--allow-missing-assets", action="store_true", help="Try running render even if assets check fails.")

    render_group = parser.add_mutually_exclusive_group()
    render_group.add_argument("--render", dest="render", action="store_true", help="Open a 3D MetaDrive window.")
    render_group.add_argument("--no-render", dest="render", action="store_false", help="Run without a 3D window.")
    parser.set_defaults(render=True)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_and_render_3d(parse_args())
