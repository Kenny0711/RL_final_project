"""
Render a real MetaDrive top-down simulator demo.

This is different from the offline trajectory proxy plots. It launches a
SafeMetaDrive simulator episode, steps the ego car with a simple fixed action,
and saves the final top-down frame.

Run:
  python -m safe_gta.render_metadrive_topdown

If MetaDrive assets are not installed yet:
  python -m safe_gta.render_metadrive_topdown --allow_download
"""
import argparse
import os
import sys

import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

RESULTS_DIR = os.path.join(_PROJECT_ROOT, "safe_gta", "results")


def metadrive_assets_ready():
    from metadrive.engine.asset_loader import AssetLoader

    version_path = os.path.join(str(AssetLoader.asset_path), "version.txt")
    return os.path.exists(version_path), str(AssetLoader.asset_path)


def save_frame(frame, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        import pygame

        if isinstance(frame, pygame.Surface):
            pygame.image.save(frame, out_path)
            return
    except Exception:
        pass

    try:
        from PIL import Image

        arr = np.asarray(frame)
        Image.fromarray(arr.astype(np.uint8)).save(out_path)
        return
    except Exception as exc:
        raise RuntimeError(f"Cannot save frame type {type(frame)}") from exc


def run_demo(steps, seed, steering, throttle, out_path):
    from metadrive.envs import SafeMetaDriveEnv

    config = dict(
        start_seed=seed,
        traffic_density=0.1,
        map=3,
        map_config={"type": "block_sequence", "config": "XST"},
        accident_prob=0.0,
        num_scenarios=1,
        horizon=1000,
        use_render=False,
    )
    env = SafeMetaDriveEnv(config)
    frame = None
    last_info = {}

    try:
        reset_out = env.reset(seed=seed)
        if isinstance(reset_out, tuple):
            _ = reset_out[0]

        action = np.array([steering, throttle], dtype=np.float32)
        for step in range(steps):
            step_out = env.step(action)
            if len(step_out) == 5:
                _, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                _, reward, done, info = step_out
            last_info = info
            frame = env.render(
                mode="topdown",
                semantic_map=True,
                film_size=(900, 900),
                screen_size=(900, 900),
            )
            if done:
                break
    finally:
        env.close()

    if frame is None:
        raise RuntimeError("No frame was rendered.")

    save_frame(frame, out_path)
    print(f"Saved: {out_path}")
    print(f"Steps: {step + 1}")
    print(f"Reward: {float(reward):.4f}")
    print(f"Cost: {float(last_info.get('cost', 0.0)):.4f}")
    print(f"Crash: {bool(last_info.get('crash', False))}")
    print(f"Out of road: {bool(last_info.get('out_of_road', False))}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steering", type=float, default=0.0)
    parser.add_argument("--throttle", type=float, default=0.35)
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(RESULTS_DIR, "metadrive_topdown_demo.png"),
    )
    parser.add_argument(
        "--allow_download",
        action="store_true",
        help="Allow MetaDrive to download its simulator assets on first run.",
    )
    args = parser.parse_args()

    ready, asset_path = metadrive_assets_ready()
    if not ready and not args.allow_download:
        print("MetaDrive simulator assets are not installed yet.")
        print(f"Expected assets path: {asset_path}")
        print("Run again with --allow_download if you want MetaDrive to download them.")
        return

    run_demo(
        steps=args.steps,
        seed=args.seed,
        steering=args.steering,
        throttle=args.throttle,
        out_path=args.output,
    )


if __name__ == "__main__":
    main()
