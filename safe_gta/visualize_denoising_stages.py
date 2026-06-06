"""
Denoising Stage Grid – Safe-GTA "bad → good trajectory" visualization.

Shows how a dangerous trajectory is repaired step-by-step during guided
diffusion denoising.  Each column is one snapshot in the process:

    Original | Noisy | guided t=20 | guided t=15 | guided t=10 | guided t=5 | Final

Per column:
  • speed profile over 50 timesteps (filled area, red→green gradient)
  • Safety Critic cost shown as a colored bar annotation

Bottom row:
  • Cost descent curve across all stages

Run:
    python -m safe_gta.visualize_denoising_stages --start_t 25 --beta 2.0
    python -m safe_gta.visualize_denoising_stages --start_t 25 --beta 2.0 --index 3039
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from safe_gta.generate_safe_gta import denormalize, load_ddpm, normalize
from safe_gta.safety_critic import load_checkpoint as load_critic_checkpoint

RESULTS_DIR = os.path.join(_PROJECT_ROOT, "safe_gta", "results")
DATA_DIR    = os.path.join(_PROJECT_ROOT, "safe_gta", "data")
CKPT_DIR    = os.path.join(_PROJECT_ROOT, "safe_gta", "checkpoints")

# MetaDrive observation feature indices (0-based inside joint obs+act tensor)
SPEED_IDX       = 7   # ego speed
LATERAL_IDX     = 4   # lane-relative lateral position
HEADING_COS_IDX = 1   # heading cosine  (for route reconstruction)
HEADING_SIN_IDX = 2   # heading sine


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def find_hdf5():
    for name in ["metadrive_mediumsparse.hdf5", "metadrive_mediummean.hdf5",
                 "metadrive_mediumdense.hdf5"]:
        p = os.path.join(DATA_DIR, name)
        if os.path.exists(p):
            return p
    matches = sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))
    if not matches:
        raise FileNotFoundError("No HDF5 found in safe_gta/data/. Run download_dataset.py")
    return matches[0]


def load_chunks(hdf5_path, seq_len):
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        obs  = np.asarray(f["observations"][:], dtype=np.float32)
        act  = np.asarray(f["actions"][:],      dtype=np.float32)
        cost = np.asarray(f["costs"][:],        dtype=np.float32)
        done = np.asarray(f["terminals"][:],    dtype=np.float32)
        if "timeouts" in f:
            done = np.logical_or(done, f["timeouts"][:]).astype(np.float32)
        if "velocity_costs" in f:
            vel_cost = np.asarray(f["velocity_costs"][:], dtype=np.float32)
        else:
            vel_cost = cost.copy()

    joints, costs, vel_costs = [], [], []
    start = 0
    ends = list(np.where(done > 0.5)[0]) + [len(obs) - 1]
    for end in ends:
        ep_obs   = obs[start:end + 1]
        ep_act   = act[start:end + 1]
        ep_cost  = cost[start:end + 1]
        ep_vel   = vel_cost[start:end + 1]
        for i in range(0, len(ep_obs) - seq_len + 1, seq_len):
            joints.append(np.concatenate([ep_obs[i:i + seq_len],
                                          ep_act[i:i + seq_len]], axis=-1))
            costs.append(ep_cost[i:i + seq_len])
            vel_costs.append(ep_vel[i:i + seq_len])
        start = end + 1

    return (np.stack(joints).astype(np.float32),
            np.stack(costs).astype(np.float32),
            np.stack(vel_costs).astype(np.float32))


# ---------------------------------------------------------------------------
# Repair with snapshot capture
# ---------------------------------------------------------------------------

def trace_repair(ddpm, critic, x_norm_batch, start_t, beta, capture_at, device):
    """
    Run guided repair on a batch, capturing normalized snapshots at each stage.
    Returns dict: stage_label -> (B, seq_len, n_features) numpy array
    """
    x = torch.tensor(x_norm_batch, dtype=torch.float32, device=device)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    B = x.shape[0]

    t_batch = torch.full((B,), start_t, dtype=torch.long, device=device)
    x_noisy = ddpm.schedule.q_sample(x, t_batch)

    snapshots = {
        "Original":          x.detach().cpu().numpy(),
        f"Noisy\n(t={start_t})": x_noisy.detach().cpu().numpy(),
    }

    ddpm.model.eval()
    critic.eval()
    x_cur = x_noisy.clone()

    for t_val in range(start_t - 1, -1, -1):
        t = torch.full((B,), t_val, dtype=torch.long, device=device)
        with torch.no_grad():
            x_cur = ddpm.schedule.p_sample_step(ddpm.model, x_cur, t)
        if beta > 0:
            x_g = x_cur.detach().requires_grad_(True)
            loss = critic(x_g).mean()
            grad = torch.autograd.grad(loss, x_g)[0].clamp(-1.0, 1.0)
            x_cur = (x_g - beta * grad).detach().clamp(-1.0, 1.0)

        if t_val in capture_at:
            snapshots[f"Guided\n(t={t_val})"] = x_cur.detach().cpu().numpy()

    snapshots["Final\n(Safe-GTA)"] = x_cur.detach().cpu().numpy()
    return snapshots


def batch_critic_cost(critic, arr, device):
    """arr: (B, seq_len, n_feat) or (seq_len, n_feat)"""
    critic.eval()
    with torch.no_grad():
        x = torch.tensor(arr, dtype=torch.float32, device=device)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        return critic(x).mean(dim=1).cpu().numpy()   # (B,)


# ---------------------------------------------------------------------------
# Trajectory selection
# ---------------------------------------------------------------------------

def select_best(joints, total_cost, meta, ddpm, critic,
                start_t, beta, capture_at, search_n, unsafe_min_cost,
                seed, device):
    rng = np.random.default_rng(seed)
    pool = np.where(total_cost > unsafe_min_cost)[0]
    if len(pool) == 0:
        pool = np.arange(len(joints))
    n = min(search_n, len(pool))
    cand_idx = rng.choice(pool, size=n, replace=False)

    batch = joints[cand_idx, :, :meta["n_features"]]
    batch_norm = normalize(batch, meta["norm_stats"], meta["n_features"]).astype(np.float32)

    torch.manual_seed(seed)
    snaps = trace_repair(ddpm, critic, batch_norm, start_t, beta, capture_at, device)

    orig_cost  = batch_critic_cost(critic, snaps["Original"], device)
    final_cost = batch_critic_cost(critic, snaps["Final\n(Safe-GTA)"], device)
    improvement = orig_cost - final_cost

    if np.any(improvement > 0):
        best_local = int(np.argmax(improvement))
    else:
        best_local = int(np.argmin(final_cost))

    global_idx = int(cand_idx[best_local])
    print(f"Selected trajectory index={global_idx}  "
          f"dataset_cost={total_cost[global_idx]:.3f}  "
          f"orig_critic={orig_cost[best_local]:.4f}  "
          f"final_critic={final_cost[best_local]:.4f}  "
          f"improvement={improvement[best_local]:.4f}")

    single_snaps = {lbl: arr[best_local] for lbl, arr in snaps.items()}
    return global_idx, single_snaps


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _cost_color(cost_val: float):
    """Interpolate red (1.0) → yellow (0.5) → green (0.0)."""
    c = float(np.clip(cost_val, 0.0, 1.0))
    if c >= 0.5:
        t = (c - 0.5) / 0.5
        r, g, b = 1.0, 1.0 - t, 0.0
    else:
        t = c / 0.5
        r, g, b = t, 1.0, 0.0
    return (r, g, b)


def plot_denoising_stages(
    raw_snapshots: dict,
    critic_costs: dict,
    source_total_cost: float,
    vel_cost_profile: np.ndarray,
    out_path: str,
    start_t: int,
    beta: float,
    norm_stats: dict,
    n_features: int,
):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

    labels = list(raw_snapshots.keys())
    n_stages = len(labels)
    steps = np.arange(raw_snapshots[labels[0]].shape[0])

    # ── figure layout ──────────────────────────────────────────────────────
    # Row 0 : speed profiles   (n_stages columns)
    # Row 1 : critic cost bars (n_stages columns)
    # Row 2 : cost descent curve (spans all columns)
    fig = plt.figure(figsize=(2.6 * n_stages, 9))
    fig.patch.set_facecolor("#F7F7F7")

    gs = GridSpec(
        3, n_stages,
        figure=fig,
        height_ratios=[3, 0.9, 2],
        hspace=0.55,
        wspace=0.32,
        top=0.88, bottom=0.07, left=0.06, right=0.98,
    )

    # Speed range (for shared y-axis)
    all_speeds = np.concatenate([
        raw_snapshots[lbl][:, SPEED_IDX] for lbl in labels
    ])
    s_lo = max(0.0, all_speeds.min() - 0.05)
    s_hi = all_speeds.max() + 0.05

    FILL_ALPHA = 0.30

    for col, lbl in enumerate(labels):
        raw   = raw_snapshots[lbl]           # (seq_len, n_feat)
        cost  = critic_costs[lbl]
        color = _cost_color(cost)

        # ── top panel: speed profile ──────────────────────────────────────
        ax_speed = fig.add_subplot(gs[0, col])
        speed = raw[:, SPEED_IDX]
        ax_speed.fill_between(steps, s_lo, speed,
                              color=color, alpha=FILL_ALPHA)
        ax_speed.plot(steps, speed, color=color, linewidth=1.8)

        # For "Original" also overlay the velocity cost from real labels
        if col == 0 and vel_cost_profile is not None:
            ax_speed.fill_between(steps, s_lo,
                                  s_lo + (vel_cost_profile / max(vel_cost_profile.max(), 1e-6))
                                  * (s_hi - s_lo) * 0.4,
                                  color="#cc0000", alpha=0.18, label="true vel cost")

        ax_speed.set_ylim(s_lo, s_hi)
        ax_speed.set_xlim(0, len(steps) - 1)
        ax_speed.set_xticks([])
        if col == 0:
            ax_speed.set_ylabel("Speed", fontsize=8)
        else:
            ax_speed.set_yticklabels([])

        ax_speed.spines["top"].set_visible(False)
        ax_speed.spines["right"].set_visible(False)
        ax_speed.grid(axis="y", alpha=0.2, linewidth=0.7)

        # Stage label + cost in title
        is_original = (col == 0)
        is_final    = (col == n_stages - 1)
        border_lw   = 2.2 if (is_original or is_final) else 0.8
        for spine in ax_speed.spines.values():
            spine.set_linewidth(border_lw)
            spine.set_edgecolor(color if not is_original else "#333333")

        cost_pct = int(round(cost * 100))
        ax_speed.set_title(
            lbl.replace("\n", " ") + f"\ncost={cost:.3f}",
            fontsize=8,
            fontweight="bold" if (is_original or is_final) else "normal",
            color="#333333",
        )

        # Arrow between columns (except after last)
        if col < n_stages - 1:
            ax_speed.annotate(
                "",
                xy=(1.12, 0.5), xycoords="axes fraction",
                xytext=(1.0, 0.5), textcoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2),
            )

        # ── middle panel: cost color bar ───────────────────────────────────
        ax_bar = fig.add_subplot(gs[1, col])
        ax_bar.barh([0], [cost], color=color, height=0.55, alpha=0.85)
        ax_bar.barh([0], [1.0],  color="#DDDDDD", height=0.55, alpha=0.45, zorder=0)
        ax_bar.set_xlim(0, 1.05)
        ax_bar.set_ylim(-0.7, 0.7)
        ax_bar.axis("off")
        ax_bar.text(cost + 0.03, 0, f"{cost:.2f}",
                    va="center", fontsize=8,
                    color="#333333",
                    fontweight="bold" if (is_original or is_final) else "normal")

    # ── bottom panel: cost descent curve ───────────────────────────────────
    ax_curve = fig.add_subplot(gs[2, :])
    cost_vals  = [critic_costs[lbl] for lbl in labels]
    x_pos      = np.arange(n_stages)
    bar_colors = [_cost_color(c) for c in cost_vals]

    for i, (cv, bc) in enumerate(zip(cost_vals, bar_colors)):
        ax_curve.bar(i, cv, color=bc, alpha=0.75, width=0.55, zorder=2)

    ax_curve.plot(x_pos, cost_vals, color="#333333", marker="o",
                  linewidth=2.0, zorder=3, markersize=6)

    # Shade the "safe zone"
    ax_curve.axhspan(0, 0.35, alpha=0.10, color="#00BB44", zorder=0)
    ax_curve.axhline(0.35, color="#00BB44", linestyle="--",
                     linewidth=1.2, alpha=0.7, label="safe threshold (0.35)")

    ax_curve.set_xticks(x_pos)
    ax_curve.set_xticklabels([lbl.replace("\n", " ") for lbl in labels],
                              fontsize=8.5, rotation=12, ha="right")
    ax_curve.set_ylabel("Mean Safety Critic Cost", fontsize=9)
    ax_curve.set_ylim(0, 1.05)
    ax_curve.set_title("Safety Critic score drops through guided denoising",
                       fontsize=10)
    ax_curve.legend(fontsize=8, loc="upper right")
    ax_curve.spines["top"].set_visible(False)
    ax_curve.spines["right"].set_visible(False)
    ax_curve.grid(axis="y", alpha=0.25)

    # ── super title ────────────────────────────────────────────────────────
    orig_cost  = cost_vals[0]
    final_cost = cost_vals[-1]
    fig.suptitle(
        f"Safe-GTA: Bad → Good Trajectory (start_t={start_t}, β={beta})\n"
        f"Safety Critic cost: {orig_cost:.3f} → {final_cost:.3f}  "
        f"(dataset velocity cost = {source_total_cost:.2f})",
        fontsize=12,
        fontweight="bold",
        y=0.97,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    start_t:        int   = 25,
    beta:           float = 2.0,
    index:          int   = None,
    search_n:       int   = 2000,
    seed:           int   = 7,
    unsafe_min_cost:float = 1.0,
    device:         str   = None,
    output:         str   = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    diffusion_ckpt = os.path.join(CKPT_DIR, "diffusion_metadrive_final.pt")
    critic_ckpt    = os.path.join(CKPT_DIR, "metadrive_safety_critic.pt")
    hdf5_path      = find_hdf5()

    print(f"Loading diffusion: {diffusion_ckpt}")
    ddpm, meta = load_ddpm(diffusion_ckpt, device)
    print(f"Loading critic:    {critic_ckpt}")
    critic, critic_payload = load_critic_checkpoint(critic_ckpt, device=device)

    if critic_payload["n_features"] != meta["n_features"]:
        raise ValueError("Critic / diffusion feature dimension mismatch")

    print(f"Loading dataset:   {hdf5_path}")
    joints, costs, vel_costs = load_chunks(hdf5_path, seq_len=meta["seq_len"])
    joints     = joints[:, :, :meta["n_features"]]
    total_cost = costs.sum(axis=1)

    # Stages to capture during denoising
    capture_at = {t for t in [20, 15, 10, 5] if t < start_t}

    if index is None:
        index, norm_snaps = select_best(
            joints, total_cost, meta, ddpm, critic,
            start_t, beta, capture_at, search_n, unsafe_min_cost, seed, device,
        )
    else:
        print(f"Using requested trajectory index: {index}")
        x_raw  = joints[index, :, :meta["n_features"]]
        x_norm = normalize(x_raw[None], meta["norm_stats"], meta["n_features"]).astype(np.float32)
        torch.manual_seed(seed)
        snaps_batch = trace_repair(ddpm, critic, x_norm, start_t, beta, capture_at, device)
        norm_snaps  = {lbl: arr[0] for lbl, arr in snaps_batch.items()}

    # Denormalize
    raw_snaps = {
        lbl: denormalize(arr[None], meta["norm_stats"], meta["n_features"])[0]
        for lbl, arr in norm_snaps.items()
    }

    # Critic costs (scalar per stage)
    critic_costs = {}
    for lbl, arr in norm_snaps.items():
        critic.eval()
        with torch.no_grad():
            x = torch.tensor(arr[None], dtype=torch.float32, device=device)
            critic_costs[lbl] = float(critic(x).mean().cpu())

    print("\nStage-by-stage Safety Critic cost:")
    for lbl, cv in critic_costs.items():
        print(f"  {lbl.replace(chr(10), ' '):22s}: {cv:.4f}")

    # Velocity cost profile from real HDF5 labels (for "Original" panel overlay)
    vel_profile = vel_costs[index]   # (seq_len,)
    # Clip to [0, 1] for display
    vel_profile = np.clip(vel_profile, 0.0, 1.0)

    out_path = output or os.path.join(RESULTS_DIR, "denoising_stage_grid.png")
    plot_denoising_stages(
        raw_snaps, critic_costs,
        source_total_cost=float(total_cost[index]),
        vel_cost_profile=vel_profile,
        out_path=out_path,
        start_t=start_t,
        beta=beta,
        norm_stats=meta["norm_stats"],
        n_features=meta["n_features"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_t",         type=int,   default=25)
    parser.add_argument("--beta",             type=float, default=2.0)
    parser.add_argument("--index",            type=int,   default=None,
                        help="Fixed trajectory index (skips auto search)")
    parser.add_argument("--search_n",         type=int,   default=2000,
                        help="How many trajectories to scan when auto-selecting")
    parser.add_argument("--seed",             type=int,   default=7)
    parser.add_argument("--unsafe_min_cost",  type=float, default=1.0,
                        help="Minimum dataset cost for auto-selection pool")
    parser.add_argument("--device",           type=str,   default=None)
    parser.add_argument("--output",           type=str,   default=None,
                        help="Output PNG path (default: results/denoising_stage_grid.png)")
    args = parser.parse_args()

    main(
        start_t        = args.start_t,
        beta           = args.beta,
        index          = args.index,
        search_n       = args.search_n,
        seed           = args.seed,
        unsafe_min_cost= args.unsafe_min_cost,
        device         = args.device,
        output         = args.output,
    )
