"""
Intuitive bird-eye trajectory comparison: Original (dangerous) vs Safe-GTA Repaired (safe).

Layout
------
  Top-left  : Original trajectory on a schematic road  (red/hot speed coloring)
  Top-right : Repaired trajectory on the same road     (green/cool speed coloring)
  Bottom-left : Speed profile comparison
  Bottom-right: Safety Critic step scores comparison

The 2-D path is reconstructed from heading cos/sin + lateral-offset features.
Speed is used to scale step length so faster sections look longer on the map.

Run
---
  python -m safe_gta.visualize_trajectory_repair
  python -m safe_gta.visualize_trajectory_repair --index 3039 --start_t 25 --beta 2.0
  python -m safe_gta.visualize_trajectory_repair --search_n 3000 --unsafe_min_cost 1.0
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

RESULTS_DIR     = os.path.join(_PROJECT_ROOT, "safe_gta", "results")
CKPT_DIR        = os.path.join(_PROJECT_ROOT, "safe_gta", "checkpoints")
DATA_DIR        = os.path.join(_PROJECT_ROOT, "safe_gta", "data")

HEADING_COS_IDX = 2
HEADING_SIN_IDX = 3
LATERAL_IDX     = 4
SPEED_IDX       = 7

ROAD_HALF_WIDTH = 4.5   # units on the schematic map
LANE_WIDTH      = 2.25


# ── data helpers ─────────────────────────────────────────────────────────────

def find_hdf5():
    for name in ["metadrive_mediumsparse.hdf5",
                 "metadrive_mediummean.hdf5",
                 "metadrive_mediumdense.hdf5"]:
        p = os.path.join(DATA_DIR, name)
        if os.path.exists(p):
            return p
    m = sorted(glob.glob(os.path.join(DATA_DIR, "*.hdf5")))
    if not m:
        raise FileNotFoundError("No HDF5 in safe_gta/data/. Run download_dataset.py")
    return m[0]


def load_chunks(hdf5_path, seq_len):
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        obs  = np.asarray(f["observations"][:], dtype=np.float32)
        act  = np.asarray(f["actions"][:],      dtype=np.float32)
        cost = np.asarray(f["costs"][:],        dtype=np.float32)
        done = np.asarray(f["terminals"][:],    dtype=np.float32)
        if "timeouts" in f:
            done = np.logical_or(done, f["timeouts"][:]).astype(np.float32)
        vel_cost = np.asarray(
            f["velocity_costs"][:] if "velocity_costs" in f else f["costs"][:],
            dtype=np.float32,
        )
    joints, all_cost, all_vel = [], [], []
    start = 0
    ends = list(np.where(done > 0.5)[0]) + [len(obs) - 1]
    for end in ends:
        eo, ea, ec, ev = (obs[start:end+1], act[start:end+1],
                          cost[start:end+1], vel_cost[start:end+1])
        for i in range(0, len(eo) - seq_len + 1, seq_len):
            joints.append(np.concatenate([eo[i:i+seq_len], ea[i:i+seq_len]], -1))
            all_cost.append(ec[i:i+seq_len])
            all_vel.append(ev[i:i+seq_len])
        start = end + 1
    return (np.stack(joints).astype(np.float32),
            np.stack(all_cost).astype(np.float32),
            np.stack(all_vel).astype(np.float32))


# ── path reconstruction ───────────────────────────────────────────────────────

def reconstruct_path(raw):
    """
    Reconstruct approximate 2-D bird-eye path from heading + lateral features.

    - Heading cos/sin → direction at each step
    - Speed → step length (faster = car moved farther)
    - Lateral offset → displacement perpendicular to heading (lane-relative)

    Returns x, y (arrays of length seq_len), and rel_angle (heading).
    """
    hcos = np.clip(raw[:, HEADING_COS_IDX], -1, 1)
    hsin = np.clip(raw[:, HEADING_SIN_IDX], -1, 1)
    angle     = np.unwrap(np.arctan2(hsin, hcos))
    rel_angle = angle - angle[0]

    speed     = np.clip(raw[:, SPEED_IDX], 0.0, None)
    spd_norm  = speed / max(speed.max(), 1e-6)   # 0-1
    step_len  = 0.8 + spd_norm * 1.2             # 0.8 – 2.0 units

    # Forward integration along heading
    dx = np.cos(rel_angle) * step_len
    dy = np.sin(rel_angle) * step_len
    x  = np.concatenate([[0.0], np.cumsum(dx[:-1])])
    y  = np.concatenate([[0.0], np.cumsum(dy[:-1])])

    # Lateral offset (perpendicular to heading)
    # Use larger scale so lane-relative differences are visible
    lateral  = (raw[:, LATERAL_IDX] - raw[0, LATERAL_IDX]) * 14.0
    nx, ny   = -np.sin(rel_angle), np.cos(rel_angle)
    x += lateral * nx
    y += lateral * ny

    return x, y, rel_angle, speed


def smooth(arr, k=3):
    """Simple moving-average smoothing."""
    out = arr.copy()
    for i in range(len(arr)):
        lo, hi = max(0, i-k), min(len(arr), i+k+1)
        out[i] = arr[lo:hi].mean()
    return out


# ── repair ────────────────────────────────────────────────────────────────────

def run_repair(ddpm, critic, x_norm_single, start_t, beta, device):
    """Return repaired normalized array (seq_len, n_feat)."""
    x = torch.tensor(x_norm_single[None], dtype=torch.float32, device=device)
    B = 1
    t_b = torch.full((B,), start_t, dtype=torch.long, device=device)
    x_cur = ddpm.schedule.q_sample(x, t_b)
    ddpm.model.eval();  critic.eval()
    for t_val in range(start_t - 1, -1, -1):
        t = torch.full((B,), t_val, dtype=torch.long, device=device)
        with torch.no_grad():
            x_cur = ddpm.schedule.p_sample_step(ddpm.model, x_cur, t)
        if beta > 0:
            xg = x_cur.detach().requires_grad_(True)
            loss = critic(xg).mean()
            grad = torch.autograd.grad(loss, xg)[0].clamp(-1.0, 1.0)
            x_cur = (xg - beta * grad).detach().clamp(-1.0, 1.0)
    return x_cur[0].detach().cpu().numpy()


def critic_scores(critic, x_norm, device):
    with torch.no_grad():
        x = torch.tensor(x_norm[None], dtype=torch.float32, device=device)
        return critic(x).squeeze(0).cpu().numpy()


# ── trajectory selection ──────────────────────────────────────────────────────

def auto_select(joints, vel_costs, total_cost, meta, ddpm, critic,
                start_t, beta, search_n, unsafe_min_cost, seed, device):
    rng = np.random.default_rng(seed)
    pool = np.where(total_cost > unsafe_min_cost)[0]
    if len(pool) == 0:
        pool = np.arange(len(joints))
    cand = rng.choice(pool, size=min(search_n, len(pool)), replace=False)

    best_idx, best_score = None, -1.0
    print(f"Searching {len(cand)} candidates for best repair example ...")
    for i in cand:
        x_raw  = joints[i, :, :meta["n_features"]]
        x_norm = normalize(x_raw[None], meta["norm_stats"],
                           meta["n_features"]).astype(np.float32)[0]
        orig_c = float(critic_scores(critic, x_norm, device).mean())
        if orig_c < 0.6:
            continue
        torch.manual_seed(seed)
        rep_norm = run_repair(ddpm, critic, x_norm, start_t, beta, device)
        rep_c    = float(critic_scores(critic, rep_norm, device).mean())
        improve  = orig_c - rep_c

        # Also reward trajectories with more lateral variation (more visible)
        orig_raw = denormalize(x_norm[None], meta["norm_stats"], meta["n_features"])[0]
        rep_raw  = denormalize(rep_norm[None], meta["norm_stats"], meta["n_features"])[0]
        lateral_diff = float(np.abs(
            orig_raw[:, LATERAL_IDX] - rep_raw[:, LATERAL_IDX]
        ).mean())

        # Combined score: prioritise big critic improvement + visible lateral change
        score = improve * 0.7 + lateral_diff * 0.3

        if score > best_score:
            best_score = score
            best_idx = i
            print(f"  New best: idx={i}  orig={orig_c:.3f}  repaired={rep_c:.3f}  "
                  f"improve={improve:.3f}  lateral_diff={lateral_diff:.4f}  score={score:.3f}")

    if best_idx is None:
        best_idx = int(cand[0])
    return best_idx


# ── drawing helpers ───────────────────────────────────────────────────────────

def _draw_road(ax, x, y, angle):
    """Draw a schematic road centered on the given reference path."""
    from matplotlib.patches import FancyArrowPatch
    from matplotlib.collections import LineCollection

    n  = len(x)
    nx = -np.sin(angle)   # road-normal x
    ny =  np.cos(angle)   # road-normal y

    # Road surface (gray polygon)
    left_x  = x + ROAD_HALF_WIDTH * nx
    left_y  = y + ROAD_HALF_WIDTH * ny
    right_x = x - ROAD_HALF_WIDTH * nx
    right_y = y - ROAD_HALF_WIDTH * ny

    road_poly_x = np.concatenate([left_x, right_x[::-1]])
    road_poly_y = np.concatenate([left_y, right_y[::-1]])
    ax.fill(road_poly_x, road_poly_y, color="#D0D0D0", zorder=0, alpha=0.9)

    # Road edges (yellow solid)
    ax.plot(left_x,  left_y,  color="#F5C518", linewidth=2.2, zorder=1)
    ax.plot(right_x, right_y, color="#F5C518", linewidth=2.2, zorder=1)

    # Center dashed line (white)
    every = max(1, n // 12)
    for i in range(0, n - every, every * 2):
        ax.plot(x[i:i+every], y[i:i+every],
                color="white", linewidth=1.4, linestyle="-", zorder=2, alpha=0.7)

    # Lane boundary (dashed gray)
    for sign in (+1, -1):
        lx = x + sign * LANE_WIDTH * nx
        ly = y + sign * LANE_WIDTH * ny
        ax.plot(lx, ly, color="#999999", linewidth=0.8, linestyle="--",
                zorder=1, alpha=0.5)


def _plot_path_colored(ax, x, y, values, cmap, vmin, vmax, lw=3.2, zorder=4):
    """Plot path with line segments colored by values array."""
    from matplotlib.collections import LineCollection
    import matplotlib.pyplot as plt

    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=lw, zorder=zorder)
    lc.set_array(values[:-1])
    ax.add_collection(lc)
    return lc


def _car_marker(ax, x, y, angle, color, size=180, zorder=6):
    """Draw a triangle 'car' marker pointing in the direction of travel."""
    ax.scatter([x], [y], marker=(3, 0, np.rad2deg(angle) - 90),
               s=size, color=color, edgecolors="white",
               linewidths=1.2, zorder=zorder)


# ── main plot ─────────────────────────────────────────────────────────────────

def plot_comparison(
    orig_raw, rep_raw,
    orig_norm, rep_norm,
    vel_cost_profile,
    critic, device,
    out_path, index, start_t, beta, total_cost_scalar,
):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch

    ox, oy, o_ang, o_speed = reconstruct_path(orig_raw)
    rx, ry, r_ang, r_speed = reconstruct_path(rep_raw)

    # Smooth paths slightly for cleaner look
    ox, oy = smooth(ox, 2), smooth(oy, 2)
    rx, ry = smooth(rx, 2), smooth(ry, 2)

    orig_scores = critic_scores(critic, orig_norm, device)
    rep_scores  = critic_scores(critic, rep_norm,  device)
    steps       = np.arange(len(ox))

    orig_mean = float(orig_scores.mean())
    rep_mean  = float(rep_scores.mean())

    # Use original path as road reference (shows where the car 'should' go)
    road_x, road_y, road_ang = ox, oy, o_ang

    # Speed range (shared color scale)
    spd_all = np.concatenate([o_speed, r_speed])
    spd_min, spd_max = spd_all.min(), max(spd_all.max(), spd_all.min() + 0.1)

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10), facecolor="#1C1C1C")
    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        height_ratios=[3, 1.2],
        width_ratios=[1, 1, 0.05],
        hspace=0.38, wspace=0.18,
        top=0.88, bottom=0.08, left=0.05, right=0.97,
    )

    ax_orig  = fig.add_subplot(gs[0, 0])
    ax_rep   = fig.add_subplot(gs[0, 1])
    ax_cbar  = fig.add_subplot(gs[0, 2])
    ax_speed = fig.add_subplot(gs[1, 0])
    ax_critic= fig.add_subplot(gs[1, 1])

    for ax in [ax_orig, ax_rep, ax_speed, ax_critic]:
        ax.set_facecolor("#2A2A2A")
        ax.tick_params(colors="#BBBBBB", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#555555")

    # ── TOP-LEFT: original ─────────────────────────────────────────────────
    _draw_road(ax_orig, road_x, road_y, road_ang)
    lc_orig = _plot_path_colored(ax_orig, ox, oy, o_speed,
                                 cmap="hot_r", vmin=spd_min, vmax=spd_max,
                                 lw=3.8, zorder=4)

    # ⚠️ Danger markers where velocity cost > 0
    danger_steps = np.where(vel_cost_profile > 0.01)[0]
    if len(danger_steps):
        danger_sizes = 120 + 800 * np.clip(vel_cost_profile[danger_steps], 0, 1)
        ax_orig.scatter(ox[danger_steps], oy[danger_steps],
                        s=danger_sizes, marker="^",
                        color="#FF4444", edgecolors="white",
                        linewidths=0.8, zorder=5, alpha=0.85)

    # Car markers at start / quarter / mid / end
    for t, col in [(0, "white"), (len(ox)//3, "#FF9900"),
                   (2*len(ox)//3, "#FF4444"), (-1, "#DD2222")]:
        _car_marker(ax_orig, ox[t], oy[t], o_ang[t], color=col, zorder=7)

    ax_orig.set_title("[X]  Original Trajectory  (DANGEROUS)",
                      fontsize=13, fontweight="bold", color="#FF5555", pad=8)
    ax_orig.set_aspect("equal")
    ax_orig.axis("off")

    # Critic cost badge
    ax_orig.text(0.03, 0.04,
                 f"Safety Critic: {orig_mean:.3f}",
                 transform=ax_orig.transAxes,
                 fontsize=10, fontweight="bold",
                 color="white",
                 bbox=dict(facecolor="#CC2222", edgecolor="none",
                           boxstyle="round,pad=0.35", alpha=0.85))

    # Dataset cost badge
    ax_orig.text(0.03, 0.13,
                 f"Dataset cost: {total_cost_scalar:.2f}",
                 transform=ax_orig.transAxes,
                 fontsize=9, color="#FFAAAA",
                 bbox=dict(facecolor="#333333", edgecolor="none",
                           boxstyle="round,pad=0.3", alpha=0.75))

    # ── TOP-RIGHT: repaired ────────────────────────────────────────────────
    _draw_road(ax_rep, road_x, road_y, road_ang)
    lc_rep = _plot_path_colored(ax_rep, rx, ry, r_speed,
                                cmap="cool", vmin=spd_min, vmax=spd_max,
                                lw=3.8, zorder=4)

    # Safe markers where repaired critic < 0.3
    safe_steps = np.where(rep_scores < 0.30)[0]
    if len(safe_steps):
        ax_rep.scatter(rx[safe_steps], ry[safe_steps],
                       s=55, marker="o", color="#44FF88",
                       edgecolors="white", linewidths=0.6,
                       zorder=5, alpha=0.55)

    for t, col in [(0, "white"), (len(rx)//3, "#44DDAA"),
                   (2*len(rx)//3, "#22BB88"), (-1, "#00AA66")]:
        _car_marker(ax_rep, rx[t], ry[t], r_ang[t], color=col, zorder=7)

    ax_rep.set_title("[OK]  Repaired Trajectory  (SAFER)",
                     fontsize=13, fontweight="bold", color="#55FF88", pad=8)
    ax_rep.set_aspect("equal")
    ax_rep.axis("off")

    ax_rep.text(0.03, 0.04,
                f"Safety Critic: {rep_mean:.3f}",
                transform=ax_rep.transAxes,
                fontsize=10, fontweight="bold",
                color="white",
                bbox=dict(facecolor="#117733", edgecolor="none",
                          boxstyle="round,pad=0.35", alpha=0.85))

    improvement_pct = (orig_mean - rep_mean) / max(orig_mean, 1e-6) * 100
    ax_rep.text(0.03, 0.13,
                f"Cost ↓ {improvement_pct:.0f}%",
                transform=ax_rep.transAxes,
                fontsize=9, color="#AAFFCC",
                bbox=dict(facecolor="#333333", edgecolor="none",
                          boxstyle="round,pad=0.3", alpha=0.75))

    # Align axes limits so both maps use the same scale
    all_x = np.concatenate([ox, rx])
    all_y = np.concatenate([oy, ry])
    margin = ROAD_HALF_WIDTH * 1.8
    xlo, xhi = all_x.min() - margin, all_x.max() + margin
    ylo, yhi = all_y.min() - margin, all_y.max() + margin
    ax_orig.set_xlim(xlo, xhi);  ax_orig.set_ylim(ylo, yhi)
    ax_rep.set_xlim(xlo, xhi);   ax_rep.set_ylim(ylo, yhi)

    # ── Shared colorbar ────────────────────────────────────────────────────
    import matplotlib as mpl
    sm = mpl.cm.ScalarMappable(
        cmap="RdYlGn_r",
        norm=mpl.colors.Normalize(vmin=spd_min, vmax=spd_max),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label("Speed  (higher = more risk)", color="#BBBBBB", fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#BBBBBB")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#BBBBBB", fontsize=7)
    ax_cbar.set_facecolor("#1C1C1C")

    # ── BOTTOM-LEFT: speed comparison ─────────────────────────────────────
    ax_speed.plot(steps, o_speed, color="#FF6655", linewidth=2.2,
                  label=f"Original  (mean={o_speed.mean():.2f})")
    ax_speed.fill_between(steps, 0, o_speed, color="#FF4444", alpha=0.18)
    ax_speed.plot(steps, r_speed, color="#44FFAA", linewidth=2.2,
                  label=f"Repaired  (mean={r_speed.mean():.2f})")
    ax_speed.fill_between(steps, 0, r_speed, color="#22CC66", alpha=0.18)

    if len(danger_steps):
        ax_speed.scatter(danger_steps,
                         o_speed[danger_steps],
                         s=40, color="#FF4444", marker="^", zorder=5,
                         label="True velocity cost steps")

    ax_speed.set_title("Speed Profile", color="#DDDDDD", fontsize=9)
    ax_speed.set_xlabel("Trajectory step", color="#AAAAAA", fontsize=8)
    ax_speed.set_ylabel("Speed", color="#AAAAAA", fontsize=8)
    ax_speed.legend(fontsize=7.5, facecolor="#2A2A2A",
                    labelcolor="#DDDDDD", edgecolor="#555555")
    ax_speed.grid(axis="y", alpha=0.2, color="#888888")
    ax_speed.set_xlim(0, len(steps) - 1)

    # ── BOTTOM-RIGHT: critic scores ────────────────────────────────────────
    ax_critic.plot(steps, orig_scores, color="#FF6655", linewidth=2.2,
                   label=f"Original  (mean={orig_mean:.3f})")
    ax_critic.fill_between(steps, 0, orig_scores, color="#FF4444", alpha=0.18)
    ax_critic.plot(steps, rep_scores, color="#44FFAA", linewidth=2.2,
                   label=f"Repaired  (mean={rep_mean:.3f})")
    ax_critic.fill_between(steps, 0, rep_scores, color="#22CC66", alpha=0.18)
    ax_critic.axhline(0.35, color="#FFCC00", linestyle="--", linewidth=1.2,
                      alpha=0.7, label="Safe threshold (0.35)")
    ax_critic.set_ylim(0, 1.05)
    ax_critic.set_title("Safety Critic Score per Step", color="#DDDDDD", fontsize=9)
    ax_critic.set_xlabel("Trajectory step", color="#AAAAAA", fontsize=8)
    ax_critic.set_ylabel("Critic cost (0=safe, 1=dangerous)",
                         color="#AAAAAA", fontsize=8)
    ax_critic.legend(fontsize=7.5, facecolor="#2A2A2A",
                     labelcolor="#DDDDDD", edgecolor="#555555")
    ax_critic.grid(axis="y", alpha=0.2, color="#888888")
    ax_critic.set_xlim(0, len(steps) - 1)

    # ── super title ────────────────────────────────────────────────────────
    fig.suptitle(
        f"Safe-GTA  —  Bad Trajectory → Repaired Trajectory  "
        f"(start_t={start_t}, β={beta},  traj index={index})\n"
        f"Safety Critic: {orig_mean:.3f}  →  {rep_mean:.3f}   "
        f"(↓ {improvement_pct:.0f}%)",
        fontsize=13, fontweight="bold", color="white", y=0.97,
    )

    # Legend for ⚠️ markers
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#FF5555", linewidth=2.5, label="Original path"),
        Line2D([0], [0], color="#44FFAA", linewidth=2.5, label="Repaired path"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#FF4444",
               markersize=9, label="[!] True velocity-cost steps"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#44FF88",
               markersize=7, label="[OK] Safe steps (critic < 0.3)"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="white",
               markersize=9, label="[CAR] Position markers"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5,
               facecolor="#2A2A2A", labelcolor="#DDDDDD",
               edgecolor="#555555", fontsize=8.5,
               bbox_to_anchor=(0.5, 0.01))

    note = ("Bird-eye path reconstructed from heading + lateral-offset features  "
            "│  Not a MetaDrive world-coordinate replay")
    fig.text(0.5, 0.045, note, ha="center", fontsize=7.5, color="#777777")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=155, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n✓  Saved: {out_path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main(
    start_t:         int   = 25,
    beta:            float = 2.0,
    index:           int   = None,
    search_n:        int   = 2000,
    seed:            int   = 7,
    unsafe_min_cost: float = 1.0,
    device:          str   = None,
    output:          str   = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ddpm, meta = load_ddpm(
        os.path.join(CKPT_DIR, "diffusion_metadrive_final.pt"), device)
    critic, payload = load_critic_checkpoint(
        os.path.join(CKPT_DIR, "metadrive_safety_critic.pt"), device=device)
    if payload["n_features"] != meta["n_features"]:
        raise ValueError("Critic / diffusion feature dimension mismatch")

    hdf5_path = find_hdf5()
    print(f"Loading dataset: {hdf5_path}")
    joints, costs, vel_costs = load_chunks(hdf5_path, seq_len=meta["seq_len"])
    joints     = joints[:, :, :meta["n_features"]]
    total_cost = costs.sum(axis=1)

    # ── select trajectory ──────────────────────────────────────────────────
    if index is None:
        index = auto_select(
            joints, vel_costs, total_cost, meta, ddpm, critic,
            start_t, beta, search_n, unsafe_min_cost, seed, device,
        )
    print(f"\nUsing trajectory index={index}  "
          f"dataset_total_cost={total_cost[index]:.3f}")

    x_raw  = joints[index, :, :meta["n_features"]]
    x_norm = normalize(x_raw[None], meta["norm_stats"],
                       meta["n_features"]).astype(np.float32)[0]

    # ── repair ─────────────────────────────────────────────────────────────
    torch.manual_seed(seed)
    rep_norm = run_repair(ddpm, critic, x_norm, start_t, beta, device)

    orig_raw = denormalize(x_norm[None], meta["norm_stats"],
                           meta["n_features"])[0]
    rep_raw  = denormalize(rep_norm[None], meta["norm_stats"],
                           meta["n_features"])[0]

    orig_c = float(critic_scores(critic, x_norm, device).mean())
    rep_c  = float(critic_scores(critic, rep_norm, device).mean())
    print(f"Original  critic mean = {orig_c:.4f}")
    print(f"Repaired  critic mean = {rep_c:.4f}")
    print(f"Improvement           = {orig_c - rep_c:.4f}")

    vel_profile = np.clip(vel_costs[index], 0.0, 1.0)

    out_path = output or os.path.join(RESULTS_DIR, "trajectory_repair_comparison.png")
    plot_comparison(
        orig_raw, rep_raw,
        x_norm, rep_norm,
        vel_profile,
        critic, device,
        out_path, index, start_t, beta,
        total_cost_scalar=float(total_cost[index]),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start_t",          type=int,   default=25)
    ap.add_argument("--beta",             type=float, default=2.0)
    ap.add_argument("--index",            type=int,   default=None,
                    help="Fixed trajectory index. Omit to auto-search.")
    ap.add_argument("--search_n",         type=int,   default=2000,
                    help="Candidates to scan during auto-search")
    ap.add_argument("--seed",             type=int,   default=7)
    ap.add_argument("--unsafe_min_cost",  type=float, default=1.0)
    ap.add_argument("--device",           type=str,   default=None)
    ap.add_argument("--output",           type=str,   default=None)
    args = ap.parse_args()
    main(
        start_t         = args.start_t,
        beta            = args.beta,
        index           = args.index,
        search_n        = args.search_n,
        seed            = args.seed,
        unsafe_min_cost = args.unsafe_min_cost,
        device          = args.device,
        output          = args.output,
    )
