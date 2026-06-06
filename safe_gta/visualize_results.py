"""
Visualize and compare baseline results.

Run:
  python -m safe_gta.visualize_results
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(_PROJECT_ROOT, "safe_gta", "results")


def load_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    dataset = str(data.get("dataset", "")).lower()
    if "_debug" in dataset:
        print(f"Skipping debug result file: {filename}")
        return None
    return data


def plot_baseline_comparison(b1, b2, safe_gta):
    methods, returns, safeties, costs = [], [], [], []

    if b1:
        methods.append("Baseline 1\n(CQL, raw data)")
        returns.append(b1["normalized_return"])
        safeties.append(b1["safety_success_rate"])
        costs.append(b1["constraint_violation_cost"])

    if b2:
        methods.append("Baseline 2\n(CQL + GTA)")
        returns.append(b2["normalized_return"])
        safeties.append(b2["safety_success_rate"])
        costs.append(b2["constraint_violation_cost"])

    methods.append("Safe-GTA\n(CQL + GTA + Critic)")
    if safe_gta:
        returns.append(safe_gta["normalized_return"])
        safeties.append(safe_gta["safety_success_rate"])
        costs.append(safe_gta["constraint_violation_cost"])
    else:
        returns.append(None)
        safeties.append(None)
        costs.append(None)

    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.suptitle("Baseline Comparison - MetaDrive Offline RL",
                 fontsize=14, fontweight="bold")

    colors = ["#4C72B0", "#55A868", "#C44E52"]
    labels = ["Normalized Return (higher is better)",
              "Safety Success Rate (higher is better)",
              "Constraint Violation Cost (lower is better)"]
    data_series = [returns, safeties, costs]

    for ax, data, label, color in zip(axes, data_series, labels, colors):
        vals = [v if v is not None else 0 for v in data]
        bars = ax.bar(x, vals, color=color, alpha=0.85, width=0.5)

        for bar, original in zip(bars, data):
            if original is None:
                bar.set_hatch("////")
                bar.set_alpha(0.3)
                bar.set_edgecolor("gray")
                ax.text(bar.get_x() + bar.get_width() / 2, 0.05, "pending",
                        ha="center", va="bottom", fontsize=9,
                        color="gray", style="italic")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01, f"{original:.3f}",
                        ha="center", va="bottom", fontsize=10,
                        fontweight="bold")

        present_vals = [v for v in vals if v is not None]
        ymax = max(max(present_vals), 1.0) * 1.25 if present_vals else 2
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=9)
        ax.set_title(label, fontsize=11)
        ax.set_ylim(0, ymax)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "baseline_comparison.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_diffusion_loss():
    import torch

    ckpt_path = os.path.join(_PROJECT_ROOT, "safe_gta", "checkpoints",
                             "diffusion_metadrive_final.pt")
    if not os.path.exists(ckpt_path):
        return

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    losses = payload.get("losses", [])
    if not losses:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, color="#4C72B0", linewidth=1.5)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("MetaDrive Diffusion Model - Training Loss", fontsize=13)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.annotate(f"Final: {losses[-1]:.4f}",
                xy=(len(losses) - 1, losses[-1]),
                xytext=(len(losses) * 0.7, losses[0] * 0.6),
                arrowprops=dict(arrowstyle="->", color="gray"),
                fontsize=10, color="#C44E52")

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "diffusion_metadrive_loss.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    b1 = load_json("baseline1_cql_results.json")
    b2 = load_json("baseline2_cql_results.json")
    safe_gta = load_json("safe_gta_cql_results.json")

    print("=== Loaded Results ===")
    if b1:
        print(f"Baseline 1 - Return={b1['normalized_return']}  "
              f"Safety={b1['safety_success_rate']}  "
              f"Cost={b1['constraint_violation_cost']}")
    if b2:
        print(f"Baseline 2 - Return={b2['normalized_return']}  "
              f"Safety={b2['safety_success_rate']}  "
              f"Cost={b2['constraint_violation_cost']}")
    if safe_gta:
        print(f"Safe-GTA - Return={safe_gta['normalized_return']}  "
              f"Safety={safe_gta['safety_success_rate']}  "
              f"Cost={safe_gta['constraint_violation_cost']}")

    plot_baseline_comparison(b1, b2, safe_gta)
    plot_diffusion_loss()

    print("\nAll plots saved to safe_gta/results/")
    print(f"Open the folder to view:\n  {RESULTS_DIR}")


if __name__ == "__main__":
    main()
