import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

results_dir = Path(__file__).parent.parent / "benchmark_results"

csv_files = sorted(results_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in '{results_dir}'.")

csv_path = csv_files[0]
print(f"Loading: {csv_path}")


def prepare_dataframe(df):
    df = df.copy()

    df["plot_pct"] = pd.to_numeric(df["watermark_pct"], errors="coerce") if "watermark_pct" in df.columns else df["experiment"].apply(lambda t: float(re.search(r"pct=([0-9]*\.?[0-9]+)", str(t)).group(1)) if re.search(r"pct=([0-9]*\.?[0-9]+)", str(t)) else None)
    df["plot_chain"] = pd.to_numeric(df["chain_extension"], errors="coerce") if "chain_extension" in df.columns else df["experiment"].apply(lambda t: int(re.search(r"chain=\+?(\d+)", str(t)).group(1)) if re.search(r"chain=\+?(\d+)", str(t)) else None)

    df = df[df["plot_pct"].notna() & df["plot_chain"].notna()].copy()
    df["plot_pct"] = pd.to_numeric(df["plot_pct"])
    df["plot_chain"] = pd.to_numeric(df["plot_chain"])

    # watermark_gap > 0 means suspect looks more like watermarked than benign
    df["watermark_gap"] = (
        df["baseline_test_avg_distance_to_benign"]
        - df["baseline_test_avg_distance_to_watermarked"]
    )

    df = df.sort_values(["plot_chain", "plot_pct"])
    return df


df = pd.read_csv(csv_path)
df = prepare_dataframe(df)

if df.empty:
    raise ValueError("No rows found with both watermark_pct and chain_extension.")

chains = sorted(df["plot_chain"].unique())
pcts   = sorted(df["plot_pct"].unique())
pct_labels = [f"{int(p*100)}%" for p in pcts]

COLORS = ["#185FA5", "#0F6E56", "#993C1D"]


# ── FIGURE 1: Watermark gap mean ± std per chain ──────────────────────────────
fig1, axes1 = plt.subplots(1, len(chains), figsize=(5.5 * len(chains), 4.5), sharey=True)
if len(chains) == 1:
    axes1 = [axes1]

for ax, chain, color in zip(axes1, chains, COLORS):
    sub = df[df["plot_chain"] == chain].groupby("plot_pct")["watermark_gap"]
    means = sub.mean().reindex(pcts).values
    stds  = sub.std().reindex(pcts).fillna(0).values
    x = np.arange(len(pcts))

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5, label="No signal (gap = 0)")
    ax.plot(x, means, marker="o", color=color, linewidth=2, label="Mean watermark gap")
    ax.fill_between(x, means - stds, means + stds, color=color, alpha=0.15, label="± 1 std dev")

    ax.set_title(f"Chain +{int(chain)}", fontsize=13)
    ax.set_xlabel("Watermark percentage")
    if ax is axes1[0]:
        ax.set_ylabel("Watermark gap\n(dist_to_benign − dist_to_watermarked)")
    ax.set_xticks(x)
    ax.set_xticklabels(pct_labels)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9)

fig1.suptitle(
    "Watermark gap vs watermark percentage (mean ± 1 std dev, 3 repeats)\n"
    "Positive = suspect behaves more like watermarked model than benign",
    fontsize=13
)
fig1.tight_layout()


# ── FIGURE 2: All chains overlaid on one plot ────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(pcts))

ax2.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

for chain, color in zip(chains, COLORS):
    sub = df[df["plot_chain"] == chain].groupby("plot_pct")["watermark_gap"]
    means = sub.mean().reindex(pcts).values
    stds  = sub.std().reindex(pcts).fillna(0).values

    ax2.plot(x, means, marker="o", color=color, linewidth=2, label=f"Chain +{int(chain)}")
    ax2.fill_between(x, means - stds, means + stds, color=color, alpha=0.12)

ax2.set_xticks(x)
ax2.set_xticklabels(pct_labels)
ax2.set_xlabel("Watermark percentage")
ax2.set_ylabel("Watermark gap\n(dist_to_benign − dist_to_watermarked)")
ax2.set_title(
    "Watermark gap across chain lengths (mean ± 1 std dev, 3 repeats)\n"
    "Positive = suspect behaves more like watermarked model than benign",
    fontsize=12
)
ax2.grid(True, alpha=0.25)
ax2.legend(fontsize=10)
fig2.tight_layout()


# ── FIGURE 3: Heatmap of mean watermark gap ───────────────────────────────────
gap_matrix = np.array([
    [df[(df["plot_chain"] == c) & (df["plot_pct"] == p)]["watermark_gap"].mean() for p in pcts]
    for c in chains
])

fig3, ax3 = plt.subplots(figsize=(8, len(chains) * 1.6 + 2))
vmax = np.abs(gap_matrix).max()
im = ax3.imshow(gap_matrix, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)

ax3.set_xticks(range(len(pcts)))
ax3.set_xticklabels(pct_labels)
ax3.set_yticks(range(len(chains)))
ax3.set_yticklabels([f"+{int(c)}" for c in chains])
ax3.set_xlabel("Watermark percentage")
ax3.set_ylabel("Chain length")
ax3.set_title(
    "Mean watermark gap heatmap (3 repeats)\n"
    "Green = detectable signal  |  Red = no signal / reversed",
    fontsize=12
)

for i in range(len(chains)):
    for j in range(len(pcts)):
        val = gap_matrix[i, j]
        ax3.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=11, fontweight="bold")

fig3.colorbar(im, ax=ax3, label="Watermark gap")
fig3.tight_layout()

plt.show()