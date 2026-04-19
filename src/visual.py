import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# SET THIS TO YOUR CSVbenchmark_results\benchmark_proteins_20260419_233845.csv
# --------------------------------------------------
csv_path = Path("benchmark_results/benchmark_proteins_20260419_233845.csv")


# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def parse_pct(text):
    match = re.search(r"pct=([0-9]*\.?[0-9]+)", str(text))
    return float(match.group(1)) if match else None


def parse_chain(text):
    match = re.search(r"chain=\+?(\d+)", str(text))
    return int(match.group(1)) if match else None


def prepare_dataframe(df):
    df = df.copy()

    # Try to get pct directly from column first, otherwise from experiment text
    if "watermark_pct" in df.columns:
        df["plot_pct"] = pd.to_numeric(df["watermark_pct"], errors="coerce")
    else:
        df["plot_pct"] = df["experiment"].apply(parse_pct)

    # Chain is usually encoded in experiment string like pct=0.1_chain=+2
    if "chain_length" in df.columns:
        df["plot_chain"] = pd.to_numeric(df["chain_length"], errors="coerce")
    else:
        df["plot_chain"] = df["experiment"].apply(parse_chain)

    # Keep only rows that actually belong to the pct/chain benchmark grid
    df = df[df["plot_pct"].notna() & df["plot_chain"].notna()].copy()

    # Convert to numeric and sort
    df["plot_pct"] = pd.to_numeric(df["plot_pct"])
    df["plot_chain"] = pd.to_numeric(df["plot_chain"])
    df = df.sort_values(["plot_chain", "plot_pct"])

    return df


def aggregate_for_plot(df):
    grouped = (
        df.groupby(["plot_chain", "plot_pct"], as_index=False)
        .agg({
            "baseline_test_benign_avg_confidence": "mean",
            "baseline_test_watermarked_avg_confidence": "mean",
            "benign_test_acc": "mean",
            "watermarked_test_acc": "mean",
        })
    )

    grouped["confidence_gap"] = (
        grouped["baseline_test_watermarked_avg_confidence"]
        - grouped["baseline_test_benign_avg_confidence"]
    )

    return grouped


# --------------------------------------------------
# LOAD + PREPARE
# --------------------------------------------------
df = pd.read_csv(csv_path)
df = prepare_dataframe(df)

if df.empty:
    raise ValueError(
        "No rows found with both watermark percentage and chain in the experiment names.\n"
        "Make sure your experiment strings look like: pct=0.05_chain=+1"
    )

plot_df = aggregate_for_plot(df)

chains = sorted(plot_df["plot_chain"].unique())
pcts = sorted(plot_df["plot_pct"].unique())

if len(chains) == 0:
    raise ValueError("No chain values found after parsing.")


# --------------------------------------------------
# FIGURE 1: CONFIDENCE VS WATERMARK PERCENTAGE
# --------------------------------------------------
fig1, axes1 = plt.subplots(1, len(chains), figsize=(7 * len(chains), 5), sharey=True)

if len(chains) == 1:
    axes1 = [axes1]

for ax, chain in zip(axes1, chains):
    sub = plot_df[plot_df["plot_chain"] == chain].sort_values("plot_pct")

    ax.plot(
        sub["plot_pct"],
        sub["baseline_test_benign_avg_confidence"],
        marker="o",
        linewidth=2,
        label="Benign model confidence",
    )
    ax.plot(
        sub["plot_pct"],
        sub["baseline_test_watermarked_avg_confidence"],
        marker="o",
        linewidth=2,
        label="Watermarked model confidence",
    )

    ax.set_title(f"Chain +{int(chain)}")
    ax.set_xlabel("Watermark percentage")
    ax.set_ylabel("Average confidence")
    ax.set_ylim(0, 1)
    ax.set_xticks(pcts)
    ax.grid(True, alpha=0.3)
    ax.legend()

fig1.suptitle("Confidence vs watermark percentage for different chain lengths", fontsize=18)
fig1.tight_layout()
plt.show()


# --------------------------------------------------
# FIGURE 2: ACCURACY VS WATERMARK PERCENTAGE
# --------------------------------------------------
fig2, axes2 = plt.subplots(1, len(chains), figsize=(7 * len(chains), 5), sharey=True)

if len(chains) == 1:
    axes2 = [axes2]

for ax, chain in zip(axes2, chains):
    sub = plot_df[plot_df["plot_chain"] == chain].sort_values("plot_pct")

    ax.plot(
        sub["plot_pct"],
        sub["benign_test_acc"],
        marker="o",
        linewidth=2,
        label="Benign test accuracy",
    )
    ax.plot(
        sub["plot_pct"],
        sub["watermarked_test_acc"],
        marker="o",
        linewidth=2,
        label="Watermarked test accuracy",
    )

    ax.set_title(f"Chain +{int(chain)}")
    ax.set_xlabel("Watermark percentage")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1)
    ax.set_xticks(pcts)
    ax.grid(True, alpha=0.3)
    ax.legend()

fig2.suptitle("Accuracy vs watermark percentage for different chain lengths", fontsize=18)
fig2.tight_layout()
plt.show()


# --------------------------------------------------
# FIGURE 3: HEATMAP OF CONFIDENCE GAP
# watermarked confidence - benign confidence
# --------------------------------------------------
heatmap = plot_df.pivot(
    index="plot_chain",
    columns="plot_pct",
    values="confidence_gap"
).sort_index()

fig3, ax3 = plt.subplots(figsize=(8, 5))
im = ax3.imshow(heatmap.values, aspect="auto")

ax3.set_title("Confidence gap heatmap (watermarked - benign)")
ax3.set_xlabel("Watermark percentage")
ax3.set_ylabel("Chain length")

ax3.set_xticks(range(len(heatmap.columns)))
ax3.set_xticklabels([f"{x:.2f}" for x in heatmap.columns])

ax3.set_yticks(range(len(heatmap.index)))
ax3.set_yticklabels([f"+{int(x)}" for x in heatmap.index])

for i in range(len(heatmap.index)):
    for j in range(len(heatmap.columns)):
        value = heatmap.values[i, j]
        ax3.text(j, i, f"{value:.3f}", ha="center", va="center")

fig3.colorbar(im, ax=ax3, label="Confidence gap")
fig3.tight_layout()
plt.show()