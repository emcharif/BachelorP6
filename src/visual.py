"""
visual.py — Watermark benchmark visualiser
Usage: python visual.py [csv_path]
Defaults to 'data.csv' in the current directory if no argument given.
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ── colour palette ────────────────────────────────────────────────────────────
BLUE   = "#378ADD"
GREEN  = "#1D9E75"
CORAL  = "#D85A30"
AMBER  = "#BA7517"
PURPLE = "#7F77DD"
GRAY   = "#888780"
LIGHT  = "#F1EFE8"

CHAIN_COLORS = {"+1": BLUE, "+2": GREEN, "+3": CORAL}
PCT_COLORS   = {"5%": PURPLE, "10%": BLUE, "20%": GREEN, "30%": AMBER}
PCT_ORDER    = ["5%", "10%", "20%", "30%"]
CHAIN_ORDER  = ["+1", "+2", "+3"]

plt.rcParams.update({
    "font.family":      "sans-serif",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.color":       "#E0DDD6",
    "grid.linewidth":   0.5,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
})


# ── helpers ───────────────────────────────────────────────────────────────────

def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["pct_label"]   = df["watermark_pct"].apply(lambda x: f"{int(round(x*100))}%")
    df["chain_label"] = df["chain_extension"].apply(lambda x: f"+{x}")
    return df


def _mean_ci(series: pd.Series):
    """Return (mean, half-width of 95% CI) for a series."""
    n   = len(series)
    m   = series.mean()
    if n < 2:
        return m, 0.0
    se  = series.std(ddof=1) / np.sqrt(n)
    return m, 1.96 * se


def pivot_mean(df, index, columns, values):
    """Pivot with mean aggregation."""
    return df.pivot_table(index=index, columns=columns, values=values,
                          aggfunc="mean")


# ── individual panels ─────────────────────────────────────────────────────────

def plot_accuracy_heatmap(ax, df):
    """Heatmap: watermarked test accuracy by pct × chain_extension."""
    piv = pivot_mean(df, "pct_label", "chain_label", "watermarked_test_acc")
    piv = piv.reindex(index=PCT_ORDER, columns=CHAIN_ORDER)

    im  = ax.imshow(piv.values, cmap="RdYlGn", vmin=0.3, vmax=0.8,
                    aspect="auto")
    ax.set_xticks(range(len(CHAIN_ORDER)))
    ax.set_xticklabels(CHAIN_ORDER)
    ax.set_yticks(range(len(PCT_ORDER)))
    ax.set_yticklabels(PCT_ORDER)
    ax.set_xlabel("Chain extension")
    ax.set_ylabel("Watermark %")
    ax.set_title("Watermarked test accuracy", fontweight="bold")
    ax.grid(False)

    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=8, color="black")
    plt.colorbar(im, ax=ax, shrink=0.8)


def plot_accuracy_drop_heatmap(ax, df):
    """Heatmap: accuracy drop by pct × chain_extension."""
    piv = pivot_mean(df, "pct_label", "chain_label", "accuracy_drop")
    piv = piv.reindex(index=PCT_ORDER, columns=CHAIN_ORDER)

    im  = ax.imshow(piv.values, cmap="RdYlGn_r", vmin=-0.1, vmax=0.15,
                    aspect="auto")
    ax.set_xticks(range(len(CHAIN_ORDER)))
    ax.set_xticklabels(CHAIN_ORDER)
    ax.set_yticks(range(len(PCT_ORDER)))
    ax.set_yticklabels(PCT_ORDER)
    ax.set_xlabel("Chain extension")
    ax.set_ylabel("Watermark %")
    ax.set_title("Accuracy drop (benign − watermarked)", fontweight="bold")
    ax.grid(False)

    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center",
                        fontsize=8, color="black")
    plt.colorbar(im, ax=ax, shrink=0.8)


def plot_confidence_gap_by_chain(ax, df):
    """Bar chart: mean confidence gap (watermarked − benign) by chain extension."""
    df = df.copy()
    df["conf_gap"] = (df["reference_watermarked_test_watermarked_avg_confidence"]
                      - df["reference_watermarked_test_benign_avg_confidence"])

    grouped = df.groupby("chain_label")["conf_gap"]
    means, cis, labels = [], [], []
    for lbl in CHAIN_ORDER:
        if lbl in grouped.groups:
            m, ci = _mean_ci(grouped.get_group(lbl))
            means.append(m); cis.append(ci); labels.append(lbl)

    colors = [CHAIN_COLORS[l] for l in labels]
    bars = ax.bar(labels, means, color=colors, width=0.5,
                  yerr=cis, capsize=5, error_kw={"linewidth": 1.2})
    ax.axhline(0, color=GRAY, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Chain extension")
    ax.set_ylabel("Confidence gap")
    ax.set_title("Watermark confidence signal\n(watermarked − benign avg)", fontweight="bold")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, m + (0.001 if m >= 0 else -0.003),
                f"{m:+.4f}", ha="center", va="bottom" if m >= 0 else "top",
                fontsize=8)


def plot_confidence_gap_by_pct(ax, df):
    """Bar chart: mean confidence gap by watermark percentage."""
    df = df.copy()
    df["conf_gap"] = (df["reference_watermarked_test_watermarked_avg_confidence"]
                      - df["reference_watermarked_test_benign_avg_confidence"])

    grouped = df.groupby("pct_label")["conf_gap"]
    means, cis, labels = [], [], []
    for lbl in PCT_ORDER:
        if lbl in grouped.groups:
            m, ci = _mean_ci(grouped.get_group(lbl))
            means.append(m); cis.append(ci); labels.append(lbl)

    colors = [PCT_COLORS[l] for l in labels]
    bars = ax.bar(labels, means, color=colors, width=0.5,
                  yerr=cis, capsize=5, error_kw={"linewidth": 1.2})
    ax.axhline(0, color=GRAY, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Watermark %")
    ax.set_ylabel("Confidence gap")
    ax.set_title("Watermark confidence signal\nby watermark percentage", fontweight="bold")
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, m + (0.001 if m >= 0 else -0.003),
                f"{m:+.4f}", ha="center", va="bottom" if m >= 0 else "top",
                fontsize=8)


def plot_distance_to_benign(ax, df):
    """Grouped bars: mean distance to benign distribution by pct × chain."""
    col   = "reference_watermarked_test_avg_distance_to_benign"
    piv   = pivot_mean(df, "pct_label", "chain_label", col)
    piv   = piv.reindex(index=PCT_ORDER, columns=CHAIN_ORDER)

    n_groups = len(PCT_ORDER)
    n_chains = len(CHAIN_ORDER)
    x        = np.arange(n_groups)
    width    = 0.22

    for i, ch in enumerate(CHAIN_ORDER):
        vals = [piv.loc[p, ch] if ch in piv.columns and p in piv.index
                else np.nan for p in PCT_ORDER]
        ax.bar(x + (i - 1) * width, vals, width,
               label=ch, color=CHAIN_COLORS[ch], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(PCT_ORDER)
    ax.set_xlabel("Watermark %")
    ax.set_ylabel("Distance (avg)")
    ax.set_title("Avg distance to benign distribution", fontweight="bold")
    ax.legend(title="Chain ext.", fontsize=8, title_fontsize=8)


def plot_signal_positive_rate(ax, df):
    """Stacked bars: fraction of runs where signal_positive_vs_benign is True."""
    col = "reference_signal_positive_vs_benign"

    piv = df.groupby(["pct_label", "chain_label"])[col].mean().unstack("chain_label")
    piv = piv.reindex(index=PCT_ORDER, columns=CHAIN_ORDER).fillna(0)

    x     = np.arange(len(PCT_ORDER))
    width = 0.22
    for i, ch in enumerate(CHAIN_ORDER):
        ax.bar(x + (i - 1) * width, piv[ch].values, width,
               label=ch, color=CHAIN_COLORS[ch], alpha=0.85)

    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(PCT_ORDER)
    ax.set_xlabel("Watermark %")
    ax.set_ylabel("Fraction positive")
    ax.set_title("Detection rate\n(signal positive vs benign)", fontweight="bold")
    ax.axhline(0.5, color=GRAY, linewidth=0.8, linestyle="--", label="50%")
    ax.legend(title="Chain ext.", fontsize=8, title_fontsize=8)


def plot_scatter_acc_vs_confidence(ax, df):
    """Scatter: watermarked accuracy vs confidence gap, coloured by chain extension."""
    df = df.copy()
    df["conf_gap"] = (df["reference_watermarked_test_watermarked_avg_confidence"]
                      - df["reference_watermarked_test_benign_avg_confidence"])

    for ch in CHAIN_ORDER:
        sub = df[df["chain_label"] == ch]
        ax.scatter(sub["watermarked_test_acc"], sub["conf_gap"],
                   color=CHAIN_COLORS[ch], label=ch,
                   s=50, alpha=0.75, edgecolors="white", linewidths=0.4)

    ax.axhline(0, color=GRAY, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Watermarked test accuracy")
    ax.set_ylabel("Confidence gap")
    ax.set_title("Accuracy vs confidence signal", fontweight="bold")
    ax.legend(title="Chain ext.", fontsize=8, title_fontsize=8)


def plot_pvalue_distribution(ax, df):
    """Box plot of p-values (vs benign) across seeds, grouped by chain extension."""
    col = "reference_watermarked_test_p_value_vs_benign"
    data_by_chain = [df[df["chain_label"] == ch][col].dropna().values
                     for ch in CHAIN_ORDER]

    bp = ax.boxplot(data_by_chain, patch_artist=True, widths=0.45,
                    medianprops={"color": "black", "linewidth": 1.5})
    for patch, ch in zip(bp["boxes"], CHAIN_ORDER):
        patch.set_facecolor(CHAIN_COLORS[ch])
        patch.set_alpha(0.75)

    ax.axhline(0.05, color=CORAL, linewidth=1.2, linestyle="--", label="p=0.05")
    ax.set_xticks(range(1, len(CHAIN_ORDER)+1))
    ax.set_xticklabels(CHAIN_ORDER)
    ax.set_xlabel("Chain extension")
    ax.set_ylabel("p-value (vs benign)")
    ax.set_title("p-value distribution\n(watermarked vs benign)", fontweight="bold")
    ax.legend(fontsize=8)


def plot_benign_vs_watermarked_conf(ax, df):
    """Scatter: benign avg confidence vs watermarked avg confidence per run."""
    benign_col = "reference_watermarked_test_benign_avg_confidence"
    wm_col     = "reference_watermarked_test_watermarked_avg_confidence"

    for ch in CHAIN_ORDER:
        sub = df[df["chain_label"] == ch]
        ax.scatter(sub[benign_col], sub[wm_col],
                   color=CHAIN_COLORS[ch], label=ch,
                   s=50, alpha=0.75, edgecolors="white", linewidths=0.4)

    lims = [min(df[benign_col].min(), df[wm_col].min()) - 0.01,
            max(df[benign_col].max(), df[wm_col].max()) + 0.01]
    ax.plot(lims, lims, color=GRAY, linewidth=0.8, linestyle="--")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel("Benign avg confidence")
    ax.set_ylabel("Watermarked avg confidence")
    ax.set_title("Confidence: benign vs watermarked\n(above diagonal → watermark raises conf.)", fontweight="bold")
    ax.legend(title="Chain ext.", fontsize=8, title_fontsize=8)


# ── main figure ───────────────────────────────────────────────────────────────

def build_figure(df: pd.DataFrame) -> plt.Figure:
    fig = plt.figure(figsize=(18, 20))
    fig.suptitle("GNN Watermark Benchmark — Proteins (pct × chain_extension sweep)",
                 fontsize=14, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(4, 3, figure=fig,
                           hspace=0.55, wspace=0.38,
                           top=0.94, bottom=0.04)

    plot_accuracy_heatmap(fig.add_subplot(gs[0, 0]), df)
    plot_accuracy_drop_heatmap(fig.add_subplot(gs[0, 1]), df)
    plot_benign_vs_watermarked_conf(fig.add_subplot(gs[0, 2]), df)

    plot_confidence_gap_by_chain(fig.add_subplot(gs[1, 0]), df)
    plot_confidence_gap_by_pct(fig.add_subplot(gs[1, 1]), df)
    plot_distance_to_benign(fig.add_subplot(gs[1, 2]), df)

    plot_signal_positive_rate(fig.add_subplot(gs[2, 0]), df)
    plot_pvalue_distribution(fig.add_subplot(gs[2, 1]), df)
    plot_scatter_acc_vs_confidence(fig.add_subplot(gs[2, 2]), df)

    # bottom row: summary note
    ax_note = fig.add_subplot(gs[3, :])
    ax_note.axis("off")

    # per-run summary table
    summary = df.groupby(["pct_label", "chain_label"]).agg(
        seeds=("seed", "nunique"),
        mean_wm_acc=("watermarked_test_acc", "mean"),
        mean_acc_drop=("accuracy_drop", "mean"),
        pct_positive=("reference_signal_positive_vs_benign", "mean"),
    ).reset_index()
    summary.columns = ["WM %", "Chain", "Seeds",
                       "Mean WM acc", "Mean acc drop", "% signal positive"]

    col_labels = summary.columns.tolist()
    cell_text  = [[str(r["WM %"]), str(r["Chain"]),
                   str(int(r["Seeds"])),
                   f"{r['Mean WM acc']:.3f}",
                   f"{r['Mean acc drop']:+.3f}",
                   f"{r['% signal positive']*100:.0f}%"]
                  for _, r in summary.iterrows()]

    tbl = ax_note.table(cellText=cell_text, colLabels=col_labels,
                        loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.35)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        if r == 0:
            cell.set_facecolor("#378ADD")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F7F6F2")

    ax_note.set_title("Summary table (means over seeds)", fontsize=10,
                      fontweight="bold", pad=8)

    return fig


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "src/benchmark/results/base/base_benchmark_proteins_20260428_124956.csv"
    print(f"Loading {path} …")
    df  = load(path)
    print(f"  {len(df)} rows, {df['seed'].nunique()} seeds, "
          f"{df['watermark_pct'].nunique()} pct levels, "
          f"{df['chain_extension'].nunique()} chain levels")

    fig = build_figure(df)

    out = path.replace(".csv", "_visual.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    main()