"""
visualize.py — plots for the watermarking benchmark results

Reads a CSV produced by benchmark.py. Point CSV_PATH at your file.

Columns used (wm_score columns are optional — skipped if absent):
    watermark_pct, chain_extension,
    benign_test_acc, watermarked_test_acc, accuracy_drop,
    reference_suspect_minus_benign_confidence,
    reference_suspect_minus_benign_wm_score,   # optional
    control_suspect_minus_benign_confidence,

Usage:
    python visualize.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # remove this line if you want interactive windows
import matplotlib.pyplot as plt

# ── Config ─────────────────────────────────────────────────────────────────────

DATASET_NAME = "ENZYMES"    # used for titles and output filenames
VARIANT      = "LABEL AGNOSTIC"      # used for titles and output filenames
SHOW_PLOTS   = False         # set True to open windows (requires a display)

# ── Point this at your CSV ─────────────────────────────────────────────────────
# Option A: explicit path
CSV_PATH = Path("benchmark_results/subtle/enzymes_20260519_144549.csv")

# Un-comment the line below (and comment out CSV_PATH above) to use auto-discover:
# CSV_PATH = find_latest_csv(Path("benchmark_results") / VARIANT, DATASET_NAME)

# ── Output directory ───────────────────────────────────────────────────────────

FIGURES_DIR = Path("benchmark_results") / VARIANT / "figures" / DATASET_NAME.lower()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

def pct_label(x: float) -> str:
    return "NA" if pd.isna(x) else f"{int(round(x * 100))}%"

def chain_label(x: float) -> str:
    return "NA" if pd.isna(x) else f"+{int(round(x))}"

def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

# ── Plot functions ─────────────────────────────────────────────────────────────

def plot_heatmap(
    pivot_df: pd.DataFrame,
    title: str,
    cbar_label: str,
    filename: str,
    center_zero: bool = False,
    fmt: str = "{:+.3f}",
    cmap: str = "RdYlGn",
) -> None:
    if pivot_df.empty:
        print(f"Skipped empty heatmap: {title}")
        return

    values = pivot_df.values.astype(float)
    if np.all(np.isnan(values)):
        print(f"Skipped all-NaN heatmap: {title}")
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    if center_zero:
        vmax = np.nanmax(np.abs(values))
        vmax = vmax if vmax > 0 else 1e-6
        vmin = -vmax
    else:
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        if np.isclose(vmin, vmax):
            delta = 1e-6 if vmin == 0 else abs(vmin) * 0.05
            vmin -= delta
            vmax += delta

    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels([pct_label(c) for c in pivot_df.columns])
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels([chain_label(i) for i in pivot_df.index])
    ax.set_xlabel("Watermark percentage")
    ax.set_ylabel("Chain extension")
    ax.set_title(title)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            ax.text(
                j, i,
                "NA" if np.isnan(val) else fmt.format(val),
                ha="center", va="center", fontsize=11, fontweight="bold",
            )

    fig.colorbar(im, ax=ax).set_label(cbar_label)
    save_fig(fig, FIGURES_DIR / filename)


def plot_signal_by_chain(
    summary: pd.DataFrame,
    value_col: str,
    std_col: str,
    title: str,
    ylabel: str,
    filename: str,
) -> None:
    chains = sorted(summary["chain_extension"].dropna().unique())
    if not chains:
        print(f"Skipped empty line plot: {title}")
        return

    fig, axes = plt.subplots(1, len(chains), figsize=(6 * len(chains), 4.5), sharey=True)
    if len(chains) == 1:
        axes = [axes]

    for ax, chain in zip(axes, chains):
        sub = summary[summary["chain_extension"] == chain].sort_values("watermark_pct")
        x = sub["watermark_pct"].to_numpy()
        y = sub[value_col].to_numpy()
        s = sub[std_col].fillna(0).to_numpy()

        ax.plot(x, y, marker="o", linewidth=2)
        ax.fill_between(x, y - s, y + s, alpha=0.18)
        ax.axhline(0.0, linestyle="--", linewidth=1)
        ax.set_title(f"Chain {chain_label(chain)}")
        ax.set_xlabel("Watermark percentage")
        ax.set_xticks(x)
        ax.set_xticklabels([pct_label(v) for v in x])
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=16)
    save_fig(fig, FIGURES_DIR / filename)


def plot_accuracy(summary: pd.DataFrame) -> None:
    chains = sorted(summary["chain_extension"].dropna().unique())
    fig, axes = plt.subplots(1, len(chains), figsize=(6 * len(chains), 4.5), sharey=True)
    if len(chains) == 1:
        axes = [axes]

    for ax, chain in zip(axes, chains):
        sub = summary[summary["chain_extension"] == chain].sort_values("watermark_pct")
        x   = sub["watermark_pct"].to_numpy()
        ax.plot(x, sub["mean_benign_acc"].to_numpy(),       marker="o", label="Benign")
        ax.plot(x, sub["mean_watermarked_acc"].to_numpy(),  marker="s", label="Watermarked")
        ax.set_title(f"Chain {chain_label(chain)}")
        ax.set_xlabel("Watermark percentage")
        ax.set_xticks(x)
        ax.set_xticklabels([pct_label(v) for v in x])
        ax.set_ylabel("Test accuracy")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.suptitle(
        f"Test accuracy vs watermark percentage ({DATASET_NAME} / {VARIANT})",
        fontsize=16,
    )
    save_fig(fig, FIGURES_DIR / f"{DATASET_NAME.lower()}_accuracy.png")


# ── Load & validate ────────────────────────────────────────────────────────────

if CSV_PATH is None or not Path(CSV_PATH).exists():
    raise FileNotFoundError(
        f"CSV not found: {CSV_PATH}\n"
        "Set CSV_PATH at the top of this script to point at your results file."
    )

print(f"Loading: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"Shape:   {df.shape}")
print(f"Columns: {list(df.columns)}")

# Cast all numeric columns that are present (skip any that are absent)
NUMERIC_COLS = [
    "watermark_pct", "chain_extension",
    "benign_test_acc", "watermarked_test_acc", "accuracy_drop",
    "reference_benign_avg_confidence",
    "reference_watermarked_avg_confidence",
    "reference_suspect_avg_confidence",
    "reference_suspect_minus_benign_confidence",
    "reference_suspect_minus_watermarked_confidence",
    "reference_suspect_minus_benign_wm_score",        # optional
    "reference_suspect_minus_watermarked_wm_score",   # optional
    "control_suspect_minus_benign_confidence",
    "control_suspect_minus_watermarked_confidence",
]
for col in NUMERIC_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Derived columns
df["watermark_signal"] = df["reference_suspect_minus_benign_confidence"]
df["control_gap"]      = df["control_suspect_minus_benign_confidence"]
df["accuracy_drop"]    = df["benign_test_acc"] - df["watermarked_test_acc"]

# wm_score signal — only if the column exists
HAS_WM_SCORE = "reference_suspect_minus_benign_wm_score" in df.columns
if HAS_WM_SCORE:
    df["watermark_signal_wm"] = df["reference_suspect_minus_benign_wm_score"]

# ── Summarise over repeats ─────────────────────────────────────────────────────

agg_dict = dict(
    mean_signal          = ("watermark_signal", "mean"),
    std_signal           = ("watermark_signal", "std"),
    mean_control_gap     = ("control_gap",      "mean"),
    std_control_gap      = ("control_gap",      "std"),
    mean_utility_drop    = ("accuracy_drop",    "mean"),
    std_utility_drop     = ("accuracy_drop",    "std"),
    mean_benign_acc      = ("benign_test_acc",  "mean"),
    mean_watermarked_acc = ("watermarked_test_acc", "mean"),
    count                = ("watermark_signal", "count"),
)
if HAS_WM_SCORE:
    agg_dict["mean_signal_wm"] = ("watermark_signal_wm", "mean")
    agg_dict["std_signal_wm"]  = ("watermark_signal_wm", "std")

summary = (
    df.groupby(["chain_extension", "watermark_pct"])
    .agg(**agg_dict)
    .reset_index()
)

print(f"\nSummary shape: {summary.shape}")
print(summary.to_string())

# ── Heatmaps ───────────────────────────────────────────────────────────────────

def _pivot(col):
    return (
        summary.pivot(index="chain_extension", columns="watermark_pct", values=col)
        .sort_index()
        .sort_index(axis=1)
    )

plot_heatmap(
    pivot_df    = _pivot("mean_signal"),
    title       = f"Watermark signal — confidence ({DATASET_NAME} / {VARIANT})",
    cbar_label  = "suspect conf − benign conf",
    filename    = f"{DATASET_NAME.lower()}_signal_heatmap.png",
    center_zero = True,
)

if HAS_WM_SCORE:
    plot_heatmap(
        pivot_df    = _pivot("mean_signal_wm"),
        title       = f"Watermark signal — WM head score ({DATASET_NAME} / {VARIANT})",
        cbar_label  = "suspect WM score − benign WM score",
        filename    = f"{DATASET_NAME.lower()}_signal_wm_heatmap.png",
        center_zero = True,
    )
else:
    print("Skipped WM-score heatmap (column not present in CSV).")

plot_heatmap(
    pivot_df    = _pivot("mean_utility_drop"),
    title       = f"Accuracy drop ({DATASET_NAME} / {VARIANT})",
    cbar_label  = "benign acc − watermarked acc",
    filename    = f"{DATASET_NAME.lower()}_accuracy_drop_heatmap.png",
    center_zero = True,
)

plot_heatmap(
    pivot_df    = _pivot("mean_control_gap"),
    title       = f"Control gap ({DATASET_NAME} / {VARIANT})",
    cbar_label  = "control conf − benign conf",
    filename    = f"{DATASET_NAME.lower()}_control_gap_heatmap.png",
    center_zero = True,
)

# ── Line plots ─────────────────────────────────────────────────────────────────

plot_signal_by_chain(
    summary   = summary,
    value_col = "mean_signal",
    std_col   = "std_signal",
    title     = f"Watermark signal vs watermark % ({DATASET_NAME} / {VARIANT})",
    ylabel    = "suspect conf − benign conf",
    filename  = f"{DATASET_NAME.lower()}_signal_by_chain.png",
)

plot_signal_by_chain(
    summary   = summary,
    value_col = "mean_control_gap",
    std_col   = "std_control_gap",
    title     = f"Control gap vs watermark % ({DATASET_NAME} / {VARIANT})",
    ylabel    = "control conf − benign conf",
    filename  = f"{DATASET_NAME.lower()}_control_gap_by_chain.png",
)

plot_accuracy(summary)

print("\nDone.")