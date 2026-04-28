from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================
DATASET_NAME = "ENZYMES"   # change to "ENZYMES" when needed
SHOW_PLOTS = True


# ============================================================
# PATHS
# ============================================================
THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parent
RESULTS_DIR = SRC_ROOT / "benchmark" / "results" / DATASET_NAME.lower()
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================
def find_latest_csv(results_dir: Path, prefix: str, dataset_name: str) -> Path | None:
    dataset_slug = dataset_name.lower()
    matches = sorted(
        results_dir.glob(f"{prefix}_{dataset_slug}_*.csv"),
        key=lambda p: p.stat().st_mtime,
    )
    return matches[-1] if matches else None


def load_csv_or_none(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def ensure_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def boolish_to_float(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(float)

    lowered = series.astype(str).str.strip().str.lower()
    mapped = lowered.map(
        {
            "true": 1.0,
            "false": 0.0,
            "1": 1.0,
            "0": 0.0,
        }
    )
    return pd.to_numeric(mapped, errors="coerce")


def pct_to_label(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"{int(round(x * 100))}%"


def chain_to_label(x: float) -> str:
    if pd.isna(x):
        return "NA"
    return f"+{int(round(x))}"


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Saved: {path}")


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
        if vmax == 0 or np.isnan(vmax):
            vmax = 1e-6
        vmin = -vmax
    else:
        vmin = np.nanmin(values)
        vmax = np.nanmax(values)
        if np.isclose(vmin, vmax):
            delta = 1e-6 if vmin == 0 else abs(vmin) * 0.05
            vmin -= delta
            vmax += delta

    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels([pct_to_label(x) for x in pivot_df.columns])
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels([chain_to_label(x) for x in pivot_df.index])

    ax.set_xlabel("Watermark percentage")
    ax.set_ylabel("Chain length")
    ax.set_title(title)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            text = "NA" if np.isnan(val) else fmt.format(val)
            ax.text(j, i, text, ha="center", va="center", fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    save_fig(fig, FIGURES_DIR / filename)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def plot_signal_by_chain(
    summary_df: pd.DataFrame,
    value_col: str,
    std_col: str,
    title: str,
    ylabel: str,
    filename: str,
) -> None:
    chains = sorted(summary_df["chain_extension"].dropna().unique())
    if len(chains) == 0:
        print(f"Skipped empty line plot: {title}")
        return

    fig, axes = plt.subplots(1, len(chains), figsize=(6 * len(chains), 4.5), sharey=True)
    if len(chains) == 1:
        axes = [axes]

    for ax, chain in zip(axes, chains):
        sub = summary_df[summary_df["chain_extension"] == chain].sort_values("watermark_pct")
        x = sub["watermark_pct"].to_numpy()
        y = sub[value_col].to_numpy()
        s = sub[std_col].fillna(0).to_numpy()

        ax.plot(x, y, marker="o", linewidth=2, label="Mean watermark signal")
        ax.fill_between(x, y - s, y + s, alpha=0.18, label="± 1 std dev")
        ax.axhline(0.0, linestyle="--", linewidth=1)

        ax.set_title(f"Chain +{int(chain)}")
        ax.set_xlabel("Watermark percentage")
        ax.set_xticks(x)
        ax.set_xticklabels([pct_to_label(v) for v in x])
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.suptitle(title, fontsize=18)
    save_fig(fig, FIGURES_DIR / filename)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def plot_grouped_attack_panels(
    attack_df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    filename: str,
    title: str,
) -> None:
    panels = [
        {
            "attack_name": "blind_pruning",
            "panel_title": "Blind pruning",
            "x_col": "pruning_rate",
            "hue_col": None,
            "x_label": "Pruning rate",
        },
        {
            "attack_name": "blind_finetune",
            "panel_title": "Blind fine-tuning",
            "x_col": "finetune_epochs",
            "hue_col": "attack_learning_rate",
            "x_label": "Fine-tune epochs",
        },
        {
            "attack_name": "informed_pruning",
            "panel_title": "Informed pruning",
            "x_col": "pruning_rate",
            "hue_col": "clean_preservation_weight",
            "x_label": "Pruning rate",
        },
        {
            "attack_name": "informed_finetune",
            "panel_title": "Informed fine-tuning",
            "x_col": "finetune_epochs",
            "hue_col": "lambda_adv",
            "x_label": "Fine-tune epochs",
        },
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, panel in zip(axes, panels):
        sub = attack_df[attack_df["attack_name"] == panel["attack_name"]].copy()

        if sub.empty or metric_col not in sub.columns:
            ax.set_title(panel["panel_title"])
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            continue

        x_col = panel["x_col"]
        hue_col = panel["hue_col"]

        sub = sub.dropna(subset=[x_col, metric_col])

        if sub.empty:
            ax.set_title(panel["panel_title"])
            ax.text(0.5, 0.5, "No valid data", ha="center", va="center")
            ax.axis("off")
            continue

        if hue_col is None or hue_col not in sub.columns or sub[hue_col].dropna().nunique() <= 1:
            grouped = (
                sub.groupby(x_col)[metric_col]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values(x_col)
            )

            x = grouped[x_col].to_numpy()
            y = grouped["mean"].to_numpy()
            s = grouped["std"].fillna(0).to_numpy()

            ax.plot(x, y, marker="o", linewidth=2)
            ax.fill_between(x, y - s, y + s, alpha=0.18)
        else:
            for hue_val, hue_sub in sorted(sub.groupby(hue_col), key=lambda item: str(item[0])):
                grouped = (
                    hue_sub.groupby(x_col)[metric_col]
                    .agg(["mean", "std"])
                    .reset_index()
                    .sort_values(x_col)
                )

                x = grouped[x_col].to_numpy()
                y = grouped["mean"].to_numpy()
                s = grouped["std"].fillna(0).to_numpy()

                ax.plot(x, y, marker="o", linewidth=2, label=f"{hue_col}={hue_val}")
                ax.fill_between(x, y - s, y + s, alpha=0.12)

            ax.legend(fontsize=9)

        if metric_col in {"gap_retention_ratio"}:
            ax.axhline(1.0, linestyle="--", linewidth=1, alpha=0.7)
            ax.axhline(0.0, linestyle=":", linewidth=1, alpha=0.7)
        elif metric_col in {"attack_success_by_confidence"}:
            ax.set_ylim(-0.05, 1.05)

        ax.set_title(panel["panel_title"])
        ax.set_xlabel(panel["x_label"])
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    fig.suptitle(title, fontsize=18)
    save_fig(fig, FIGURES_DIR / filename)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def plot_attack_success_bar(attack_df: pd.DataFrame, filename: str) -> None:
    if attack_df.empty or "attack_success_by_confidence" not in attack_df.columns:
        print("Skipped attack success bar plot: no valid attack data")
        return

    summary = (
        attack_df.groupby("attack_name")["attack_success_by_confidence"]
        .mean()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(summary.index, summary.values)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Attack success rate")
    ax.set_title(f"Attack success rate by confidence ({DATASET_NAME})")
    ax.grid(True, axis="y", alpha=0.25)

    for i, v in enumerate(summary.values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontweight="bold")

    save_fig(fig, FIGURES_DIR / filename)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# LOAD BASE BENCHMARK
# ============================================================
base_csv = find_latest_csv(RESULTS_DIR, "base_benchmark", DATASET_NAME)
base_df = load_csv_or_none(base_csv)

if base_df is None or base_df.empty:
    raise FileNotFoundError(
        f"Could not find base benchmark CSV for {DATASET_NAME} in {RESULTS_DIR}"
    )

print(f"Loaded base benchmark: {base_df.shape}")
print(f"Base CSV: {base_csv}")

base_df = ensure_numeric(
    base_df,
    [
        "watermark_pct",
        "chain_extension",
        "benign_test_acc",
        "watermarked_test_acc",
        "accuracy_drop",
        "reference_suspect_minus_benign_confidence",
        "reference_watermarked_minus_benign_confidence",
        "control_suspect_minus_benign_confidence",
    ],
)

if "reference_signal_positive_vs_benign" in base_df.columns:
    base_df["reference_signal_positive_vs_benign"] = boolish_to_float(
        base_df["reference_signal_positive_vs_benign"]
    )

if "control_signal_positive_vs_benign" in base_df.columns:
    base_df["control_signal_positive_vs_benign"] = boolish_to_float(
        base_df["control_signal_positive_vs_benign"]
    )

# Core metrics from the base benchmark file you showed
base_df["watermark_signal"] = base_df["reference_suspect_minus_benign_confidence"]
base_df["control_gap"] = base_df["control_suspect_minus_benign_confidence"]
base_df["utility_drop"] = base_df["accuracy_drop"]

# Summary over repeats
base_summary = (
    base_df.groupby(["chain_extension", "watermark_pct"])
    .agg(
        mean_signal=("watermark_signal", "mean"),
        std_signal=("watermark_signal", "std"),
        mean_control_gap=("control_gap", "mean"),
        std_control_gap=("control_gap", "std"),
        mean_utility_drop=("utility_drop", "mean"),
        std_utility_drop=("utility_drop", "std"),
        mean_benign_acc=("benign_test_acc", "mean"),
        mean_watermarked_acc=("watermarked_test_acc", "mean"),
        positive_rate=("reference_signal_positive_vs_benign", "mean")
        if "reference_signal_positive_vs_benign" in base_df.columns
        else ("watermark_signal", lambda s: np.nan),
        count=("watermark_signal", "count"),
    )
    .reset_index()
)

# Heatmaps
signal_heatmap = base_summary.pivot(
    index="chain_extension", columns="watermark_pct", values="mean_signal"
).sort_index().sort_index(axis=1)

accuracy_drop_heatmap = base_summary.pivot(
    index="chain_extension", columns="watermark_pct", values="mean_utility_drop"
).sort_index().sort_index(axis=1)

control_gap_heatmap = base_summary.pivot(
    index="chain_extension", columns="watermark_pct", values="mean_control_gap"
).sort_index().sort_index(axis=1)

positive_rate_heatmap = base_summary.pivot(
    index="chain_extension", columns="watermark_pct", values="positive_rate"
).sort_index().sort_index(axis=1)

plot_heatmap(
    pivot_df=signal_heatmap,
    title=f"Mean watermark signal heatmap ({DATASET_NAME})",
    cbar_label="Watermark signal\n(reference suspect confidence - benign confidence)",
    filename=f"{DATASET_NAME.lower()}_base_signal_heatmap.png",
    center_zero=True,
    fmt="{:+.3f}",
)

plot_signal_by_chain(
    summary_df=base_summary,
    value_col="mean_signal",
    std_col="std_signal",
    title=(
        f"Watermark signal vs watermark percentage ({DATASET_NAME})\n"
        f"Positive = watermarked model is more confident than benign on verification graphs"
    ),
    ylabel="Watermark signal",
    filename=f"{DATASET_NAME.lower()}_base_signal_gap_by_chain.png",
)

plot_heatmap(
    pivot_df=accuracy_drop_heatmap,
    title=f"Mean accuracy drop heatmap ({DATASET_NAME})",
    cbar_label="Accuracy drop\n(benign test acc - watermarked test acc)",
    filename=f"{DATASET_NAME.lower()}_base_accuracy_drop_heatmap.png",
    center_zero=True,
    fmt="{:+.3f}",
)

plot_heatmap(
    pivot_df=control_gap_heatmap,
    title=f"Benign control gap heatmap ({DATASET_NAME})",
    cbar_label="Control gap\n(benign copy confidence - benign confidence)",
    filename=f"{DATASET_NAME.lower()}_base_control_gap_heatmap.png",
    center_zero=True,
    fmt="{:+.3f}",
)

plot_heatmap(
    pivot_df=positive_rate_heatmap,
    title=f"Positive-signal rate heatmap ({DATASET_NAME})",
    cbar_label="Rate of repeats with positive signal",
    filename=f"{DATASET_NAME.lower()}_base_positive_rate_heatmap.png",
    center_zero=False,
    fmt="{:.2f}",
    cmap="YlGn",
)


# ============================================================
# LOAD ATTACK BENCHMARK
# ============================================================
attack_csv = find_latest_csv(RESULTS_DIR, "attack_benchmark", DATASET_NAME)
attack_df = load_csv_or_none(attack_csv)

if attack_df is None or attack_df.empty:
    print(f"No attack benchmark CSV found for {DATASET_NAME} in {RESULTS_DIR}")
else:
    print(f"Loaded attack benchmark: {attack_df.shape}")
    print(f"Attack CSV: {attack_csv}")

    attack_df = ensure_numeric(
        attack_df,
        [
            "watermark_pct",
            "chain_extension",
            "pruning_rate",
            "finetune_epochs",
            "attack_learning_rate",
            "clean_preservation_weight",
            "lambda_adv",
            "suspect_test_acc",
            "suspect_minus_benign_confidence",
            "watermarked_minus_benign_confidence",
            "baseline_confidence_gap",
            "gap_retention_ratio",
        ],
    )

    if "attack_success_by_confidence" in attack_df.columns:
        attack_df["attack_success_by_confidence"] = boolish_to_float(
            attack_df["attack_success_by_confidence"]
        )

    if "detected_by_confidence" in attack_df.columns:
        attack_df["detected_by_confidence"] = boolish_to_float(
            attack_df["detected_by_confidence"]
        )

    # Exclude reference rows for the attack robustness plots
    attack_only_df = attack_df[attack_df["attack_family"] != "reference"].copy()

    if not attack_only_df.empty:
        plot_grouped_attack_panels(
            attack_df=attack_only_df,
            metric_col="gap_retention_ratio",
            ylabel="Gap retention ratio",
            filename=f"{DATASET_NAME.lower()}_attack_gap_retention.png",
            title=(
                f"Attack robustness: retained watermark signal ({DATASET_NAME})\n"
                f"1.0 = full signal remains, 0.0 = erased, negative = reversed"
            ),
        )

        plot_grouped_attack_panels(
            attack_df=attack_only_df,
            metric_col="attack_success_by_confidence",
            ylabel="Attack success rate",
            filename=f"{DATASET_NAME.lower()}_attack_success_rate.png",
            title=(
                f"Attack success by confidence ({DATASET_NAME})\n"
                f"1.0 = attack removed positive watermark signal"
            ),
        )

        plot_grouped_attack_panels(
            attack_df=attack_only_df,
            metric_col="suspect_test_acc",
            ylabel="Suspect test accuracy",
            filename=f"{DATASET_NAME.lower()}_attack_suspect_accuracy.png",
            title=f"Post-attack suspect accuracy ({DATASET_NAME})",
        )

        plot_attack_success_bar(
            attack_df=attack_only_df,
            filename=f"{DATASET_NAME.lower()}_attack_success_bar.png",
        )

print("\nDone.")