"""
benchmark.py — GNN watermarking benchmark

Three named variants:
  - "subtle"     : graphs_to_watermark      + feature_mode="subtle"
  - "same_label" : graphs_to_watermark_same_label + feature_mode="subtle"

Usage:
    python benchmark.py
"""

import copy
import csv
import hashlib
import json
import os
import random
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch
from dotenv import load_dotenv

from src.GNN.Trainer import Trainer
from src.GNN.Evaluator import Evaluator
from src.graph_analyzer import GraphAnalyzer
from src.inject_chain import inject_chain
from src.utils import UtilityFunctions

load_dotenv()

# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_DATASETS              = ["PROTEINS"]
DEFAULT_WATERMARK_PERCENTAGES = [0.05, 0.1, 0.2, 0.3]
DEFAULT_CHAIN_EXTENSIONS      = [1,3,5,10,20]

DEFAULT_EPOCHS        = 50
DEFAULT_BATCH_SIZE    = 64
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_HIDDEN_DIM    = 128

DEFAULT_REPEATS            = 10
DEFAULT_VERIFICATION_COUNT = 50
DEFAULT_TRAIN_PCT          = 0.70
DEFAULT_VAL_PCT            = 0.15


# "same_label", "subtle"
VARIANT = "subtle"

# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    # Sweep axes
    datasets:              list[str]   = field(default_factory=lambda: DEFAULT_DATASETS)
    watermark_percentages: list[float] = field(default_factory=lambda: DEFAULT_WATERMARK_PERCENTAGES)
    chain_extensions:      list[int]   = field(default_factory=lambda: DEFAULT_CHAIN_EXTENSIONS)

    # Training
    epochs:        int   = DEFAULT_EPOCHS
    batch_size:    int   = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    hidden_dim:    int   = DEFAULT_HIDDEN_DIM
    train_pct:     float = DEFAULT_TRAIN_PCT
    val_pct:       float = DEFAULT_VAL_PCT

    # Benchmark
    repeats:            int = DEFAULT_REPEATS
    verification_count: int = DEFAULT_VERIFICATION_COUNT

    # Watermark variant — one of: "subtle" | "same_label"
    variant: str = VARIANT

    # Output
    output_dir: str = "benchmark_results"

# ── Helpers ────────────────────────────────────────────────────────────────────

def _derive_seed(secret_key: str, dataset: str, repeat: int) -> int:
    raw = f"{secret_key}|{dataset}|{repeat}"
    return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Watermark injection ────────────────────────────────────────────────────────

def _select_and_inject(
    dataset,
    watermark_pct: float,
    target_chain_length: int,
    is_binary: bool,
    seed: int,
    variant: str,
    utils: UtilityFunctions,
    analyzer: GraphAnalyzer,
) -> tuple[list, list]:
    """
    Selects graphs and injects watermark chains according to the variant.

    "subtle"     uses graphs_to_watermark (any label).
    "same_label" uses graphs_to_watermark_same_label (anchor label).

    Returns:
        watermarked_graphs : injected graphs to be appended to training set
        unselected_graphs  : remaining graphs that form the base training set
    """
    if variant not in ("subtle", "same_label"):
        raise ValueError(f"Unknown variant '{variant}'. Expected: subtle | same_label")

    rng = random.Random(seed + 101)

    if variant == "same_label":
        _, anchor_idx = analyzer.get_longest_global_chain_length(dataset)
        selected, unselected = utils.graphs_to_watermark_same_label(
            dataset=list(dataset),
            graph_index=anchor_idx,
            rng=rng,
            percentage=watermark_pct,
        )
    else:
        selected, unselected = utils.graphs_to_watermark(
            dataset=list(dataset),
            rng=rng,
            percentage=watermark_pct,
        )

    rng_inject = random.Random(seed + 202)

    watermarked_graphs = []
    for graph in selected:
        wm = inject_chain(
            copy.deepcopy(graph),
            target_chain_length,
            is_binary,
            rng_inject,
            feature_mode="subtle",
        )
        watermarked_graphs.append(wm)

    return watermarked_graphs, list(unselected)


def _build_verification_graphs(
    dataset,
    target_chain_length: int,
    is_binary: bool,
    seed: int,
    count: int,
) -> list:
    rng = random.Random(seed + 303)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    graphs = []
    for idx in indices[: min(count, len(dataset))]:
        wm = inject_chain(
            copy.deepcopy(dataset[idx]),
            target_chain_length,
            is_binary,
            rng,
            feature_mode="subtle",
        )
        graphs.append(wm)
    return graphs


# ── Evaluation ─────────────────────────────────────────────────────────────────

def _collect_scores(model, graphs, batch_size: int) -> dict:
    from torch_geometric.loader import DataLoader
    import torch.nn.functional as F

    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    model.eval()

    confidences = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            confidences.extend(probs.max(dim=1).values.cpu().tolist())

    avg = lambda lst: sum(lst) / len(lst) if lst else 0.0
    return {"avg_confidence": avg(confidences)}


def _compare_models(benign_model, watermarked_model, suspect_model, verification_graphs, batch_size) -> dict:
    b = _collect_scores(benign_model,      verification_graphs, batch_size)
    w = _collect_scores(watermarked_model, verification_graphs, batch_size)
    s = _collect_scores(suspect_model,     verification_graphs, batch_size)

    return {
        "benign_avg_confidence":                b["avg_confidence"],
        "watermarked_avg_confidence":           w["avg_confidence"],
        "suspect_avg_confidence":               s["avg_confidence"],
        "suspect_minus_benign_confidence":      s["avg_confidence"] - b["avg_confidence"],
        "suspect_minus_watermarked_confidence": s["avg_confidence"] - w["avg_confidence"],
    }


# ── Single run ─────────────────────────────────────────────────────────────────

def run_single(
    dataset_name: str,
    repeat: int,
    wm_pct: float,
    chain_ext: int,
    cfg: BenchmarkConfig,
    key: str,
) -> dict:
    seed = _derive_seed(key, dataset_name, repeat)
    _set_seeds(seed)

    utils     = UtilityFunctions()
    analyzer  = GraphAnalyzer()
    evaluator = Evaluator()

    dataset             = utils.load_dataset(dataset_name)
    global_chain_len, _ = analyzer.get_longest_global_chain_length(dataset)
    is_binary           = utils.is_binary(dataset)
    target_len          = global_chain_len + chain_ext

    watermarked_graphs, unselected_graphs = _select_and_inject(
        dataset=dataset,
        watermark_pct=wm_pct,
        target_chain_length=target_len,
        is_binary=is_binary,
        seed=seed,
        variant=cfg.variant,
        utils=utils,
        analyzer=analyzer,
    )

    trainer_kwargs = dict(
        dataset_name=dataset_name,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        hidden_dim=cfg.hidden_dim,
        epochs=cfg.epochs,
        train_pct=cfg.train_pct,
        val_pct=cfg.val_pct,
    )

    # Benign model: trained on full dataset, no watermarks
    benign_trainer = Trainer(dataset=list(dataset), seed=seed + 1, **trainer_kwargs)
    benign_model   = benign_trainer.train(modeltype="benign")

    # Watermarked model: trained on unselected + injected graphs appended
    watermarked_trainer = Trainer(
        dataset=unselected_graphs,
        watermarked_graphs=watermarked_graphs,
        seed=seed + 2,
        **trainer_kwargs,
    )
    watermarked_model = watermarked_trainer.train(modeltype="watermarked")

    evaluator.model = benign_model
    benign_acc      = evaluator.evaluate(benign_trainer.test_loader)

    evaluator.model = watermarked_model
    watermarked_acc = evaluator.evaluate(watermarked_trainer.test_loader)

    verification_graphs = _build_verification_graphs(
        dataset=list(dataset),
        target_chain_length=target_len,
        is_binary=is_binary,
        seed=seed,
        count=cfg.verification_count,
    )

    # Self-check: watermarked model as suspect → upper-bound signal
    reference = _compare_models(
        benign_model, watermarked_model, copy.deepcopy(watermarked_model),
        verification_graphs, cfg.batch_size,
    )
    # Control: benign model as suspect → lower-bound signal
    control = _compare_models(
        benign_model, watermarked_model, copy.deepcopy(benign_model),
        verification_graphs, cfg.batch_size,
    )

    return {
        "dataset":              dataset_name,
        "variant":              cfg.variant,
        "repeat":               repeat,
        "seed":                 seed,
        "watermark_pct":        wm_pct,
        "chain_extension":      chain_ext,
        "global_chain_len":     global_chain_len,
        "target_chain_len":     target_len,
        "is_binary":            is_binary,
        "num_watermarked":      len(watermarked_graphs),
        "num_verification":     len(verification_graphs),
        "benign_test_acc":      round(benign_acc, 4),
        "watermarked_test_acc": round(watermarked_acc, 4),
        "accuracy_drop":        round(benign_acc - watermarked_acc, 4),
        "reference":            reference,   # suspect = watermarked model
        "control":              control,     # suspect = benign model
    }


# ── Output ─────────────────────────────────────────────────────────────────────

def _flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}_{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif not isinstance(v, list):
            out[key] = v
    return out


def _save(results: list[dict], cfg: BenchmarkConfig, dataset_name: str) -> tuple[Path, Path]:
    slug    = dataset_name.lower().replace("-", "_")
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir) / cfg.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{slug}_{ts}.json"
    csv_path  = out_dir / f"{slug}_{ts}.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    flat_rows = [_flatten(r) for r in results]
    if flat_rows:
        keys = list(dict.fromkeys(k for row in flat_rows for k in row))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(flat_rows)

    print(f"  JSON → {json_path}")
    print(f"  CSV  → {csv_path}")
    return json_path, csv_path


# ── Entry point ────────────────────────────────────────────────────────────────

def run_benchmark(cfg: BenchmarkConfig = None) -> dict:
    """
    Run the full sweep defined by cfg.
    Returns a dict keyed by dataset name with output file paths.
    """
    if cfg is None:
        cfg = BenchmarkConfig()

    key   = os.getenv("SECRET_KEY", "default_key")
    total = len(cfg.datasets) * cfg.repeats * len(cfg.watermark_percentages) * len(cfg.chain_extensions)

    print(f"\nBenchmark  [variant={cfg.variant}]  —  {total} runs across {len(cfg.datasets)} dataset(s)\n")

    outputs = {}

    for dataset_name in cfg.datasets:
        results, run_n = [], 0

        for repeat in range(cfg.repeats):
            for pct in cfg.watermark_percentages:
                for ext in cfg.chain_extensions:
                    run_n += 1
                    label = f"[{run_n}/{total}] {dataset_name}  pct={pct:.0%}  ext=+{ext}  rep={repeat}"
                    print(label, end=" ... ", flush=True)
                    try:
                        result = run_single(dataset_name, repeat, pct, ext, cfg, key)
                        results.append(result)
                        ref = result["reference"]
                        print(
                            f"acc_b={result['benign_test_acc']:.3f}"
                            f"  acc_wm={result['watermarked_test_acc']:.3f}"
                            f"  Δconf={ref['suspect_minus_benign_confidence']:+.4f}"
                        )
                    except Exception as e:
                        print(f"FAILED — {e}")
                        traceback.print_exc()

        json_path, csv_path = _save(results, cfg, dataset_name)
        outputs[dataset_name] = {"results": len(results), "json": str(json_path), "csv": str(csv_path)}

    print("\nDone.")
    return outputs


if __name__ == "__main__":
    cfg = BenchmarkConfig(
        datasets=DEFAULT_DATASETS,
        variant=VARIANT,
        repeats=DEFAULT_REPEATS,
    )
    run_benchmark(cfg)