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

DEFAULT_DATASETS = ["PROTEINS", "ENZYMES"]
DEFAULT_WATERMARK_PERCENTAGES = [0.05, 0.10, 0.20, 0.30]
DEFAULT_CHAIN_EXTENSIONS = [1, 3, 5, 10, 20]

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_HIDDEN_DIM = 128

DEFAULT_REPEATS = 10
DEFAULT_VERIFICATION_COUNT = 50
DEFAULT_TRAIN_PCT = 0.70
DEFAULT_VAL_PCT = 0.15

# Change manually between "subtle" and "same_label" for separate benchmark runs.
VARIANT = "same_label"


@dataclass
class BenchmarkConfig:
    datasets: list[str] = field(default_factory=lambda: DEFAULT_DATASETS)
    watermark_percentages: list[float] = field(default_factory=lambda: DEFAULT_WATERMARK_PERCENTAGES)
    chain_extensions: list[int] = field(default_factory=lambda: DEFAULT_CHAIN_EXTENSIONS)

    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    hidden_dim: int = DEFAULT_HIDDEN_DIM

    repeats: int = DEFAULT_REPEATS
    verification_count: int = DEFAULT_VERIFICATION_COUNT
    train_pct: float = DEFAULT_TRAIN_PCT
    val_pct: float = DEFAULT_VAL_PCT

    variant: str = VARIANT
    feature_mode: str = "subtle"
    output_dir: str = "benchmark_results"


def derive_seed(secret_key: str, dataset_name: str, repeat: int) -> int:
    raw = f"{secret_key}|{dataset_name}|repeat={repeat}"
    return int(hashlib.sha256(raw.encode()).hexdigest()[:8], 16)


def set_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(dataset, seed: int, train_pct: float, val_pct: float):
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    train_end = int(train_pct * len(dataset))
    val_end = train_end + int(val_pct * len(dataset))

    train = [dataset[i] for i in indices[:train_end]]
    val = [dataset[i] for i in indices[train_end:val_end]]
    test = [dataset[i] for i in indices[val_end:]]

    return train, val, test


def build_watermarked_train(
    train_clean,
    watermark_pct: float,
    target_chain_length: int,
    is_binary: bool,
    seed: int,
    cfg: BenchmarkConfig,
    utils: UtilityFunctions,
    analyzer: GraphAnalyzer,
):
    rng_select = random.Random(seed + 101)

    if cfg.variant == "same_label":
        _, anchor_idx = analyzer.get_longest_global_chain_length(train_clean)
        selected, unselected = utils.graphs_to_watermark_same_label(
            dataset=list(train_clean),
            graph_index=anchor_idx,
            percentage=watermark_pct,
            rng=rng_select,
        )
    elif cfg.variant == "subtle":
        selected, unselected = utils.graphs_to_watermark(
            dataset=list(train_clean),
            percentage=watermark_pct,
            rng=rng_select,
        )
    else:
        raise ValueError(f"Unknown variant: {cfg.variant}")

    rng_inject = random.Random(seed + 202)
    watermarked = [
        inject_chain(
            graph,
            target_chain_length,
            is_binary,
            rng_inject,
            feature_mode=cfg.feature_mode,
        ).clone()
        for graph in selected
    ]

    clean = [graph.clone() for graph in unselected]
    return watermarked + clean, watermarked, clean


def select_verification_graphs(watermarked_graphs, seed: int, count: int):
    rng = random.Random(seed + 303)
    indices = list(range(len(watermarked_graphs)))
    rng.shuffle(indices)
    return [watermarked_graphs[i].clone() for i in indices[: min(count, len(indices))]]


def add_signal_metrics(result: dict) -> dict:
    result["watermarked_minus_benign_confidence"] = (
        result["watermarked_avg_confidence"] - result["benign_avg_confidence"]
    )
    result["suspect_minus_benign_confidence"] = (
        result["suspect_avg_confidence"] - result["benign_avg_confidence"]
    )
    result["suspect_minus_watermarked_confidence"] = (
        result["suspect_avg_confidence"] - result["watermarked_avg_confidence"]
    )
    result["suspect_signal_positive_vs_benign"] = (
        result["suspect_minus_benign_confidence"] > 0
    )
    return result


def train_model(
    train_dataset,
    val_dataset,
    test_dataset,
    seed: int,
    cfg: BenchmarkConfig,
    dataset_name: str,
    modeltype: str,
):
    trainer = Trainer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        dataset_name=dataset_name,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        hidden_dim=cfg.hidden_dim,
        epochs=cfg.epochs,
        seed=seed,
    )
    model = trainer.train(modeltype=modeltype)
    return trainer, model


def run_single(dataset_name: str, repeat: int, wm_pct: float, chain_ext: int, cfg: BenchmarkConfig, key: str):
    seed = derive_seed(key, dataset_name, repeat)
    set_seeds(seed)

    utils = UtilityFunctions()
    analyzer = GraphAnalyzer()
    evaluator = Evaluator()

    dataset = list(utils.load_dataset(dataset_name))
    global_chain_len, _ = analyzer.get_longest_global_chain_length(dataset)
    is_binary = utils.is_binary(dataset)
    target_len = global_chain_len + chain_ext

    train_clean, val_clean, test_clean = split_dataset(dataset, seed, cfg.train_pct, cfg.val_pct)

    watermarked_train, watermarked_graphs, clean_training_graphs = build_watermarked_train(
        train_clean=train_clean,
        watermark_pct=wm_pct,
        target_chain_length=target_len,
        is_binary=is_binary,
        seed=seed,
        cfg=cfg,
        utils=utils,
        analyzer=analyzer,
    )

    benign_trainer, benign_model = train_model(
        train_dataset=train_clean,
        val_dataset=val_clean,
        test_dataset=test_clean,
        seed=seed + 1,
        cfg=cfg,
        dataset_name=dataset_name,
        modeltype="benign",
    )

    watermarked_trainer, watermarked_model = train_model(
        train_dataset=watermarked_train,
        val_dataset=val_clean,
        test_dataset=test_clean,
        seed=seed + 2,
        cfg=cfg,
        dataset_name=dataset_name,
        modeltype="watermarked",
    )

    evaluator.model = benign_model
    benign_acc = evaluator.evaluate(benign_trainer.test_loader)

    evaluator.model = watermarked_model
    watermarked_acc = evaluator.evaluate(watermarked_trainer.test_loader)

    verification_graphs = select_verification_graphs(
        watermarked_graphs=watermarked_graphs,
        seed=seed,
        count=cfg.verification_count,
    )

    reference = add_signal_metrics(evaluator.test_models_with_watermark(
        benign_model=benign_model,
        watermarked_model=watermarked_model,
        suspect_model=watermarked_model,
        watermarked_graphs=verification_graphs,
    ))

    control = add_signal_metrics(evaluator.test_models_with_watermark(
        benign_model=benign_model,
        watermarked_model=watermarked_model,
        suspect_model=benign_model,
        watermarked_graphs=verification_graphs,
    ))

    return {
        "dataset": dataset_name,
        "variant": cfg.variant,
        "feature_mode": cfg.feature_mode,
        "repeat": repeat,
        "seed": seed,
        "watermark_pct": wm_pct,
        "chain_extension": chain_ext,
        "global_chain_len": global_chain_len,
        "target_chain_len": target_len,
        "is_binary": is_binary,
        "train_size": len(train_clean),
        "val_size": len(val_clean),
        "test_size": len(test_clean),
        "num_watermarked": len(watermarked_graphs),
        "num_clean_train": len(clean_training_graphs),
        "num_verification": len(verification_graphs),
        "benign_test_acc": round(benign_acc, 4),
        "watermarked_test_acc": round(watermarked_acc, 4),
        "accuracy_drop": round(benign_acc - watermarked_acc, 4),
        "reference": reference,
        "control": control,
    }


def flatten(row: dict, prefix: str = "") -> dict:
    flat = {}
    for key, value in row.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(flatten(value, name))
        elif not isinstance(value, list):
            flat[name] = value
    return flat


def save_results(results: list[dict], cfg: BenchmarkConfig, dataset_name: str):
    out_dir = Path(cfg.output_dir) / cfg.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    slug = dataset_name.lower().replace("-", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = out_dir / f"{slug}_{timestamp}.json"
    csv_path = out_dir / f"{slug}_{timestamp}.csv"

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    rows = [flatten(r) for r in results]
    if rows:
        fieldnames = list(dict.fromkeys(k for row in rows for k in row))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"  JSON → {json_path}")
    print(f"  CSV  → {csv_path}")

    return json_path, csv_path


def run_benchmark(cfg: BenchmarkConfig = None):
    cfg = cfg or BenchmarkConfig()

    key = os.getenv("SECRET_KEY")
    if key is None:
        raise ValueError("SECRET_KEY not found in environment/.env")

    total = len(cfg.datasets) * cfg.repeats * len(cfg.watermark_percentages) * len(cfg.chain_extensions)
    print(f"\nBenchmark [variant={cfg.variant}] — {total} runs\n")

    outputs = {}

    for dataset_name in cfg.datasets:
        results = []
        run_idx = 0

        for repeat in range(cfg.repeats):
            for pct in cfg.watermark_percentages:
                for ext in cfg.chain_extensions:
                    run_idx += 1
                    print(f"[{run_idx}/{total}] {dataset_name} pct={pct:.0%} ext=+{ext} rep={repeat}", end=" ... ")

                    try:
                        result = run_single(dataset_name, repeat, pct, ext, cfg, key)
                        results.append(result)

                        signal = result["reference"]["watermarked_minus_benign_confidence"]
                        print(
                            f"acc_b={result['benign_test_acc']:.3f} "
                            f"acc_wm={result['watermarked_test_acc']:.3f} "
                            f"Δconf={signal:+.4f}"
                        )

                    except Exception as e:
                        print(f"FAILED — {e}")
                        traceback.print_exc()

        json_path, csv_path = save_results(results, cfg, dataset_name)
        outputs[dataset_name] = {
            "results": len(results),
            "json": str(json_path),
            "csv": str(csv_path),
        }

    print("\nDone.")
    return outputs


if __name__ == "__main__":
    run_benchmark(BenchmarkConfig())
