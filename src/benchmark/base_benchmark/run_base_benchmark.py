from pathlib import Path
import sys
import os
import copy
import json
import csv
import re
import random
import hashlib
from datetime import datetime

import torch
from dotenv import load_dotenv


# ---------------------------------------------------------------------
# Project paths
# Assumes this file lives in: Benchmark/BaseBenchmark/run_base_benchmark.py
# and the project source code lives in: src/
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
RESULTS_ROOT = PROJECT_ROOT / "Benchmark" / "results" / "base"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from utils import UtilityFunctions
from graph_analyzer import GraphAnalyzer
from GNN.Trainer import Trainer
from GNN.Evaluator import Evaluator
from inject_chain import inject_chain


# ---------------------------------------------------------------------
# Default benchmark configuration
# ---------------------------------------------------------------------
DEFAULT_DATASETS = ["ENZYMES", "PROTEINS"]
DEFAULT_WATERMARK_PERCENTAGES = [0.05, 0.10, 0.20, 0.30]
DEFAULT_CHAIN_EXTENSIONS = [1, 2, 3]

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_HIDDEN_DIM = 128

DEFAULT_REPEATS = 3
DEFAULT_VERIFICATION_COUNT = 20
DEFAULT_TRAIN_PCT = 0.70
DEFAULT_VAL_PCT = 0.15


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------
def slugify_dataset_name(dataset_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", dataset_name).strip("_").lower()


def derive_seed(secret_key: str, dataset_name: str, repeat_idx: int) -> int:
    raw = f"{secret_key}|{dataset_name}|repeat={repeat_idx}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_dataset(dataset, seed: int, train_pct: float = 0.70, val_pct: float = 0.15):
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    train_size = int(train_pct * len(dataset))
    val_size = int(val_pct * len(dataset))

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    train_split = [copy.deepcopy(dataset[i]) for i in train_idx]
    val_split = [copy.deepcopy(dataset[i]) for i in val_idx]
    test_split = [copy.deepcopy(dataset[i]) for i in test_idx]

    return train_split, val_split, test_split


def build_watermarked_train_split(
    train_clean,
    watermark_pct: float,
    chain_length: int,
    is_binary: bool,
    seed: int,
    utility_functions: UtilityFunctions,
):
    rng_select = random.Random(seed + 101)
    selected_graphs, unselected_graphs = utility_functions.graphs_to_watermark(
        dataset=train_clean,
        percentage=watermark_pct,
        rng=rng_select,
    )

    rng_inject = random.Random(seed + 202)
    watermarked_graphs = [
        inject_chain(copy.deepcopy(graph), chain_length, is_binary, rng_inject)
        for graph in selected_graphs
    ]

    watermarked_train = watermarked_graphs + [copy.deepcopy(g) for g in unselected_graphs]

    return watermarked_train, watermarked_graphs, unselected_graphs


def build_verification_graphs(
    test_clean,
    chain_length: int,
    is_binary: bool,
    seed: int,
    verification_count: int = 20,
):
    rng = random.Random(seed + 303)

    max_count = min(verification_count, len(test_clean))
    indices = list(range(len(test_clean)))
    rng.shuffle(indices)
    selected_indices = indices[:max_count]

    verification_graphs = []
    for idx in selected_indices:
        modified = inject_chain(copy.deepcopy(test_clean[idx]), chain_length, is_binary, rng)
        verification_graphs.append(modified)

    return verification_graphs


def trim_test_results_for_csv(test_results: dict) -> dict:
    return {k: v for k, v in test_results.items() if not isinstance(v, list)}


def flatten_dict(d, parent_key: str = "", sep: str = "_"):
    flat = {}

    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)

        if isinstance(value, dict):
            flat.update(flatten_dict(value, new_key, sep=sep))
        elif isinstance(value, list):
            continue
        else:
            flat[new_key] = value

    return flat


def extract_confidence_signal_metrics(test_results: dict, prefix: str) -> dict:
    benign_avg = test_results["benign_avg_confidence"]
    watermarked_avg = test_results["watermarked_avg_confidence"]
    suspect_avg = test_results["suspect_avg_confidence"]

    suspect_minus_benign = suspect_avg - benign_avg
    watermarked_minus_benign = watermarked_avg - benign_avg
    suspect_minus_watermarked = suspect_avg - watermarked_avg

    return {
        f"{prefix}_suspect_minus_benign_confidence": suspect_minus_benign,
        f"{prefix}_watermarked_minus_benign_confidence": watermarked_minus_benign,
        f"{prefix}_suspect_minus_watermarked_confidence": suspect_minus_watermarked,
        f"{prefix}_signal_positive_vs_benign": suspect_minus_benign > 0,
    }


# ---------------------------------------------------------------------
# Core single experiment
# ---------------------------------------------------------------------
def run_single_base_experiment(
    dataset_name: str,
    repeat_idx: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
    watermark_pct: float = 0.10,
    chain_extension: int = 1,
    verification_count: int = 20,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
    experiment: str | None = None,
    section: str | None = None,
):
    if chain_extension < 1:
        raise ValueError("chain_extension must be >= 1")

    load_dotenv()
    secret_key = os.getenv("SECRET_KEY")
    if secret_key is None:
        raise ValueError("SECRET_KEY not found in environment/.env")

    seed = derive_seed(secret_key, dataset_name, repeat_idx)
    set_global_seeds(seed)

    utility_functions = UtilityFunctions()
    graph_analyzer = GraphAnalyzer()
    evaluator = Evaluator()

    results = {
        "dataset": dataset_name,
        "section": section,
        "experiment": experiment,
        "repeat_idx": repeat_idx,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_dim": hidden_dim,
        "watermark_pct": watermark_pct,
        "chain_extension": chain_extension,
        "verification_count": verification_count,
        "train_pct": train_pct,
        "val_pct": val_pct,
        "test_pct": round(1.0 - train_pct - val_pct, 2),
    }

    print(f"\n{'=' * 88}")
    print(
        f"Dataset={dataset_name} | section={section} | experiment={experiment} | "
        f"repeat={repeat_idx} | seed={seed}"
    )
    print(
        f"epochs={epochs} | lr={learning_rate} | bs={batch_size} | "
        f"hd={hidden_dim} | pct={watermark_pct} | chain=+{chain_extension}"
    )
    print(f"{'=' * 88}")

    # 1. Load full dataset and compute dataset-level watermark target
    dataset = utility_functions.load_dataset(name=dataset_name)
    global_chain_length = graph_analyzer.get_global_chain_length(dataset)
    is_binary = utility_functions.is_binary(dataset)

    # Current inject_chain behavior:
    # inject_chain(..., global_chain_length) produces chain +1
    injector_chain_length = global_chain_length + (chain_extension - 1)
    target_watermark_chain_length = global_chain_length + chain_extension

    results["global_chain_length"] = global_chain_length
    results["injector_chain_length"] = injector_chain_length
    results["target_watermark_chain_length"] = target_watermark_chain_length
    results["is_binary"] = is_binary
    results["dataset_size"] = len(dataset)

    # 2. Controlled split for this repeat
    train_clean, val_clean, test_clean = split_dataset(
        dataset=dataset,
        seed=seed,
        train_pct=train_pct,
        val_pct=val_pct,
    )

    results["train_size"] = len(train_clean)
    results["val_size"] = len(val_clean)
    results["test_size"] = len(test_clean)

    print(
        f"Base chain length: {global_chain_length} | "
        f"Injector chain length: {injector_chain_length} | "
        f"Target watermark chain length: {target_watermark_chain_length} | "
        f"Binary: {is_binary} | Full={len(dataset)} | "
        f"Train={len(train_clean)} | Val={len(val_clean)} | Test={len(test_clean)}"
    )

    # 3. Watermark only the training split
    watermarked_train, watermarked_training_graphs, unselected_graphs = build_watermarked_train_split(
        train_clean=train_clean,
        watermark_pct=watermark_pct,
        chain_length=injector_chain_length,
        is_binary=is_binary,
        seed=seed,
        utility_functions=utility_functions,
    )

    results["num_watermarked_train_graphs"] = len(watermarked_training_graphs)
    results["num_clean_train_graphs"] = len(unselected_graphs)

    # 4. Structural verification on watermarked training graphs
    watermark_present = evaluator.verify_watermark(
        original_dataset=train_clean,
        watermarked_graphs=watermarked_training_graphs,
        chain_length=injector_chain_length,
    )
    results["watermark_structurally_verified"] = watermark_present

    # 5. Train benign model on clean data
    benign_trainer = Trainer(
        train_dataset=copy.deepcopy(train_clean),
        val_dataset=copy.deepcopy(val_clean),
        test_dataset=copy.deepcopy(test_clean),
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        epochs=epochs,
        seed=seed + 1,
    )
    benign_model = benign_trainer.train(modeltype="benign")

    # 6. Train watermarked model on watermarked training data
    watermarked_trainer = Trainer(
        train_dataset=copy.deepcopy(watermarked_train),
        val_dataset=copy.deepcopy(val_clean),
        test_dataset=copy.deepcopy(test_clean),
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        epochs=epochs,
        seed=seed + 2,
    )
    watermarked_model = watermarked_trainer.train(modeltype="watermarked")

    benign_test_acc = benign_trainer.evaluate(benign_trainer.test_loader)
    watermarked_test_acc = watermarked_trainer.evaluate(watermarked_trainer.test_loader)

    results["benign_test_acc"] = round(benign_test_acc, 4)
    results["watermarked_test_acc"] = round(watermarked_test_acc, 4)
    results["accuracy_drop"] = round(benign_test_acc - watermarked_test_acc, 4)

    print(
        f"Benign acc: {benign_test_acc:.4f} | "
        f"Watermarked acc: {watermarked_test_acc:.4f} | "
        f"Drop: {results['accuracy_drop']:.4f}"
    )

    # 7. Build unseen verification graphs from clean test graphs
    verification_graphs = build_verification_graphs(
        test_clean=test_clean,
        chain_length=injector_chain_length,
        is_binary=is_binary,
        seed=seed,
        verification_count=verification_count,
    )
    results["num_verification_graphs"] = len(verification_graphs)

    # 8. Reference signal test
    # suspect = exact copy of watermarked model
    # This is NOT ownership proof; it is the positive reference case.
    print("\n--- Reference watermark test (suspect = watermarked model) ---")
    reference_watermarked_test = evaluator.test_models_with_watermark(
        benign_model=benign_model,
        watermarked_model=watermarked_model,
        suspect_model=copy.deepcopy(watermarked_model),
        watermarked_graphs=verification_graphs,
    )
    results["reference_watermarked_test"] = reference_watermarked_test
    results.update(extract_confidence_signal_metrics(reference_watermarked_test, "reference"))

    # 9. Benign control
    # suspect = exact copy of benign model
    print("\n--- Benign control test (suspect = benign model) ---")
    benign_control_test = evaluator.test_models_with_watermark(
        benign_model=benign_model,
        watermarked_model=watermarked_model,
        suspect_model=copy.deepcopy(benign_model),
        watermarked_graphs=verification_graphs,
    )
    results["benign_control_test"] = benign_control_test
    results.update(extract_confidence_signal_metrics(benign_control_test, "control"))

    return results


# ---------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------
def save_results(all_results, dataset_name: str, output_dir: Path = RESULTS_ROOT):
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_slug = slugify_dataset_name(dataset_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"base_benchmark_{dataset_slug}_{timestamp}.json"
    csv_path = output_dir / f"base_benchmark_{dataset_slug}_{timestamp}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nFull results saved to {json_path}")

    flat_rows = []

    for result in all_results:
        row = {}

        for key, value in result.items():
            if isinstance(value, dict):
                if key in {"reference_watermarked_test", "benign_control_test"}:
                    row.update(flatten_dict(trim_test_results_for_csv(value), parent_key=key))
                else:
                    row.update(flatten_dict(value, parent_key=key))
            elif isinstance(value, list):
                continue
            else:
                row[key] = value

        flat_rows.append(row)

    if flat_rows:
        fieldnames = []
        seen = set()

        for row in flat_rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(flat_rows)

    print(f"CSV summary saved to {csv_path}")

    return json_path, csv_path


# ---------------------------------------------------------------------
# Dataset-level benchmark runner
# ---------------------------------------------------------------------
def run_base_benchmark(
    dataset_name: str = "PROTEINS",
    repeats: int = DEFAULT_REPEATS,
    verification_count: int = DEFAULT_VERIFICATION_COUNT,
    watermark_percentages=None,
    chain_extensions=None,
):
    if watermark_percentages is None:
        watermark_percentages = DEFAULT_WATERMARK_PERCENTAGES

    if chain_extensions is None:
        chain_extensions = DEFAULT_CHAIN_EXTENSIONS

    all_results = []

    print(f"\nRunning base watermark benchmark for dataset: {dataset_name}")

    for repeat_idx in range(repeats):
        print(f"\n>>> Repeat {repeat_idx + 1}/{repeats} for {dataset_name}")

        for pct in watermark_percentages:
            for chain_extension in chain_extensions:
                try:
                    result = run_single_base_experiment(
                        dataset_name=dataset_name,
                        repeat_idx=repeat_idx,
                        epochs=DEFAULT_EPOCHS,
                        batch_size=DEFAULT_BATCH_SIZE,
                        learning_rate=DEFAULT_LEARNING_RATE,
                        hidden_dim=DEFAULT_HIDDEN_DIM,
                        watermark_pct=pct,
                        chain_extension=chain_extension,
                        verification_count=verification_count,
                        train_pct=DEFAULT_TRAIN_PCT,
                        val_pct=DEFAULT_VAL_PCT,
                        experiment=f"pct={pct}_chain=+{chain_extension}",
                        section="watermark_pct_chain_extension",
                    )
                    all_results.append(result)
                except Exception as e:
                    print(
                        f"ERROR dataset={dataset_name}, repeat={repeat_idx}, "
                        f"pct={pct}, chain=+{chain_extension}: {e}"
                    )

    json_path, csv_path = save_results(all_results, dataset_name=dataset_name)

    print("\n" + "=" * 88)
    print("BASE WATERMARK BENCHMARK COMPLETE")
    print(f"Dataset:     {dataset_name}")
    print(f"Repeats:     {repeats}")
    print(f"Experiments: {len(all_results)}")
    print(f"JSON:        {json_path}")
    print(f"CSV:         {csv_path}")
    print("=" * 88)

    return all_results, json_path, csv_path


# ---------------------------------------------------------------------
# Optional runner for both datasets with the exact same protocol
# ---------------------------------------------------------------------
def run_all_base_benchmarks(
    dataset_names=None,
    repeats: int = DEFAULT_REPEATS,
    verification_count: int = DEFAULT_VERIFICATION_COUNT,
):
    if dataset_names is None:
        dataset_names = DEFAULT_DATASETS

    outputs = {}

    for dataset_name in dataset_names:
        results, json_path, csv_path = run_base_benchmark(
            dataset_name=dataset_name,
            repeats=repeats,
            verification_count=verification_count,
        )
        outputs[dataset_name] = {
            "num_results": len(results),
            "json_path": str(json_path),
            "csv_path": str(csv_path),
        }

    return outputs


if __name__ == "__main__":
    # Run one dataset:
    run_base_benchmark(dataset_name="PROTEINS", repeats=3, verification_count=20)