from pathlib import Path
import sys
import os
import copy
import json
import csv
import re
import random
import hashlib
import traceback
from datetime import datetime

import torch
import torch.nn.functional as Function
from torch_geometric.loader import DataLoader
from dotenv import load_dotenv

CURRENT_FILE = Path(__file__).resolve()

SRC_ROOT = CURRENT_FILE.parents[2]
PROJECT_ROOT = SRC_ROOT.parent
RESULTS_ROOT = SRC_ROOT / "benchmark" / "results"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils import UtilityFunctions
from graph_analyzer import GraphAnalyzer
from GNN.Trainer import Trainer
from inject_chain import inject_chain


DEFAULT_DATASETS = ["ENZYMES", "PROTEINS"]
DEFAULT_WATERMARK_PERCENTAGES = [0.05, 0.10, 0.20, 0.30]
DEFAULT_CHAIN_EXTENSIONS = [1, 2, 3]

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 64
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_HIDDEN_DIM = 128

DEFAULT_REPEATS = 5
DEFAULT_VERIFICATION_COUNT = 50
DEFAULT_TRAIN_PCT = 0.70
DEFAULT_VAL_PCT = 0.15


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


def tag_graph(graph, value: float):
    graph.is_watermarked = torch.tensor([value], dtype=torch.float)
    return graph


def build_watermarked_train_split(
    train_clean,
    watermark_pct: float,
    target_chain_length: int,
    is_binary: bool,
    seed: int,
    utility_functions: UtilityFunctions,
    feature_mode: str,
    use_watermark_head: bool,
):
    rng_select = random.Random(seed + 101)
    selected_graphs, unselected_graphs = utility_functions.graphs_to_watermark(
        dataset=train_clean,
        percentage=watermark_pct,
        rng=rng_select,
    )

    rng_inject = random.Random(seed + 202)

    watermarked_graphs = []
    for graph in selected_graphs:
        wm_graph = inject_chain(
            copy.deepcopy(graph),
            target_chain_length,
            is_binary,
            rng_inject,
            feature_mode=feature_mode,
        )

        if use_watermark_head:
            wm_graph = tag_graph(wm_graph, 1.0)

        watermarked_graphs.append(wm_graph)

    clean_graphs = []
    for graph in unselected_graphs:
        clean_graph = copy.deepcopy(graph)

        if use_watermark_head:
            clean_graph = tag_graph(clean_graph, 0.0)

        clean_graphs.append(clean_graph)

    watermarked_train = watermarked_graphs + clean_graphs

    return watermarked_train, watermarked_graphs, clean_graphs


def build_verification_graphs(
    test_clean,
    target_chain_length: int,
    is_binary: bool,
    seed: int,
    verification_count: int,
    feature_mode: str,
):
    rng = random.Random(seed + 303)

    max_count = min(verification_count, len(test_clean))
    indices = list(range(len(test_clean)))
    rng.shuffle(indices)
    selected_indices = indices[:max_count]

    verification_graphs = []

    for idx in selected_indices:
        modified = inject_chain(
            copy.deepcopy(test_clean[idx]),
            target_chain_length,
            is_binary,
            rng,
            feature_mode=feature_mode,
        )
        verification_graphs.append(modified)

    return verification_graphs


def structurally_verify_watermark(watermarked_graphs, target_chain_length: int) -> bool:
    analyzer = GraphAnalyzer()
    verified = 0

    for graph in watermarked_graphs:
        _, chain_starts, neighbors = analyzer.search_graph(graph)

        lengths = []

        if len(chain_starts) != 0:
            for start in chain_starts:
                length, _ = analyzer.get_dangling_chain_length(start, neighbors)
                lengths.append(length)
        else:
            lengths.append(0)

        if max(lengths) >= target_chain_length:
            verified += 1

    ratio = verified / len(watermarked_graphs) if watermarked_graphs else 0
    print(
        f"Structural watermark verification: "
        f"{verified}/{len(watermarked_graphs)} graphs confirmed ({ratio:.0%})"
    )

    return ratio > 0.8


def collect_scores(model, graphs, batch_size: int):
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

    model.eval()

    confidences = []
    watermark_scores = []

    with torch.no_grad():
        for batch in loader:
            class_logits, wm_score = model(batch, return_watermark_score=True)

            probs = Function.softmax(class_logits, dim=1)
            conf = probs.max(dim=1).values

            confidences.extend(conf.cpu().tolist())
            watermark_scores.extend(wm_score.view(-1).cpu().tolist())

    return {
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 0.0,
        "confidences": confidences,
        "avg_watermark_score": sum(watermark_scores) / len(watermark_scores) if watermark_scores else 0.0,
        "watermark_scores": watermark_scores,
    }


def test_models_on_verification_graphs(
    benign_model,
    watermarked_model,
    suspect_model,
    verification_graphs,
    batch_size: int,
):
    benign = collect_scores(benign_model, verification_graphs, batch_size)
    watermarked = collect_scores(watermarked_model, verification_graphs, batch_size)
    suspect = collect_scores(suspect_model, verification_graphs, batch_size)

    return {
        "benign_avg_confidence": benign["avg_confidence"],
        "watermarked_avg_confidence": watermarked["avg_confidence"],
        "suspect_avg_confidence": suspect["avg_confidence"],

        "benign_confidences": benign["confidences"],
        "watermarked_confidences": watermarked["confidences"],
        "suspect_confidences": suspect["confidences"],

        "benign_avg_watermark_score": benign["avg_watermark_score"],
        "watermarked_avg_watermark_score": watermarked["avg_watermark_score"],
        "suspect_avg_watermark_score": suspect["avg_watermark_score"],

        "benign_watermark_scores": benign["watermark_scores"],
        "watermarked_watermark_scores": watermarked["watermark_scores"],
        "suspect_watermark_scores": suspect["watermark_scores"],
    }


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


def extract_signal_metrics(test_results: dict, prefix: str) -> dict:
    benign_conf = test_results["benign_avg_confidence"]
    watermarked_conf = test_results["watermarked_avg_confidence"]
    suspect_conf = test_results["suspect_avg_confidence"]

    benign_wm = test_results["benign_avg_watermark_score"]
    watermarked_wm = test_results["watermarked_avg_watermark_score"]
    suspect_wm = test_results["suspect_avg_watermark_score"]

    return {
        f"{prefix}_suspect_minus_benign_confidence": suspect_conf - benign_conf,
        f"{prefix}_watermarked_minus_benign_confidence": watermarked_conf - benign_conf,
        f"{prefix}_suspect_minus_watermarked_confidence": suspect_conf - watermarked_conf,
        f"{prefix}_confidence_signal_positive_vs_benign": suspect_conf - benign_conf > 0,

        f"{prefix}_suspect_minus_benign_watermark_score": suspect_wm - benign_wm,
        f"{prefix}_watermarked_minus_benign_watermark_score": watermarked_wm - benign_wm,
        f"{prefix}_suspect_minus_watermarked_watermark_score": suspect_wm - watermarked_wm,
        f"{prefix}_watermark_head_signal_positive_vs_benign": suspect_wm - benign_wm > 0,
    }


def run_single_chain_experiment(
    dataset_name: str,
    repeat_idx: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
    watermark_pct: float,
    chain_extension: int,
    verification_count: int,
    train_pct: float,
    val_pct: float,
    feature_mode: str,
    use_watermark_head: bool,
    watermark_loss_weight: float,
    experiment: str | None = None,
    section: str | None = None,
):
    if chain_extension < 1:
        raise ValueError("chain_extension must be >= 1")

    if feature_mode not in {"subtle", "ood"}:
        raise ValueError("feature_mode must be either 'subtle' or 'ood'")

    load_dotenv()
    secret_key = os.getenv("SECRET_KEY")

    if secret_key is None:
        raise ValueError("SECRET_KEY not found in environment/.env")

    seed = derive_seed(secret_key, dataset_name, repeat_idx)
    set_global_seeds(seed)

    utility_functions = UtilityFunctions()
    graph_analyzer = GraphAnalyzer()

    results = {
        "dataset": dataset_name,
        "section": section,
        "experiment": experiment,
        "variant": "strengthened" if use_watermark_head else "subtle",
        "feature_mode": feature_mode,
        "use_watermark_head": use_watermark_head,
        "watermark_loss_weight": watermark_loss_weight,
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

    print("\n" + "=" * 88)
    print(
        f"Dataset={dataset_name} | variant={results['variant']} | "
        f"feature_mode={feature_mode} | repeat={repeat_idx} | seed={seed}"
    )
    print(
        f"epochs={epochs} | lr={learning_rate} | bs={batch_size} | "
        f"hd={hidden_dim} | pct={watermark_pct} | chain=+{chain_extension}"
    )
    print("=" * 88)

    dataset = utility_functions.load_dataset(name=dataset_name)

    # FIX: get_global_chain_length returns a (length, graph_index) tuple — unpack it
    global_chain_length, _ = graph_analyzer.get_global_chain_length(dataset)

    is_binary = utility_functions.is_binary(dataset)

    target_chain_length = global_chain_length + chain_extension

    results["global_chain_length"] = global_chain_length
    results["target_watermark_chain_length"] = target_chain_length
    results["is_binary"] = is_binary
    results["dataset_size"] = len(dataset)

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
        f"Global chain length: {global_chain_length} | "
        f"Target watermark chain length: {target_chain_length} | "
        f"Binary: {is_binary} | Full={len(dataset)} | "
        f"Train={len(train_clean)} | Val={len(val_clean)} | Test={len(test_clean)}"
    )

    watermarked_train, watermarked_training_graphs, clean_training_graphs = (
        build_watermarked_train_split(
            train_clean=train_clean,
            watermark_pct=watermark_pct,
            target_chain_length=target_chain_length,
            is_binary=is_binary,
            seed=seed,
            utility_functions=utility_functions,
            feature_mode=feature_mode,
            use_watermark_head=use_watermark_head,
        )
    )

    results["num_watermarked_train_graphs"] = len(watermarked_training_graphs)
    results["num_clean_train_graphs"] = len(clean_training_graphs)

    watermark_present = structurally_verify_watermark(
        watermarked_graphs=watermarked_training_graphs,
        target_chain_length=target_chain_length,
    )
    results["watermark_structurally_verified"] = watermark_present

    benign_trainer = Trainer(
        train_dataset=copy.deepcopy(train_clean),
        val_dataset=copy.deepcopy(val_clean),
        test_dataset=copy.deepcopy(test_clean),
        dataset_name=dataset_name,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        epochs=epochs,
        seed=seed + 1,
        use_watermark_head=False,
    )
    benign_model = benign_trainer.train(modeltype="benign")

    watermarked_trainer = Trainer(
        train_dataset=copy.deepcopy(watermarked_train),
        val_dataset=copy.deepcopy(val_clean),
        test_dataset=copy.deepcopy(test_clean),
        dataset_name=dataset_name,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        epochs=epochs,
        seed=seed + 2,
        use_watermark_head=use_watermark_head,
        watermark_loss_weight=watermark_loss_weight,
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

    verification_graphs = build_verification_graphs(
        test_clean=test_clean,
        target_chain_length=target_chain_length,
        is_binary=is_binary,
        seed=seed,
        verification_count=verification_count,
        feature_mode=feature_mode,
    )

    results["num_verification_graphs"] = len(verification_graphs)

    print("\n--- Reference watermark test: suspect = watermarked model ---")
    reference_test = test_models_on_verification_graphs(
        benign_model=benign_model,
        watermarked_model=watermarked_model,
        suspect_model=copy.deepcopy(watermarked_model),
        verification_graphs=verification_graphs,
        batch_size=batch_size,
    )
    results["reference_watermarked_test"] = reference_test
    results.update(extract_signal_metrics(reference_test, "reference"))

    print("\n--- Benign control test: suspect = benign model ---")
    benign_control_test = test_models_on_verification_graphs(
        benign_model=benign_model,
        watermarked_model=watermarked_model,
        suspect_model=copy.deepcopy(benign_model),
        verification_graphs=verification_graphs,
        batch_size=batch_size,
    )
    results["benign_control_test"] = benign_control_test
    results.update(extract_signal_metrics(benign_control_test, "control"))

    return results


def save_results(
    all_results,
    dataset_name: str,
    output_dir: Path,
    filename_prefix: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_slug = slugify_dataset_name(dataset_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"{filename_prefix}_{dataset_slug}_{timestamp}.json"
    csv_path = output_dir / f"{filename_prefix}_{dataset_slug}_{timestamp}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

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

    print(f"\nFull results saved to {json_path}")
    print(f"CSV summary saved to {csv_path}")

    return json_path, csv_path


def run_benchmark(
    dataset_name: str = "PROTEINS",
    repeats: int = DEFAULT_REPEATS,
    verification_count: int = DEFAULT_VERIFICATION_COUNT,
    watermark_percentages=None,
    chain_extensions=None,
    feature_mode: str = "subtle",
    use_watermark_head: bool = False,
    watermark_loss_weight: float = 1.0,
    results_subdir: str = "subtle",
):
    if watermark_percentages is None:
        watermark_percentages = DEFAULT_WATERMARK_PERCENTAGES

    if chain_extensions is None:
        chain_extensions = DEFAULT_CHAIN_EXTENSIONS

    all_results = []

    variant_name = "strengthened" if use_watermark_head else "subtle"

    print(f"\nRunning {variant_name} chain benchmark for dataset: {dataset_name}")

    for repeat_idx in range(repeats):
        print(f"\n>>> Repeat {repeat_idx + 1}/{repeats} for {dataset_name}")

        for pct in watermark_percentages:
            for chain_extension in chain_extensions:
                try:
                    result = run_single_chain_experiment(
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
                        feature_mode=feature_mode,
                        use_watermark_head=use_watermark_head,
                        watermark_loss_weight=watermark_loss_weight,
                        experiment=f"pct={pct}_chain=+{chain_extension}",
                        section=f"{variant_name}_watermark_pct_chain_extension",
                    )
                    all_results.append(result)
                except Exception as e:
                    # FIX: print full traceback so errors are never silently swallowed
                    print(
                        f"ERROR dataset={dataset_name}, repeat={repeat_idx}, "
                        f"pct={pct}, chain=+{chain_extension}: {e}"
                    )
                    traceback.print_exc()

    output_dir = RESULTS_ROOT / results_subdir / slugify_dataset_name(dataset_name)

    json_path, csv_path = save_results(
        all_results=all_results,
        dataset_name=dataset_name,
        output_dir=output_dir,
        filename_prefix=f"{variant_name}_benchmark",
    )

    print("\n" + "=" * 88)
    print(f"{variant_name.upper()} CHAIN BENCHMARK COMPLETE")
    print(f"Dataset:     {dataset_name}")
    print(f"Repeats:     {repeats}")
    print(f"Experiments: {len(all_results)}")
    print(f"JSON:        {json_path}")
    print(f"CSV:         {csv_path}")
    print("=" * 88)

    return all_results, json_path, csv_path


def run_all_chain_benchmarks(
    dataset_names=None,
    repeats: int = DEFAULT_REPEATS,
    verification_count: int = DEFAULT_VERIFICATION_COUNT,
    **kwargs,
):
    if dataset_names is None:
        dataset_names = DEFAULT_DATASETS

    outputs = {}

    for dataset_name in dataset_names:
        results, json_path, csv_path = run_benchmark(
            dataset_name=dataset_name,
            repeats=repeats,
            verification_count=verification_count,
            **kwargs,
        )
        outputs[dataset_name] = {
            "num_results": len(results),
            "json_path": str(json_path),
            "csv_path": str(csv_path),
        }

    return outputs