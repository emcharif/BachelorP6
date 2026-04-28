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
# Assumes this file lives in: src/benchmark/attack_benchmark/run_attack_benchmark.py
# ---------------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[2]          # .../src
PROJECT_ROOT = THIS_FILE.parents[3]      # project root
RESULTS_ROOT = SRC_ROOT / "benchmark" / "results" / "attack"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from src.utils import UtilityFunctions
from src.graph_analyzer import GraphAnalyzer
from src.GNN.Trainer import Trainer
from src.GNN.Evaluator import Evaluator
from src.inject_chain import inject_chain
from src.benchmark.attack_benchmark.model_attacks import model_attacks


def slugify_dataset_name(dataset_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", dataset_name).strip("_").lower()


def derive_seed(secret_key: str, dataset_name: str, repeat_idx: int) -> int:
    raw = f"{secret_key}|{dataset_name}|repeat={repeat_idx}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def set_global_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model_device(model):
    return next(model.parameters()).device


def get_logits(model_output):
    """
    Supports both:
      - old classifier: logits
      - watermark-head classifier: (class_logits, watermark_score)
    """
    if isinstance(model_output, tuple):
        return model_output[0]
    return model_output


def split_dataset(dataset, seed, train_pct=0.7, val_pct=0.15):
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
    watermark_pct,
    chain_length,
    is_binary,
    seed,
    utility_functions,
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
    chain_length,
    is_binary,
    seed,
    verification_count=20,
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


def evaluate_external_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    device = get_model_device(model)

    with torch.no_grad():
        for batch in loader:
            if hasattr(batch, "to"):
                batch = batch.to(device)

            logits = get_logits(model(batch))
            pred = logits.argmax(dim=1)

            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

    return correct / total if total > 0 else 0.0


def trim_test_results_for_csv(test_results):
    return {k: v for k, v in test_results.items() if not isinstance(v, list)}


def flatten_dict(d, parent_key="", sep="_"):
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


def summarize_signal(test_results):
    suspect_minus_benign = (
        test_results["suspect_avg_confidence"] - test_results["benign_avg_confidence"]
    )
    watermarked_minus_benign = (
        test_results["watermarked_avg_confidence"] - test_results["benign_avg_confidence"]
    )

    distance_match = (
        test_results["avg_distance_to_watermarked"] < test_results["avg_distance_to_benign"]
    )

    return {
        "suspect_minus_benign_confidence": suspect_minus_benign,
        "watermarked_minus_benign_confidence": watermarked_minus_benign,
        "detected_by_confidence": suspect_minus_benign > 0.0,
        "detected_by_distance": distance_match,
    }


def build_result_row(
    base_info,
    attack_family,
    attack_name,
    attack_params,
    suspect_test_acc,
    test_results,
    baseline_conf_gap,
):
    row = copy.deepcopy(base_info)
    row["attack_family"] = attack_family
    row["attack_name"] = attack_name
    row["suspect_test_acc"] = round(suspect_test_acc, 4)

    for key, value in attack_params.items():
        row[key] = value

    row.update(trim_test_results_for_csv(test_results))
    row.update(summarize_signal(test_results))

    current_gap = row["suspect_minus_benign_confidence"]
    row["baseline_confidence_gap"] = baseline_conf_gap
    row["gap_retention_ratio"] = (
        current_gap / baseline_conf_gap if abs(baseline_conf_gap) > 1e-12 else None
    )

    if attack_family != "reference":
        row["attack_success_by_confidence"] = not row["detected_by_confidence"]
        row["attack_success_by_distance"] = not row["detected_by_distance"]
    else:
        row["attack_success_by_confidence"] = None
        row["attack_success_by_distance"] = None

    return row


def save_results(all_results, dataset_name, output_dir=None):
    if output_dir is None:
        dataset_slug = slugify_dataset_name(dataset_name)
        output_dir = RESULTS_ROOT / dataset_slug
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_slug = slugify_dataset_name(dataset_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"attack_benchmark_{dataset_slug}_{timestamp}.json"
    csv_path = output_dir / f"attack_benchmark_{dataset_slug}_{timestamp}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    flat_rows = []
    for result in all_results:
        row = {}

        for key, value in result.items():
            if isinstance(value, dict):
                row.update(flatten_dict(value, parent_key=key))
            elif isinstance(value, list):
                continue
            else:
                row[key] = value

        if "watermark_pct" in row:
            row["pct_label"] = f"{int(round(row['watermark_pct'] * 100))}%"
        if "chain_extension" in row:
            row["chain_label"] = f"+{int(row['chain_extension'])}"

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

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")

    return json_path, csv_path


def run_attack_benchmark(
    dataset_name="PROTEINS",
    repeats=3,
    verification_count=20,
    watermark_pct=0.10,
    chain_extension=2,
    epochs=50,
    batch_size=64,
    learning_rate=0.001,
    hidden_dim=128,
):
    blind_pruning_rates = [0.1, 0.3, 0.5, 0.7]
    blind_finetune_epochs = [1, 5, 10, 20]
    blind_finetune_lrs = [1e-3, 1e-4]

    informed_pruning_rates = [0.1, 0.3, 0.5, 0.7]
    informed_pruning_clean_weights = [0.5, 1.0]

    informed_finetune_epochs = [1, 5, 10, 20]
    informed_finetune_lrs = [1e-4]
    informed_finetune_lambdas = [0.1, 0.5, 1.0]

    if chain_extension < 1:
        raise ValueError("chain_extension must be >= 1")

    load_dotenv()
    secret_key = os.getenv("SECRET_KEY")
    if secret_key is None:
        raise ValueError("SECRET_KEY not found in environment/.env")

    all_results = []

    print("\n" + "=" * 80)
    print("ATTACK BENCHMARK")
    print(f"Dataset: {dataset_name}")
    print(f"Repeats: {repeats}")
    print(f"Watermark pct: {watermark_pct}")
    print(f"Chain extension: +{chain_extension}")
    print("=" * 80)

    for repeat_idx in range(repeats):
        seed = derive_seed(secret_key, dataset_name, repeat_idx)
        set_global_seeds(seed)

        utility_functions = UtilityFunctions()
        graph_analyzer = GraphAnalyzer()
        evaluator = Evaluator()
        attacks = model_attacks(batch_size=batch_size)

        dataset = utility_functions.load_dataset(name=dataset_name)

        # IMPORTANT:
        # get_global_chain_length now returns (max_chain_length_plus_one, graph_index)
        global_chain_length, global_chain_graph_index = graph_analyzer.get_global_chain_length(dataset)
        shortest_chain_length, shortest_chain_graph_index = graph_analyzer.get_shortest_chain_length(dataset)

        is_binary = utility_functions.is_binary(dataset)

        # global_chain_length is already max dangling chain + 1.
        # Therefore:
        # chain_extension=1 -> global_chain_length
        # chain_extension=2 -> global_chain_length + 1
        # chain_extension=3 -> global_chain_length + 2
        injector_chain_length = global_chain_length + (chain_extension - 1)
        target_watermark_chain_length = injector_chain_length

        train_clean, val_clean, test_clean = split_dataset(
            dataset=dataset,
            seed=seed,
            train_pct=0.70,
            val_pct=0.15,
        )

        watermarked_train, watermarked_training_graphs, unselected_graphs = (
            build_watermarked_train_split(
                train_clean=train_clean,
                watermark_pct=watermark_pct,
                chain_length=injector_chain_length,
                is_binary=is_binary,
                seed=seed,
                utility_functions=utility_functions,
            )
        )

        watermark_present = evaluator.verify_watermark(
            original_dataset=train_clean,
            watermarked_graphs=watermarked_training_graphs,
            chain_length=injector_chain_length,
        )

        print(f"\n{'=' * 80}")
        print(
            f"Repeat {repeat_idx + 1}/{repeats} | seed={seed} | "
            f"base_chain={global_chain_length} | "
            f"base_graph_idx={global_chain_graph_index} | "
            f"shortest_chain={shortest_chain_length} | "
            f"target_chain={target_watermark_chain_length}"
        )
        print(f"{'=' * 80}")

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

        print(
            f"Benign acc: {benign_test_acc:.4f} | "
            f"Watermarked acc: {watermarked_test_acc:.4f} | "
            f"Drop: {benign_test_acc - watermarked_test_acc:.4f}"
        )

        verification_graphs = build_verification_graphs(
            test_clean=test_clean,
            chain_length=injector_chain_length,
            is_binary=is_binary,
            seed=seed,
            verification_count=verification_count,
        )

        base_info = {
            "dataset": dataset_name,
            "repeat_idx": repeat_idx,
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_dim": hidden_dim,
            "watermark_pct": watermark_pct,
            "chain_extension": chain_extension,
            "global_chain_length": global_chain_length,
            "global_chain_graph_index": global_chain_graph_index,
            "shortest_chain_length": shortest_chain_length,
            "shortest_chain_graph_index": shortest_chain_graph_index,
            "injector_chain_length": injector_chain_length,
            "target_watermark_chain_length": target_watermark_chain_length,
            "verification_count": verification_count,
            "num_verification_graphs": len(verification_graphs),
            "dataset_size": len(dataset),
            "train_size": len(train_clean),
            "val_size": len(val_clean),
            "test_size": len(test_clean),
            "num_watermarked_train_graphs": len(watermarked_training_graphs),
            "num_clean_train_graphs": len(unselected_graphs),
            "watermark_structurally_verified": watermark_present,
            "benign_test_acc": round(benign_test_acc, 4),
            "watermarked_test_acc": round(watermarked_test_acc, 4),
            "accuracy_drop": round(benign_test_acc - watermarked_test_acc, 4),
        }

        print("\n--- Reference: suspect = watermarked model ---")
        baseline_test = evaluator.test_models_with_watermark(
            benign_model=benign_model,
            watermarked_model=watermarked_model,
            suspect_model=copy.deepcopy(watermarked_model),
            watermarked_graphs=verification_graphs,
        )

        baseline_conf_gap = (
            baseline_test["suspect_avg_confidence"] - baseline_test["benign_avg_confidence"]
        )

        all_results.append(
            build_result_row(
                base_info=base_info,
                attack_family="reference",
                attack_name="baseline_watermarked",
                attack_params={},
                suspect_test_acc=watermarked_test_acc,
                test_results=baseline_test,
                baseline_conf_gap=baseline_conf_gap,
            )
        )

        print("\n--- Reference: suspect = benign model ---")
        benign_control_test = evaluator.test_models_with_watermark(
            benign_model=benign_model,
            watermarked_model=watermarked_model,
            suspect_model=copy.deepcopy(benign_model),
            watermarked_graphs=verification_graphs,
        )

        all_results.append(
            build_result_row(
                base_info=base_info,
                attack_family="reference",
                attack_name="benign_control",
                attack_params={},
                suspect_test_acc=benign_test_acc,
                test_results=benign_control_test,
                baseline_conf_gap=baseline_conf_gap,
            )
        )

        print("\n--- Blind pruning attacks ---")
        for pruning_rate in blind_pruning_rates:
            print(f"Blind pruning | rate={pruning_rate}")

            attacked_model = attacks.blind_pruning_attack(
                model=watermarked_model,
                pruning_rate=pruning_rate,
            )

            attacked_test_acc = evaluate_external_model(attacked_model, benign_trainer.test_loader)

            attack_test = evaluator.test_models_with_watermark(
                benign_model=benign_model,
                watermarked_model=watermarked_model,
                suspect_model=attacked_model,
                watermarked_graphs=verification_graphs,
            )

            all_results.append(
                build_result_row(
                    base_info=base_info,
                    attack_family="blind",
                    attack_name="blind_pruning",
                    attack_params={"pruning_rate": pruning_rate},
                    suspect_test_acc=attacked_test_acc,
                    test_results=attack_test,
                    baseline_conf_gap=baseline_conf_gap,
                )
            )

        print("\n--- Blind fine-tuning attacks ---")
        for ft_lr in blind_finetune_lrs:
            for ft_epochs in blind_finetune_epochs:
                print(f"Blind fine-tune | epochs={ft_epochs} | lr={ft_lr}")

                attacked_model = attacks.blind_fine_tune_attack(
                    model=watermarked_model,
                    attacker_dataset=copy.deepcopy(train_clean),
                    epochs=ft_epochs,
                    learning_rate=ft_lr,
                    enable_prints=False,
                )

                attacked_test_acc = evaluate_external_model(attacked_model, benign_trainer.test_loader)

                attack_test = evaluator.test_models_with_watermark(
                    benign_model=benign_model,
                    watermarked_model=watermarked_model,
                    suspect_model=attacked_model,
                    watermarked_graphs=verification_graphs,
                )

                all_results.append(
                    build_result_row(
                        base_info=base_info,
                        attack_family="blind",
                        attack_name="blind_finetune",
                        attack_params={
                            "finetune_epochs": ft_epochs,
                            "attack_learning_rate": ft_lr,
                        },
                        suspect_test_acc=attacked_test_acc,
                        test_results=attack_test,
                        baseline_conf_gap=baseline_conf_gap,
                    )
                )

        print("\n--- Informed pruning attacks ---")
        for pruning_rate in informed_pruning_rates:
            for clean_weight in informed_pruning_clean_weights:
                print(
                    f"Informed pruning | rate={pruning_rate} | "
                    f"clean_preservation_weight={clean_weight}"
                )

                attacked_model = attacks.informed_pruning_attack(
                    model=watermarked_model,
                    clean_dataset=copy.deepcopy(train_clean),
                    watermark_graphs=copy.deepcopy(watermarked_training_graphs),
                    pruning_rate=pruning_rate,
                    clean_preservation_weight=clean_weight,
                    max_importance_batches=5,
                )

                attacked_test_acc = evaluate_external_model(attacked_model, benign_trainer.test_loader)

                attack_test = evaluator.test_models_with_watermark(
                    benign_model=benign_model,
                    watermarked_model=watermarked_model,
                    suspect_model=attacked_model,
                    watermarked_graphs=verification_graphs,
                )

                all_results.append(
                    build_result_row(
                        base_info=base_info,
                        attack_family="informed",
                        attack_name="informed_pruning",
                        attack_params={
                            "pruning_rate": pruning_rate,
                            "clean_preservation_weight": clean_weight,
                        },
                        suspect_test_acc=attacked_test_acc,
                        test_results=attack_test,
                        baseline_conf_gap=baseline_conf_gap,
                    )
                )

        print("\n--- Informed fine-tuning attacks ---")
        for ft_lr in informed_finetune_lrs:
            for ft_epochs in informed_finetune_epochs:
                for lambda_adv in informed_finetune_lambdas:
                    print(
                        f"Informed fine-tune | epochs={ft_epochs} | lr={ft_lr} | "
                        f"lambda_adv={lambda_adv}"
                    )

                    attacked_model = attacks.informed_fine_tune_attack(
                        model=watermarked_model,
                        clean_dataset=copy.deepcopy(train_clean),
                        watermark_graphs=copy.deepcopy(watermarked_training_graphs),
                        epochs=ft_epochs,
                        learning_rate=ft_lr,
                        lambda_adv=lambda_adv,
                        enable_prints=False,
                    )

                    attacked_test_acc = evaluate_external_model(attacked_model, benign_trainer.test_loader)

                    attack_test = evaluator.test_models_with_watermark(
                        benign_model=benign_model,
                        watermarked_model=watermarked_model,
                        suspect_model=attacked_model,
                        watermarked_graphs=verification_graphs,
                    )

                    all_results.append(
                        build_result_row(
                            base_info=base_info,
                            attack_family="informed",
                            attack_name="informed_finetune",
                            attack_params={
                                "finetune_epochs": ft_epochs,
                                "attack_learning_rate": ft_lr,
                                "lambda_adv": lambda_adv,
                            },
                            suspect_test_acc=attacked_test_acc,
                            test_results=attack_test,
                            baseline_conf_gap=baseline_conf_gap,
                        )
                    )

    json_path, csv_path = save_results(all_results, dataset_name=dataset_name)

    print("\n" + "=" * 80)
    print("ATTACK BENCHMARK COMPLETE")
    print(f"Dataset:     {dataset_name}")
    print(f"Repeats:     {repeats}")
    print(f"Experiments: {len(all_results)}")
    print(f"JSON:        {json_path}")
    print(f"CSV:         {csv_path}")
    print("=" * 80)

    return all_results


if __name__ == "__main__":
    run_attack_benchmark(
        dataset_name="ENZYMES",
        repeats=5,
        verification_count=50,
        watermark_pct=0.10,
        chain_extension=2,
        epochs=50,
        batch_size=64,
        learning_rate=0.001,
        hidden_dim=128,
    )