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

sys.path.insert(0, os.path.dirname(__file__))

from utils import UtilityFunctions
from graph_analyzer import GraphAnalyzer
from GNN.Trainer import Trainer
from GNN.Evaluator import Evaluator
from model_attacks import model_attacks
from inject_chain import inject_chain


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
        inject_chain(graph, chain_length, is_binary, rng_inject)
        for graph in selected_graphs
    ]

    watermarked_train = watermarked_graphs + [copy.deepcopy(g) for g in unselected_graphs]

    return watermarked_train, watermarked_graphs, selected_graphs, unselected_graphs


def build_verification_graphs(
    test_clean,
    chain_length,
    is_binary,
    seed,
    verification_count=50,
):
    rng = random.Random(seed + 303)
    verification_graphs = []

    max_count = min(verification_count, len(test_clean))
    for graph in test_clean[:max_count]:
        modified = inject_chain(graph, chain_length, is_binary, rng)
        verification_graphs.append(modified)

    return verification_graphs


def evaluate_external_model(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            pred = model(batch).argmax(dim=1)
            correct += (pred == batch.y).sum().item()
            total += batch.y.size(0)

    return correct / total if total > 0 else 0.0


def interpret_watermark_test(test_results):
    return test_results["avg_distance_to_watermarked"] < test_results["avg_distance_to_benign"]


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


def run_single_experiment(
    dataset_name,
    repeat_idx,
    epochs,
    batch_size,
    learning_rate,
    hidden_dim,
    watermark_pct=0.1,
    pruning_rates=None,
    finetune_epochs=10,
    verification_count=50,
    train_pct=0.70,
    val_pct=0.15,
    experiment=None,
    section=None,
):
    if pruning_rates is None:
        pruning_rates = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    load_dotenv()
    secret_key = os.getenv("SECRET_KEY")
    if secret_key is None:
        raise ValueError("SECRET_KEY not found in environment/.env")

    seed = derive_seed(secret_key, dataset_name, repeat_idx)
    set_global_seeds(seed)

    utility_functions = UtilityFunctions()
    graph_analyzer = GraphAnalyzer()
    evaluator = Evaluator()
    model_attacks_obj = model_attacks()

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
        "finetune_epochs": finetune_epochs,
        "verification_count": verification_count,
        "train_pct": train_pct,
        "val_pct": val_pct,
        "test_pct": round(1.0 - train_pct - val_pct, 2),
    }

    print(f"\n{'=' * 80}")
    print(
        f"Dataset={dataset_name} | section={section} | experiment={experiment} | "
        f"repeat={repeat_idx} | seed={seed}"
    )
    print(
        f"epochs={epochs} | lr={learning_rate} | bs={batch_size} | "
        f"hd={hidden_dim} | pct={watermark_pct}"
    )
    print(f"{'=' * 80}")

    # 1. Load full dataset
    dataset = utility_functions.load_dataset(name=dataset_name)
    global_chain_length = graph_analyzer.get_global_chain_length(dataset)
    is_binary = utility_functions.is_binary(dataset)

    results["global_chain_length"] = global_chain_length
    results["is_binary"] = is_binary
    results["dataset_size"] = len(dataset)

    # 2. Split once here at benchmark level
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
        f"Chain length: {global_chain_length} | Binary: {is_binary} | "
        f"Full={len(dataset)} | Train={len(train_clean)} | "
        f"Val={len(val_clean)} | Test={len(test_clean)}"
    )

    # 3. Watermark only the training split
    watermarked_train, watermarked_training_graphs, selected_graphs, unselected_graphs = (
        build_watermarked_train_split(
            train_clean=train_clean,
            watermark_pct=watermark_pct,
            chain_length=global_chain_length,
            is_binary=is_binary,
            seed=seed,
            utility_functions=utility_functions,
        )
    )

    results["num_watermarked_train_graphs"] = len(watermarked_training_graphs)
    results["num_clean_train_graphs"] = len(unselected_graphs)

    # 4. Structural verification on the watermarked training graphs
    watermark_present = evaluator.verify_watermark(
        original_dataset=train_clean,
        watermarked_graphs=watermarked_training_graphs,
        chain_length=global_chain_length,
    )
    results["watermark_structurally_verified"] = watermark_present

    # 5. Train benign model on clean train/val/test
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

    # 6. Train watermarked model on watermarked train, but same clean val/test
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

    # 7. Build verification graphs from unseen clean test graphs
    verification_graphs = build_verification_graphs(
        test_clean=test_clean,
        chain_length=global_chain_length,
        is_binary=is_binary,
        seed=seed,
        verification_count=verification_count,
    )

    results["num_verification_graphs"] = len(verification_graphs)

    # 8. Baseline: suspect = watermarked model
    print("\n--- Baseline watermark test (suspect = watermarked model) ---")
    baseline_test = evaluator.test_models_with_watermark(
        benign_model=benign_model,
        watermarked_model=watermarked_model,
        suspect_model=copy.deepcopy(watermarked_model),
        watermarked_graphs=verification_graphs,
    )
    results["baseline_test"] = baseline_test
    results["baseline_distance_match"] = interpret_watermark_test(baseline_test)

    # 9. Control: suspect = benign model
    print("\n--- Benign control test (suspect = benign model) ---")
    benign_control_test = evaluator.test_models_with_watermark(
        benign_model=benign_model,
        watermarked_model=watermarked_model,
        suspect_model=copy.deepcopy(benign_model),
        watermarked_graphs=verification_graphs,
    )
    results["benign_control_test"] = benign_control_test
    results["benign_control_distance_match"] = interpret_watermark_test(benign_control_test)

    # 10. Fine-tuning attack using clean train split only
    print(f"\n--- Fine-tuning attack ({finetune_epochs} epochs) ---")
    fine_tuned_model = model_attacks_obj.fine_tune_attack(
        model=copy.deepcopy(watermarked_model),
        attacker_dataset=copy.deepcopy(train_clean),
        epochs=finetune_epochs,
    )

    fine_tuned_test_acc = evaluate_external_model(fine_tuned_model, benign_trainer.test_loader)

    fine_tune_test = evaluator.test_models_with_watermark(
        benign_model=benign_model,
        watermarked_model=watermarked_model,
        suspect_model=fine_tuned_model,
        watermarked_graphs=verification_graphs,
    )

    results["finetune_attack_test_acc"] = round(fine_tuned_test_acc, 4)
    results["finetune_attack_test"] = fine_tune_test
    results["finetune_attack_distance_match"] = interpret_watermark_test(fine_tune_test)

    # 11. Pruning attacks
    print("\n--- Pruning attacks ---")
    pruning_results = {}

    for rate in pruning_rates:
        print(f"\nPruning rate {rate:.1f}")

        pruned_model = model_attacks_obj.pruning_attack(
            model=copy.deepcopy(watermarked_model),
            pruning_rate=rate,
        )

        pruned_test_acc = evaluate_external_model(pruned_model, benign_trainer.test_loader)

        prune_test = evaluator.test_models_with_watermark(
            benign_model=benign_model,
            watermarked_model=watermarked_model,
            suspect_model=pruned_model,
            watermarked_graphs=verification_graphs,
        )

        pruning_results[str(rate)] = {
            "test_acc": round(pruned_test_acc, 4),
            "distance_match": interpret_watermark_test(prune_test),
            "watermark_test": prune_test,
        }

    results["pruning_results"] = pruning_results

    return results


def save_results(all_results, dataset_name, output_dir="benchmark_results"):
    os.makedirs(output_dir, exist_ok=True)

    dataset_slug = slugify_dataset_name(dataset_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(output_dir, f"benchmark_{dataset_slug}_{timestamp}.json")
    csv_path = os.path.join(output_dir, f"benchmark_{dataset_slug}_{timestamp}.csv")

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nFull results saved to {json_path}")

    flat_rows = []

    for result in all_results:
        row = {}

        for key, value in result.items():
            if isinstance(value, dict):
                if key in {"baseline_test", "benign_control_test", "finetune_attack_test"}:
                    row.update(flatten_dict(trim_test_results_for_csv(value), parent_key=key))
                elif key == "pruning_results":
                    for rate, rate_data in value.items():
                        rate_prefix = f"pruning_{rate}"
                        test_data = rate_data.get("watermark_test", {})

                        rate_row = {
                            f"{rate_prefix}_test_acc": rate_data.get("test_acc"),
                            f"{rate_prefix}_distance_match": rate_data.get("distance_match"),
                        }
                        rate_row.update(
                            flatten_dict(
                                trim_test_results_for_csv(test_data),
                                parent_key=f"{rate_prefix}_watermark_test",
                            )
                        )
                        row.update(rate_row)
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

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(flat_rows)

    print(f"CSV summary saved to {csv_path}")

    return json_path, csv_path


def run_benchmark(
    dataset_name="ENZYMES",
    base_repeats=3,
    verification_count=50,
):
    all_results = []

    print(f"\nRunning stage 2 benchmark for dataset: {dataset_name}")

    # 1. Base configuration, repeated
    print("\n>>> SECTION 1: Base Configuration")
    for repeat_idx in range(base_repeats):
        try:
            result = run_single_experiment(
                dataset_name=dataset_name,
                repeat_idx=repeat_idx,
                epochs=50,
                batch_size=64,
                learning_rate=0.001,
                hidden_dim=128,
                watermark_pct=0.10,
                finetune_epochs=10,
                verification_count=verification_count,
                experiment="base",
                section="base_configuration",
            )
            all_results.append(result)
        except Exception as e:
            print(f"ERROR in base configuration repeat {repeat_idx}: {e}")

    # 2. Hyperparameter sensitivity
    print(f"\n>>> SECTION 2: Hyperparameter Sensitivity ({dataset_name})")

    for lr in [0.01, 0.001, 0.0001]:
        try:
            result = run_single_experiment(
                dataset_name=dataset_name,
                repeat_idx=0,
                epochs=50,
                batch_size=64,
                learning_rate=lr,
                hidden_dim=128,
                watermark_pct=0.10,
                finetune_epochs=10,
                verification_count=verification_count,
                experiment=f"lr={lr}",
                section="hyperparameter_learning_rate",
            )
            all_results.append(result)
        except Exception as e:
            print(f"ERROR lr={lr}: {e}")

    for bs in [32, 64, 128]:
        try:
            result = run_single_experiment(
                dataset_name=dataset_name,
                repeat_idx=0,
                epochs=50,
                batch_size=bs,
                learning_rate=0.001,
                hidden_dim=128,
                watermark_pct=0.10,
                finetune_epochs=10,
                verification_count=verification_count,
                experiment=f"bs={bs}",
                section="hyperparameter_batch_size",
            )
            all_results.append(result)
        except Exception as e:
            print(f"ERROR bs={bs}: {e}")

    for hd in [64, 128, 256]:
        try:
            result = run_single_experiment(
                dataset_name=dataset_name,
                repeat_idx=0,
                epochs=50,
                batch_size=64,
                learning_rate=0.001,
                hidden_dim=hd,
                watermark_pct=0.10,
                finetune_epochs=10,
                verification_count=verification_count,
                experiment=f"hd={hd}",
                section="hyperparameter_hidden_dim",
            )
            all_results.append(result)
        except Exception as e:
            print(f"ERROR hd={hd}: {e}")

    # 3. Watermark percentage sensitivity
    print(f"\n>>> SECTION 3: Watermark Percentage Sensitivity ({dataset_name})")
    for pct in [0.05, 0.10, 0.20, 0.30]:
        try:
            result = run_single_experiment(
                dataset_name=dataset_name,
                repeat_idx=0,
                epochs=50,
                batch_size=64,
                learning_rate=0.001,
                hidden_dim=128,
                watermark_pct=pct,
                finetune_epochs=10,
                verification_count=verification_count,
                experiment=f"pct={pct}",
                section="watermark_percentage",
            )
            all_results.append(result)
        except Exception as e:
            print(f"ERROR pct={pct}: {e}")

    # 4. Fine-tuning attack intensity
    print(f"\n>>> SECTION 4: Fine-Tuning Attack Intensity ({dataset_name})")
    for ft_epochs in [5, 10, 20, 50]:
        try:
            result = run_single_experiment(
                dataset_name=dataset_name,
                repeat_idx=0,
                epochs=50,
                batch_size=64,
                learning_rate=0.001,
                hidden_dim=128,
                watermark_pct=0.10,
                finetune_epochs=ft_epochs,
                verification_count=verification_count,
                experiment=f"finetune_epochs={ft_epochs}",
                section="attack_finetune_intensity",
            )
            all_results.append(result)
        except Exception as e:
            print(f"ERROR finetune_epochs={ft_epochs}: {e}")

    json_path, csv_path = save_results(all_results, dataset_name=dataset_name)

    print("\n" + "=" * 80)
    print("STAGE 2 BENCHMARK COMPLETE")
    print(f"Dataset:     {dataset_name}")
    print(f"Experiments: {len(all_results)}")
    print(f"JSON:        {json_path}")
    print(f"CSV:         {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark(dataset_name="ENZYMES", base_repeats=3, verification_count=50)