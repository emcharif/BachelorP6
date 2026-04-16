import sys
import os
import copy
import json
import csv
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from utils import UtilityFunctions
from graph_analyzer import GraphAnalyzer
from GNN.Trainer import Trainer
from model_attacks import model_attacks
from torch_geometric.data import Data
from inject_chain import inject_chain


def build_complete_dataset(dataset, watermarked_graphs, unselected_graphs):
    clean_unselected = [
        Data(x=g.x, edge_index=g.edge_index, edge_attr=g.edge_attr if g.edge_attr is not None else None, y=g.y)
        for g in unselected_graphs
    ]
    return watermarked_graphs + clean_unselected


def run_single_experiment(dataset_name, epochs, batch_size, learning_rate, hidden_dim, watermark_pct=0.1, pruning_rates=None, finetune_epochs=10):
    if pruning_rates is None:
        pruning_rates = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

    utilityFunctions = UtilityFunctions()
    graphAnalyzer = GraphAnalyzer()
    modelAttacks = model_attacks()

    results = {
        "dataset": dataset_name,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "hidden_dim": hidden_dim,
        "watermark_pct": watermark_pct,
    }

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} | epochs={epochs} lr={learning_rate} bs={batch_size} hd={hidden_dim}")
    print(f"{'='*60}")

    # Load dataset
    dataset = utilityFunctions.load_dataset(name=dataset_name)
    global_chain_length = graphAnalyzer.get_global_chain_length(dataset)
    is_binary = utilityFunctions.is_binary(dataset)

    results["global_chain_length"] = global_chain_length
    results["is_binary"] = is_binary
    results["dataset_size"] = len(dataset)

    print(f"Chain length: {global_chain_length} | Binary: {is_binary} | Size: {len(dataset)}")

    # Watermark injection
    selected_graphs, unselected_graphs = utilityFunctions.graphs_to_watermark(dataset=dataset, percentage=watermark_pct)
    watermarked_graphs = [inject_chain(g, global_chain_length, is_binary) for g in selected_graphs]
    complete_dataset = build_complete_dataset(dataset, watermarked_graphs, unselected_graphs)

    results["num_watermarked_graphs"] = len(watermarked_graphs)

    # Train models
    watermarked_trainer = Trainer(dataset=complete_dataset, batch_size=batch_size, learning_rate=learning_rate, hidden_dim=hidden_dim, epochs=epochs)
    watermarked_model = watermarked_trainer.train(modeltype="watermarked")

    benign_trainer = Trainer(dataset=dataset, batch_size=batch_size, learning_rate=learning_rate, hidden_dim=hidden_dim, epochs=epochs)
    benign_model = benign_trainer.train(modeltype="benign")

    benign_test_acc = benign_trainer.evaluate(benign_trainer.test_loader)
    watermarked_test_acc = watermarked_trainer.evaluate(watermarked_trainer.test_loader)

    results["benign_test_acc"] = round(benign_test_acc, 4)
    results["watermarked_test_acc"] = round(watermarked_test_acc, 4)
    results["accuracy_drop"] = round(benign_test_acc - watermarked_test_acc, 4)

    print(f"Benign acc: {benign_test_acc:.4f} | Watermarked acc: {watermarked_test_acc:.4f} | Drop: {results['accuracy_drop']:.4f}")

    # Baseline detection (suspect = watermarked model)
    baseline_detected = benign_trainer.is_model_trained_on_watermarked_dataset(
        benign_model=benign_model,
        watermarked_model=watermarked_model,
        suspect_model=copy.deepcopy(watermarked_model),
        watermarked_graphs=watermarked_graphs
    )
    results["baseline_detected"] = baseline_detected
    print(f"Baseline detection: {baseline_detected}")

    # False positive test (suspect = benign model, should be False)
    false_positive = benign_trainer.is_model_trained_on_watermarked_dataset(
        benign_model=benign_model,
        watermarked_model=watermarked_model,
        suspect_model=copy.deepcopy(benign_model),
        watermarked_graphs=watermarked_graphs
    )
    results["false_positive"] = false_positive
    print(f"False positive (benign as suspect): {false_positive}")

    # Fine-tuning attack
    print(f"Fine-tuning attack ({finetune_epochs} epochs)...")
    finetuned_model = modelAttacks.fine_tune_attack(
        model=copy.deepcopy(watermarked_model),
        attacker_dataset=utilityFunctions.load_dataset(name=dataset_name),
        epochs=finetune_epochs
    )
    finetune_detected = benign_trainer.is_model_trained_on_watermarked_dataset(
        benign_model=benign_model,
        watermarked_model=watermarked_model,
        suspect_model=finetuned_model,
        watermarked_graphs=watermarked_graphs
    )
    results["finetune_attack_detected"] = finetune_detected
    results["finetune_epochs"] = finetune_epochs
    print(f"Fine-tune attack detected: {finetune_detected}")

    # Pruning attack across rates
    pruning_results = {}
    for rate in pruning_rates:
        pruned_model = modelAttacks.pruning_attack(
            model=copy.deepcopy(watermarked_model),
            pruning_rate=rate
        )
        detected = benign_trainer.is_model_trained_on_watermarked_dataset(
            benign_model=benign_model,
            watermarked_model=watermarked_model,
            suspect_model=pruned_model,
            watermarked_graphs=watermarked_graphs
        )
        pruning_results[rate] = detected
        print(f"  Pruning rate {rate:.1f} → detected: {detected}")

    results["pruning_results"] = pruning_results

    return results


def save_results(all_results, output_dir="benchmark_results"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save full JSON
    json_path = os.path.join(output_dir, f"benchmark_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {json_path}")

    # Save flat CSV for easy analysis
    csv_path = os.path.join(output_dir, f"benchmark_{timestamp}.csv")
    flat_rows = []
    for r in all_results:
        row = {k: v for k, v in r.items() if k != "pruning_results"}
        # flatten pruning results
        for rate, detected in r.get("pruning_results", {}).items():
            row[f"pruning_{rate}"] = detected
        flat_rows.append(row)

    if flat_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=flat_rows[0].keys())
            writer.writeheader()
            writer.writerows(flat_rows)
    print(f"CSV summary saved to {csv_path}")

    return json_path, csv_path


def run_benchmark():
    all_results = []

    # ── 1. Dataset comparison ──────────────────────────────────────
    print("\n>>> SECTION 1: Dataset Comparison")
    for dataset_name in ["IMDB-BINARY", "PROTEINS", "ENZYMES"]:
        try:
            r = run_single_experiment(
                dataset_name=dataset_name,
                epochs=50,
                batch_size=64,
                learning_rate=0.001,
                hidden_dim=128,
            )
            all_results.append(r)
        except Exception as e:
            print(f"  ERROR on {dataset_name}: {e}")

    # ── 2. Hyperparameter sensitivity ─────────────────────────────
    print("\n>>> SECTION 2: Hyperparameter Sensitivity (IMDB-BINARY)")

    for lr in [0.01, 0.001, 0.0001]:
        try:
            r = run_single_experiment("IMDB-BINARY", epochs=50, batch_size=64, learning_rate=lr, hidden_dim=128)
            r["experiment"] = f"lr={lr}"
            all_results.append(r)
        except Exception as e:
            print(f"  ERROR lr={lr}: {e}")

    for bs in [32, 64, 128]:
        try:
            r = run_single_experiment("IMDB-BINARY", epochs=50, batch_size=bs, learning_rate=0.001, hidden_dim=128)
            r["experiment"] = f"bs={bs}"
            all_results.append(r)
        except Exception as e:
            print(f"  ERROR bs={bs}: {e}")

    for hd in [64, 128, 256]:
        try:
            r = run_single_experiment("IMDB-BINARY", epochs=50, batch_size=64, learning_rate=0.001, hidden_dim=hd)
            r["experiment"] = f"hd={hd}"
            all_results.append(r)
        except Exception as e:
            print(f"  ERROR hd={hd}: {e}")

    # ── 3. Watermark percentage sensitivity ───────────────────────
    print("\n>>> SECTION 3: Watermark Percentage Sensitivity (IMDB-BINARY)")
    for pct in [0.05, 0.10, 0.20, 0.30]:
        try:
            r = run_single_experiment("IMDB-BINARY", epochs=50, batch_size=64, learning_rate=0.001, hidden_dim=128, watermark_pct=pct)
            r["experiment"] = f"pct={pct}"
            all_results.append(r)
        except Exception as e:
            print(f"  ERROR pct={pct}: {e}")

    # ── 4. Fine-tuning attack intensity ───────────────────────────
    print("\n>>> SECTION 4: Fine-Tuning Attack Intensity (IMDB-BINARY)")
    for ft_epochs in [5, 10, 20, 50]:
        try:
            r = run_single_experiment("IMDB-BINARY", epochs=50, batch_size=64, learning_rate=0.001, hidden_dim=128, finetune_epochs=ft_epochs)
            r["experiment"] = f"finetune_epochs={ft_epochs}"
            all_results.append(r)
        except Exception as e:
            print(f"  ERROR ft_epochs={ft_epochs}: {e}")

    # ── Save everything ───────────────────────────────────────────
    json_path, csv_path = save_results(all_results)

    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print(f"  {len(all_results)} experiments run")
    print(f"  Results: {json_path}")
    print(f"  CSV:     {csv_path}")
    print("="*60)


if __name__ == "__main__":
    run_benchmark()