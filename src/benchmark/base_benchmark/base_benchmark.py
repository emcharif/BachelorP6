import os
import traceback

from pathlib import Path
from dotenv import load_dotenv

from src.utils import UtilityFunctions
from src.graph_analyzer import GraphAnalyzer
from src.GNN.Trainer import Trainer
from src.benchmark.base_benchmark.result_saver import save_results
from src.benchmark.base_benchmark.benchmark_helpers import (
    derive_seed,
    set_global_seeds,
    split_dataset,
    build_watermarked_train_split,
    build_watermarked_train_split_same_label,
    select_verification_graphs_from_training,
    structurally_verify_watermark,
    test_models_on_verification_graphs,
    extract_signal_metrics,
)


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
    use_same_label: bool = False,
) -> dict:
    """
    Runs a single experiment for the chain watermark benchmark.
    """
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

    global_chain_length, _ = graph_analyzer.get_longest_global_chain_length(dataset)

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
    results["use_same_label"] = use_same_label

    print(
        f"Global chain length: {global_chain_length} | "
        f"Target watermark chain length: {target_chain_length} | "
        f"Binary: {is_binary} | Full={len(dataset)} | "
        f"Train={len(train_clean)} | Val={len(val_clean)} | Test={len(test_clean)}"
    )

    if use_same_label:
        watermarked_train, watermarked_training_graphs, clean_training_graphs = (
            build_watermarked_train_split_same_label(
                train_clean=train_clean,
                watermark_pct=watermark_pct,
                target_chain_length=target_chain_length,
                is_binary=is_binary,
                seed=seed,
                utility_functions=utility_functions,
                feature_mode=feature_mode,
                use_watermark_head=use_watermark_head,
                graph_analyzer=graph_analyzer,
            )
    )
    else:
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
        train_dataset=train_clean,
        val_dataset=val_clean,
        test_dataset=test_clean,
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
        train_dataset=watermarked_train,
        val_dataset=val_clean,
        test_dataset=test_clean,
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

    verification_graphs = select_verification_graphs_from_training(
        watermarked_training_graphs=watermarked_training_graphs,
        seed=seed,
        verification_count=verification_count,
    )

    results["num_verification_graphs"] = len(verification_graphs)

    print("\n--- Reference watermark test: suspect = watermarked model ---")
    reference_test = test_models_on_verification_graphs(
        benign_model=benign_model,
        watermarked_model=watermarked_model,
        suspect_model=watermarked_model,
        verification_graphs=verification_graphs,
        batch_size=batch_size,
    )
    results["reference_watermarked_test"] = reference_test
    results.update(extract_signal_metrics(reference_test, "reference"))

    print("\n--- Benign control test: suspect = benign model ---")
    benign_control_test = test_models_on_verification_graphs(
        benign_model=benign_model,
        watermarked_model=watermarked_model,
        suspect_model=benign_model,
        verification_graphs=verification_graphs,
        batch_size=batch_size,
    )
    results["benign_control_test"] = benign_control_test
    results.update(extract_signal_metrics(benign_control_test, "control"))

    return results

def run_benchmark(
    dataset_names=None,
    repeats: int = DEFAULT_REPEATS,
    verification_count: int = DEFAULT_VERIFICATION_COUNT,
    watermark_percentages=None,
    chain_extensions=None,
    feature_mode: str = "subtle",
    use_watermark_head: bool = False,
    watermark_loss_weight: float = 1.0,
    results_subdir: str = "subtle",
    use_same_label: bool = False,
) -> dict:
    """
    Runs the chain watermark benchmark across multiple datasets and configurations.
    """
    if dataset_names is None:
        dataset_names = DEFAULT_DATASETS

    # Allow passing a single dataset as a string
    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]

    if watermark_percentages is None:
        watermark_percentages = DEFAULT_WATERMARK_PERCENTAGES

    if chain_extensions is None:
        chain_extensions = DEFAULT_CHAIN_EXTENSIONS

    if use_watermark_head:
        variant_name = "strengthened"
    elif use_same_label:
        variant_name = "label_specific"
    else:
        variant_name = "subtle"

    outputs = {}

    for dataset_name in dataset_names:
        all_results = []

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
                            use_same_label=use_same_label,
                        )
                        all_results.append(result)

                    except Exception as e:
                        print(
                            f"ERROR dataset={dataset_name}, repeat={repeat_idx}, "
                            f"pct={pct}, chain=+{chain_extension}: {e}"
                        )
                        traceback.print_exc()

        output_dir = Path("src") / "benchmark" / "results" / results_subdir

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

        outputs[dataset_name] = {
            "num_results": len(all_results),
            "json_path": str(json_path),
            "csv_path": str(csv_path),
        }

    return outputs