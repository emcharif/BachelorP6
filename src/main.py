import copy
import random
from pathlib import Path

from src.Utilities.utils import UtilityFunctions
from src.inject import inject_chain
from src.GNN.Trainer import Trainer


def watermark_dataset(
    selected_graphs: list,
    timesteps_per_graph: int = 1,
    perturb_features: bool = True,
    fixed_timestep: int | None = 0,
):
    """
    Watermark the selected graphs.

    Args:
        selected_graphs: graphs chosen for watermarking
        timesteps_per_graph: number of timesteps to inject into per graph
        perturb_features: whether copied node features should be perturbed
        fixed_timestep:
            - int  -> inject at that timestep for all graphs
            - None -> choose timestep(s) using seeded randomness

    Returns:
        list of watermarked graph copies
    """
    utils = UtilityFunctions()
    watermarked_graphs = []

    rng = random.Random(utils.secret_key)

    for graph in selected_graphs:
        modified_graph = copy.deepcopy(graph)
        num_timesteps = len(modified_graph["vehicle"].ptr) - 1

        # TODO:
        # Replace this with your final timestep strategy.
        # Right now:
        # - fixed_timestep = 0 means inject only at timestep 0
        # - fixed_timestep = None means choose random timestep(s)
        if fixed_timestep is not None:
            chosen_timesteps = [fixed_timestep]
        else:
            all_timesteps = list(range(num_timesteps))
            rng.shuffle(all_timesteps)
            chosen_timesteps = all_timesteps[:timesteps_per_graph]

        chosen_timesteps = sorted(chosen_timesteps)

        for timestep in chosen_timesteps:
            # TODO:
            # Decide whether chain_length should stay adaptive (None),
            # be fixed, or be key-dependent.
            modified_graph = inject_chain(
                modified_graph,
                timestep=timestep,
                chain_length=None,
                perturb_features=perturb_features,
            )

        watermarked_graphs.append(modified_graph)

    return watermarked_graphs


def prepare_datasets(
    data_path: Path,
    labels_csv: Path,
    watermark_percentage: float = 0.05,
):
    """
    Load dataset, attach labels, and choose which graphs to watermark.
    """
    utils = UtilityFunctions()

    dataset = utils.load_dataset(path_to_files=str(data_path) + "/")
    dataset = utils.attach_labels(dataset, labels_csv=str(labels_csv))

    selected_graphs, unselected_graphs = utils.graphs_to_watermark(
        dataset,
        percentage=watermark_percentage,
    )

    return dataset, selected_graphs, unselected_graphs


def train_clean_and_watermarked_models(
    clean_dataset: list,
    owned_dataset: list,
    batch_size: int = 16,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
    learning_rate: float = 0.001,
    hidden_dim: int = 64,
    epochs: int = 30,
):
    """
    Train one clean model and one watermarked/owned model.
    """
    print("\nTraining clean model...")
    clean_trainer = Trainer(
        dataset=copy.deepcopy(clean_dataset),
        batch_size=batch_size,
        train_pct=train_pct,
        val_pct=val_pct,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        epochs=epochs,
    )
    clean_trainer.train(enable_prints=True)
    clean_model = clean_trainer.get_model(modeltype="clean")

    print("\nTraining watermarked model...")
    watermarked_trainer = Trainer(
        dataset=copy.deepcopy(owned_dataset),
        batch_size=batch_size,
        train_pct=train_pct,
        val_pct=val_pct,
        learning_rate=learning_rate,
        hidden_dim=hidden_dim,
        epochs=epochs,
    )
    watermarked_trainer.train(enable_prints=True)
    watermarked_model = watermarked_trainer.get_model(modeltype="watermarked")

    return clean_trainer, clean_model, watermarked_trainer, watermarked_model


def run_placeholder_verification(clean_model, watermarked_model, probe_graphs, trainer):
    """
    Temporary ownership verification using prediction/confidence comparison.
    """
    print("\nRunning placeholder ownership verification...")

    clean_preds, clean_confs = trainer.get_predictions(clean_model, probe_graphs)
    wm_preds, wm_confs = trainer.get_predictions(watermarked_model, probe_graphs)

    print(f"Number of probe graphs: {len(probe_graphs)}")
    print("First 10 comparisons:")

    for i in range(min(10, len(probe_graphs))):
        print(
            f"[{i}] "
            f"clean_pred={clean_preds[i]}, clean_conf={clean_confs[i]:.4f} | "
            f"wm_pred={wm_preds[i]}, wm_conf={wm_confs[i]:.4f}"
        )

    # TODO:
    # Replace this with the actual ownership verification:
    # - compare confidence differences on triggered vs clean probe graphs
    # - compute mean delta confidence
    # - run statistical test
    # - repeat across runs/seeds


def main():
    # Resolve project root from this file:
    # src/main.py -> parent is src -> parent.parent is project root
    project_root = Path(__file__).resolve().parent.parent

    data_path = project_root / "data" / "training_dataset"
    labels_csv = project_root / "labels_composite_3class.csv"

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    watermark_percentage = 0.05
    timesteps_per_graph = 1
    fixed_timestep = 0
    perturb_features = True

    batch_size = 16
    train_pct = 0.70
    val_pct = 0.15
    learning_rate = 0.001
    hidden_dim = 64
    epochs = 30

    # ------------------------------------------------------------------
    # Sanity checks
    # ------------------------------------------------------------------
    if not data_path.exists():
        raise FileNotFoundError(f"Training dataset folder not found: {data_path}")

    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")

    print(f"Project root: {project_root}")
    print(f"Training data path: {data_path}")
    print(f"Labels path: {labels_csv}")

    # ------------------------------------------------------------------
    # 1. Load and prepare dataset
    # ------------------------------------------------------------------
    print("\nLoading and labeling dataset...")
    full_dataset, selected_graphs, unselected_graphs = prepare_datasets(
        data_path=data_path,
        labels_csv=labels_csv,
        watermark_percentage=watermark_percentage,
    )

    print(f"Total graphs loaded: {len(full_dataset)}")
    print(f"Selected for watermarking: {len(selected_graphs)}")
    print(f"Not selected: {len(unselected_graphs)}")

    # ------------------------------------------------------------------
    # 2. Watermark selected graphs
    # ------------------------------------------------------------------
    print("\nInjecting watermark into selected graphs...")
    watermarked_selected_graphs = watermark_dataset(
        selected_graphs=selected_graphs,
        timesteps_per_graph=timesteps_per_graph,
        perturb_features=perturb_features,
        fixed_timestep=fixed_timestep,
    )

    print(f"Watermarked graphs created: {len(watermarked_selected_graphs)}")

    # ------------------------------------------------------------------
    # 3. Build training datasets
    # ------------------------------------------------------------------
    clean_dataset_for_training = copy.deepcopy(full_dataset)
    owned_dataset_for_training = copy.deepcopy(unselected_graphs) + watermarked_selected_graphs

    print("\nDataset summary:")
    print(f"Clean dataset size: {len(clean_dataset_for_training)}")
    print(f"Owned dataset size: {len(owned_dataset_for_training)}")

    # TODO:
    # Right now Trainer shuffles and splits internally.
    # Later, move splitting outside Trainer so clean and owned models
    # use the exact same validation/test splits.

    # ------------------------------------------------------------------
    # 4. Train models
    # ------------------------------------------------------------------
    clean_trainer, clean_model, watermarked_trainer, watermarked_model = (
        train_clean_and_watermarked_models(
            clean_dataset=clean_dataset_for_training,
            owned_dataset=owned_dataset_for_training,
            batch_size=batch_size,
            train_pct=train_pct,
            val_pct=val_pct,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            epochs=epochs,
        )
    )

    # ------------------------------------------------------------------
    # 5. Probe set for verification
    # ------------------------------------------------------------------
    probe_graphs = watermarked_selected_graphs

    # TODO:
    # Make a proper verification set:
    # - clean probe graphs
    # - triggered probe graphs
    # - ideally paired before/after versions of the same graphs

    # ------------------------------------------------------------------
    # 6. Placeholder verification
    # ------------------------------------------------------------------
    run_placeholder_verification(
        clean_model=clean_model,
        watermarked_model=watermarked_model,
        probe_graphs=probe_graphs,
        trainer=watermarked_trainer,
    )

    print("\nPipeline completed.")


if __name__ == "__main__":
    main()