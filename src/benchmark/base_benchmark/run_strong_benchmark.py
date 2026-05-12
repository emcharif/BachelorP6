from src.benchmark.base_benchmark.base_benchmark import run_benchmark

run_benchmark(
    dataset_names=["ENZYMES", "PROTEINS"],
    feature_mode="ood",
    use_watermark_head=True,
    watermark_loss_weight=1.0,
    results_subdir="strengthened",
)