from src.benchmark.base_benchmark.base_benchmark import run_benchmark

run_benchmark(
    dataset_names=["ENZYMES", "PROTEINS"],
    feature_mode="subtle",
    use_watermark_head=False,
    use_same_label=True,
    results_subdir="label_specific",
)