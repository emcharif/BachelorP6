from src.benchmark.base_benchmark.base_benchmark import run_all_chain_benchmarks

if __name__ == "__main__":
    run_all_chain_benchmarks(
        repeats=5,
        verification_count=50,
        watermark_percentages=[0.05, 0.10, 0.20, 0.30],
        chain_extensions=[1, 2, 3],
        feature_mode="ood",
        use_watermark_head=True,
        watermark_loss_weight=1.0,
        results_subdir="strengthened",
    )