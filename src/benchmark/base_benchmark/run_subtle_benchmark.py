
from src.benchmark.base_benchmark.base_benchmark import run_all_chain_benchmarks

if __name__ == "__main__":
    run_all_chain_benchmarks(
        repeats=5,
        verification_count=50,
        feature_mode="subtle",
        use_watermark_head=False,
        results_subdir="subtle",
    )