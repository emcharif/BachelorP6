from pathlib import Path
from base_benchmark import run_all_chain_benchmarks

import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

if __name__ == "__main__":
    run_all_chain_benchmarks(
        repeats=5,
        verification_count=50,
        feature_mode="subtle",
        use_watermark_head=False,
        results_subdir="subtle",
    )