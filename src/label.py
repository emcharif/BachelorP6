# label_boston_split.py
# Creates graph-level labels for each .pt file using traffic density (avg #vehicles over timesteps)
# Output: labels_density_3class.csv  (path,label,avg_vehicles)

from __future__ import annotations

import os
import glob
import csv
from pathlib import Path
from typing import List, Tuple

import torch


DATA_DIR = Path("data/boston_split/graph_dataset")
OUTPUT_CSV = Path("labels_density_3class.csv")

# Choose how many classes you want (2 or 3 are easiest to start with)
N_CLASSES = 3  # set to 2 for binary (low/high)


def list_pt_files(data_dir: Path) -> List[Path]:
    files = sorted(Path(p) for p in glob.glob(str(data_dir / "*.pt")))
    if not files:
        raise FileNotFoundError(f"No .pt files found in: {data_dir.resolve()}")
    return files


def avg_vehicles_per_file(pt_path: Path) -> float:
    """
    Each .pt file appears to contain a temporal batch of graphs (timesteps),
    encoded via data['vehicle'].ptr (len = T+1).
    We compute the number of vehicles in each timestep and return the average.
    """
    data = torch.load(pt_path, weights_only=False)

    vehicle_store = data["vehicle"]
    if not hasattr(vehicle_store, "ptr"):
        # Fallback: single graph stored (no ptr)
        return float(vehicle_store.num_nodes)

    ptr = vehicle_store.ptr  # shape [T+1]
    # vehicles at timestep t = ptr[t+1] - ptr[t]
    counts = (ptr[1:] - ptr[:-1]).to(torch.float32)
    return float(counts.mean().item())


def make_thresholds(values: List[float], n_classes: int) -> List[float]:
    """
    Compute quantile thresholds so classes are roughly balanced.
    For 3 classes: thresholds at 33% and 66%.
    For 2 classes: threshold at 50%.
    Returns thresholds sorted ascending.
    """
    v = torch.tensor(values, dtype=torch.float32)
    if n_classes == 2:
        q = [0.5]
    elif n_classes == 3:
        q = [1 / 3, 2 / 3]
    else:
        raise ValueError("Only n_classes=2 or 3 supported for a clean start.")

    thresholds = [float(torch.quantile(v, torch.tensor(qq)).item()) for qq in q]
    return thresholds


def assign_class(value: float, thresholds: List[float]) -> int:
    """
    thresholds length:
      - 1 threshold => classes {0,1}
      - 2 thresholds => classes {0,1,2}
    """
    if len(thresholds) == 1:
        return 0 if value <= thresholds[0] else 1
    if len(thresholds) == 2:
        if value <= thresholds[0]:
            return 0
        elif value <= thresholds[1]:
            return 1
        else:
            return 2
    raise ValueError("Unexpected number of thresholds.")


def main():
    pt_files = list_pt_files(DATA_DIR)
    print(f"Found {len(pt_files)} .pt files in {DATA_DIR.resolve()}")

    # 1) Compute avg vehicles for each file
    rows: List[Tuple[str, float]] = []
    for i, pt in enumerate(pt_files, start=1):
        avg_v = avg_vehicles_per_file(pt)
        rows.append((str(pt.as_posix()), avg_v))
        if i % 50 == 0 or i == len(pt_files):
            print(f"Processed {i}/{len(pt_files)}")

    values = [v for _, v in rows]

    # 2) Compute balanced thresholds
    thresholds = make_thresholds(values, N_CLASSES)
    print("Thresholds:", thresholds)

    # 3) Assign labels
    labeled_rows: List[Tuple[str, int, float]] = []
    for path, avg_v in rows:
        label = assign_class(avg_v, thresholds)
        labeled_rows.append((path, label, avg_v))

    # 4) Save CSV
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "avg_vehicles"])
        writer.writerows(labeled_rows)

    print(f"Wrote: {OUTPUT_CSV.resolve()}")
    print("Label meaning (3-class): 0=low density, 1=medium density, 2=high density")
    print("Label meaning (2-class): 0=low, 1=high")


if __name__ == "__main__":
    main()