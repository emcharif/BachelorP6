import csv
import json
from datetime import datetime
from pathlib import Path


def make_csv_row(result: dict, parent_key: str = "", sep: str = "_") -> dict:
    """
    Flattens a nested dictionary into a single-level dictionary suitable for CSV writing.
    """
    row = {}

    for key, value in result.items():
        csv_key = f"{parent_key}{sep}{key}" if parent_key else str(key)

        if isinstance(value, dict):
            row.update(make_csv_row(value, parent_key=csv_key, sep=sep))
        elif isinstance(value, list):
            continue
        else:
            row[csv_key] = value

    return row


def save_results(
    all_results,
    dataset_name: str,
    output_dir: Path,
    filename_prefix: str,
) -> tuple[Path, Path]:
    """
    Saves the full results to a JSON file and a flattened summary to a CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = output_dir / f"{filename_prefix}_{dataset_name}_{timestamp}.json"
    csv_path = output_dir / f"{filename_prefix}_{dataset_name}_{timestamp}.csv"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    csv_rows = [make_csv_row(result) for result in all_results]

    if csv_rows:
        fieldnames = []
        seen = set()

        for row in csv_rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(csv_rows)

    print(f"\nFull results saved to {json_path}")
    print(f"CSV summary saved to {csv_path}")

    return json_path, csv_path