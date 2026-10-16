#!/usr/bin/env python3
"""Reference-level metadata-status table for the 1,000-paper cohort (outputs/metadata_status_1k.csv)."""

from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parents[1]
OUTPUT = HERE / "outputs" / "metadata_status_1k.csv"
EXPECTED_ROWS = {"low": 20_000, "high": 19_986}
STATUS_ORDER = ["metadata_correct", "metadata_minor_errors", "metadata_major_errors", "metadata_insufficient"]


def main():
    rows = []
    for effort, expected_n in EXPECTED_ROWS.items():
        data = pd.read_csv(HERE / "outputs" / effort / "resolved" / "all" / "references.csv", low_memory=False)
        if len(data) != expected_n or data["focal_paper_id"].nunique() != 1_000:
            raise RuntimeError(f"{effort}: expected 1,000 focal papers and {expected_n:,} references")
        unexpected = set(data["metadata_status"]) - set(STATUS_ORDER)
        if unexpected:
            raise RuntimeError(f"{effort}: unexpected metadata statuses: {sorted(map(str, unexpected))}")
        counts = data["metadata_status"].value_counts()
        for status in STATUS_ORDER:
            count = int(counts.get(status, 0))
            rows.append({
                "effort": effort,
                "metadata_status": status,
                "reference_count": count,
                "percentage_of_references": 100 * count / len(data),
                "reference_denominator": len(data),
                "focal_paper_count": data["focal_paper_id"].nunique(),
            })
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT, index=False, float_format="%.4f")
    print(result.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
