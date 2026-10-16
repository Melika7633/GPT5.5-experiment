"""Paper-level citation outcomes from the resolved references."""

from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parents[1]


def load_paper_metrics(condition):
    """Per focal paper: share of references verified (existence_rate) and dated after the paper.

    Each focal paper counts once, whatever the number of references it generated;
    possible matches stay in the denominator only.
    """
    refs = pd.read_csv(HERE / "outputs" / condition / "resolved" / "all" / "references.csv",
                       usecols=["focal_paper_id", "existence_status", "temporal_status", "focal_year"])
    refs["does_exist"] = refs["existence_status"].eq("verified_real_paper")
    refs["is_future"] = refs["temporal_status"].eq("future_reference")
    metrics = refs.groupby("focal_paper_id", as_index=False).agg(
        existence_rate=("does_exist", "mean"),
        future_rate=("is_future", "mean"),
        focal_year=("focal_year", "first"),
    )
    metrics["temporal_validity_rate"] = 1 - metrics["future_rate"]
    metrics["focal_paper_id"] = metrics["focal_paper_id"].astype(str)
    return metrics
