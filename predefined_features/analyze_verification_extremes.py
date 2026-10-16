#!/usr/bin/env python3
"""Supplementary Table S2: mean predefined-feature scores at the extremes of verification (GPT-5.5 judge).

Within each effort condition, the lowest group holds the papers at or below the 10th percentile of
the paper-level verified-reference rate and the highest group the papers at or above the 90th
percentile. The rate moves in steps of about 5 percentage points, so all papers tied at a threshold
are included and the group sizes differ from 100.
Writes .../by_judge/gpt_5_5/analysis_tables/gpt_5_5_verification_extremes_full_1k.csv.
"""

import pandas as pd

from common.cohort_plot_data import load_cohort_frame
from common.judge_config import FEATURES, JUDGE_ROOT
from common.outcome_data import load_paper_metrics


OUTPUT = JUDGE_ROOT / "gpt_5_5" / "analysis_tables" / "gpt_5_5_verification_extremes_full_1k.csv"


def main():
    rows = []
    for condition in ("low", "high"):
        frame = load_cohort_frame(condition).merge(load_paper_metrics(condition), on="focal_paper_id", validate="one_to_one")
        if len(frame) != 1000:
            raise ValueError(f"Expected 1,000 {condition}-effort papers with scores and outcomes")
        rate = 100 * frame["existence_rate"]
        lowest, highest = rate.quantile(0.10), rate.quantile(0.90)
        for group, threshold, members in (("lowest", lowest, rate <= lowest), ("highest", highest, rate >= highest)):
            rows.append({
                "condition": condition,
                "group": group,
                "threshold_verified_rate_pct": threshold,
                "papers": int(members.sum()),
                "mean_verified_rate_pct": rate[members].mean(),
                **frame.loc[members, list(FEATURES)].mean().to_dict(),
            })
    table = pd.DataFrame(rows)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False, float_format="%.4f")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.2f}"))
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
