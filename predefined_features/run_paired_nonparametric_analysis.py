#!/usr/bin/env python3
"""Paired, nonparametric analyses of the predefined features (GPT-5.5 judge, 1,000 papers).

The focal paper is the unit of analysis. Writes four tables:
  1. paired low/high differences in the paper-level citation outcomes;
  2. paired changes in the four ordinal feature scores;
  3. Spearman feature-outcome associations within each effort condition;
  4. feature-presence sensitivity at score thresholds >= 3 and >= 4.
Intervals are percentile bootstraps over focal-paper pairs; p-values are Benjamini-Hochberg
adjusted within each stated family.
"""

import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

from common.cohort_plot_data import load_cohort_frame
from common.judge_config import HERE, JUDGE_ROOT
from common.outcome_data import load_paper_metrics


OUT_DIR = JUDGE_ROOT / "gpt_5_5" / "analysis_tables"
N_BOOT = 10_000
SEED = 20260917
TRUNCATED_FOCAL_PAPER_ID = "2885348054"  # high-effort response stopped by the content filter (6 references)

FEATURES = [
    ("epistemic_caution", "Epistemic caution"),
    ("temporal_constraint_awareness", "Temporal constraint awareness"),
    ("reference_verifiability_checking", "Reference verifiability checking"),
    ("metadata_verification", "Metadata verification"),
]
OUTCOMES = [
    ("verified_rate", "Verified-reference proportion"),
    ("temporal_validity_rate", "Temporally valid-reference proportion"),
]


def paired_bootstrap_mean_ci(differences, rng):
    """Percentile 95% CI of the mean, resampling complete focal-paper pairs."""
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values)]
    indices = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    return tuple(np.quantile(values[indices].mean(axis=1), [0.025, 0.975]))


def matched_pairs_rank_biserial(differences):
    """Rank-biserial correlation of paired differences; zero differences are omitted."""
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences) & (differences != 0)]
    if not len(differences):
        return 0.0
    ranks = rankdata(np.abs(differences), method="average")
    positive, negative = ranks[differences > 0].sum(), ranks[differences < 0].sum()
    return float((positive - negative) / (positive + negative))


def load_frames():
    frames = {}
    for condition in ("low", "high"):
        metrics = load_paper_metrics(condition).rename(columns={"existence_rate": "verified_rate"})
        frame = load_cohort_frame(condition).merge(metrics, on="focal_paper_id", validate="one_to_one")
        if len(frame) != 1000:
            raise ValueError(f"Expected 1,000 {condition}-effort papers with scores and outcomes")
        frames[condition] = frame
    return frames


def paired_frame(frames):
    keep = ["focal_paper_id", *[name for name, _ in FEATURES], *[name for name, _ in OUTCOMES]]
    return frames["low"][keep].merge(frames["high"][keep], on="focal_paper_id", suffixes=("_low", "_high"),
                                     validate="one_to_one")


def paired_test(difference, rng):
    ci_low, ci_high = paired_bootstrap_mean_ci(difference, rng)
    test = wilcoxon(difference, alternative="two-sided", zero_method="wilcox")
    return ci_low, ci_high, test.statistic, test.pvalue


def outcome_effects(paired):
    rng = np.random.default_rng(SEED)
    rows = []
    analysis_sets = (
        ("all_1000_observed_denominators", paired),
        ("complete_999_excluding_content_filter_truncation", paired[paired["focal_paper_id"] != TRUNCATED_FOCAL_PAPER_ID]),
    )
    for analysis_set, data in analysis_sets:
        family = []
        for outcome, label in OUTCOMES:
            low = data[f"{outcome}_low"].to_numpy(float)
            high = data[f"{outcome}_high"].to_numpy(float)
            difference = high - low
            ci_low, ci_high, statistic, p_value = paired_test(difference, rng)
            family.append({
                "analysis_set": analysis_set,
                "outcome": outcome,
                "outcome_label": label,
                "n_paired_focal_papers": len(difference),
                "mean_low": low.mean(),
                "mean_high": high.mean(),
                "mean_paired_difference_high_minus_low": difference.mean(),
                "bootstrap_95_ci_low": ci_low,
                "bootstrap_95_ci_high": ci_high,
                "bootstrap_resamples": N_BOOT,
                "bootstrap_seed": SEED,
                "wilcoxon_statistic": statistic,
                "wilcoxon_p": p_value,
                "rank_biserial_effect_size": matched_pairs_rank_biserial(difference),
            })
        q_values = multipletests([row["wilcoxon_p"] for row in family], method="fdr_bh")[1]
        rows += [{**row, "wilcoxon_q_bh": q} for row, q in zip(family, q_values)]
    return pd.DataFrame(rows)


def ordinal_feature_effects(paired):
    rng = np.random.default_rng(SEED + 1)
    rows = []
    for feature, label in FEATURES:
        low = paired[f"{feature}_low"].to_numpy(float)
        high = paired[f"{feature}_high"].to_numpy(float)
        difference = high - low
        ci_low, ci_high, statistic, p_value = paired_test(difference, rng)
        rows.append({
            "feature": feature,
            "feature_label": label,
            "n_paired_focal_papers": len(difference),
            "mean_low": low.mean(),
            "median_low": np.median(low),
            "mean_high": high.mean(),
            "median_high": np.median(high),
            "mean_paired_difference_high_minus_low": difference.mean(),
            "median_paired_difference_high_minus_low": np.median(difference),
            "bootstrap_95_ci_low_for_mean_difference": ci_low,
            "bootstrap_95_ci_high_for_mean_difference": ci_high,
            "wilcoxon_statistic": statistic,
            "wilcoxon_p": p_value,
            "rank_biserial_effect_size": matched_pairs_rank_biserial(difference),
        })
    result = pd.DataFrame(rows)
    result["wilcoxon_q_bh"] = multipletests(result["wilcoxon_p"], method="fdr_bh")[1]
    return result


def feature_outcome_associations(frames):
    rows = []
    for condition, frame in frames.items():
        for outcome, outcome_label in OUTCOMES:
            family = []
            for feature, feature_label in FEATURES:
                rho, p_value = spearmanr(frame[feature], frame[outcome], nan_policy="omit")
                family.append({
                    "condition": condition,
                    "feature": feature,
                    "feature_label": feature_label,
                    "outcome": outcome,
                    "outcome_label": outcome_label,
                    "n_focal_papers": int(frame[[feature, outcome]].dropna().shape[0]),
                    "spearman_rho": rho,
                    "spearman_p": p_value,
                })
            q_values = multipletests([row["spearman_p"] for row in family], method="fdr_bh")[1]
            rows += [{**row, "spearman_q_bh_within_condition_outcome_family": q} for row, q in zip(family, q_values)]
    return pd.DataFrame(rows)


def threshold_sensitivity(paired):
    rng = np.random.default_rng(SEED + 2)
    rows = []
    for threshold in (3, 4):
        for feature, label in FEATURES:
            low = paired[f"{feature}_low"].to_numpy(float) >= threshold
            high = paired[f"{feature}_high"].to_numpy(float) >= threshold
            difference = high.astype(float) - low.astype(float)
            ci_low, ci_high = paired_bootstrap_mean_ci(difference, rng)
            rows.append({
                "feature": feature,
                "feature_label": label,
                "presence_threshold": threshold,
                "threshold_interpretation": "implicit_or_stronger" if threshold == 3 else "explicit_or_stronger",
                "n_paired_focal_papers": len(difference),
                "prevalence_low": low.mean(),
                "prevalence_high": high.mean(),
                "paired_prevalence_difference_high_minus_low": difference.mean(),
                "bootstrap_95_ci_low": ci_low,
                "bootstrap_95_ci_high": ci_high,
                "low_absent_high_present": int((~low & high).sum()),
                "low_present_high_absent": int((low & ~high).sum()),
            })
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = load_frames()
    paired = paired_frame(frames)
    tables = {
        "paired_citation_outcomes": outcome_effects(paired),
        "paired_ordinal_feature_scores": ordinal_feature_effects(paired),
        "feature_outcome_spearman": feature_outcome_associations(frames),
        "feature_presence_threshold_sensitivity": threshold_sensitivity(paired),
    }
    paths = {}
    for name, table in tables.items():
        path = OUT_DIR / f"gpt_5_5_{name}_full_1k.csv"
        table.to_csv(path, index=False)
        paths[name] = str(path.relative_to(HERE))
        print(f"Saved: {path}")

    manifest = {
        "analysis": "full-cohort paired nonparametric predefined-feature analysis",
        "judge": "gpt_5_5",
        "n_paired_focal_papers": len(paired),
        "content_filter_sensitivity": {
            "focal_paper_id": TRUNCATED_FOCAL_PAPER_ID,
            "reported_analysis_sets": [
                "all 1,000 papers using each paper-condition's observed reference denominator",
                "999 complete pairs excluding the content-filter-truncated paper",
            ],
        },
        "confidence_level": 0.95,
        "bootstrap": {"unit": "paired focal paper", "resamples": N_BOOT, "seed": SEED, "interval": "percentile"},
        "multiple_comparison_correction": "Benjamini-Hochberg within stated hypothesis family",
        "outputs": paths,
    }
    manifest_path = OUT_DIR / "gpt_5_5_paired_nonparametric_analysis_full_1k_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {manifest_path}")


if __name__ == "__main__":
    main()
