#!/usr/bin/env python3
"""Paper-level verified-reference rates by predefined-feature score (GPT-5.5 judge).

load_condition() also supplies the verified rates used by Fig 3 (B). Writes per-score and
grouped (scores 1-2 vs 3-5) distribution figures to .../by_judge/gpt_5_5/figures/ and their
summary tables (with bootstrap 95% CIs of the mean) to .../by_judge/gpt_5_5/analysis_tables/.
"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common.cohort_plot_data import load_cohort_frame
from common.judge_config import JUDGE_ROOT, JUDGES, judge_figure_dir
from common.outcome_data import load_paper_metrics


JUDGE_ID = "gpt_5_5"
JUDGE_LABEL = JUDGES[JUDGE_ID]["label"]
OUT_DIR = judge_figure_dir(JUDGE_ID)
TABLE_DIR = JUDGE_ROOT / JUDGE_ID / "analysis_tables"

LOW_COLOR = "#E11D48"
HIGH_COLOR = "#2563EB"
NEUTRAL_COLOR = "#475569"
GRID_COLOR = "#E2E8F0"

FEATURES = [
    ("epistemic_caution", "Epistemic Caution"),
    ("reference_verifiability_checking", "Reference Verifiability"),
    ("metadata_verification", "Metadata Verification"),
    ("temporal_constraint_awareness", "Temporal Awareness"),
]


def load_condition(condition):
    """Feature scores and verified-reference rate (%) per cohort paper."""
    scores = load_cohort_frame(condition).drop(columns="sample_index")
    data = scores.merge(load_paper_metrics(condition), on="focal_paper_id", how="inner")
    data["condition"] = condition.title()
    data["verified_rate_pct"] = 100 * data["existence_rate"]
    return data


def bootstrap_mean_ci(values, seed=20260825):
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]))


def style_axis(ax):
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94A3B8")
    ax.spines["bottom"].set_color("#94A3B8")
    ax.set_axisbelow(True)


def draw_half_violin(ax, values, position, color, side, width=0.33):
    if len(values) < 2 or np.allclose(values, values[0]):
        return
    violin = ax.violinplot(values, positions=[position], widths=2 * width, showextrema=False)
    body = violin["bodies"][0]
    body.set_facecolor(color)
    body.set_edgecolor(color)
    body.set_alpha(0.23)
    vertices = body.get_paths()[0].vertices
    vertices[:, 0] = (
        np.minimum(vertices[:, 0], position)
        if side == "left"
        else np.maximum(vertices[:, 0], position)
    )


def plot_score_distributions(low, high):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300, sharex=True, sharey=True)
    rng = np.random.default_rng(42)

    for panel, (ax, (feature, label)) in enumerate(zip(axes.flat, FEATURES)):
        for score in range(1, 6):
            for condition_data, offset, side, color in (
                (low, -0.16, "left", LOW_COLOR),
                (high, 0.16, "right", HIGH_COLOR),
            ):
                values = condition_data.loc[
                    condition_data[feature].eq(score), "verified_rate_pct"
                ].dropna().to_numpy()
                if not len(values):
                    continue
                position = score + offset
                draw_half_violin(ax, values, position, color, side)
                jitter = rng.uniform(-0.055, 0.055, len(values))
                ax.scatter(position + jitter, values, s=13, color=color, alpha=0.42,
                           edgecolors="none", rasterized=True, zorder=3)
                q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
                ax.vlines(position, q1, q3, color=color, linewidth=3.2, zorder=4)
                ax.scatter(position, median, s=28, color="white", edgecolor=color,
                           linewidth=1.4, zorder=5)
                ax.text(position, -7, f"n={len(values)}", ha="center", va="top",
                        fontsize=7, color=color, rotation=90)

        ax.set_title(f"({chr(65 + panel)}) {label}", fontsize=12, fontweight="bold", loc="left")
        ax.set_xlim(0.55, 5.45)
        ax.set_ylim(-16, 105)
        ax.set_xticks(range(1, 6))
        ax.set_yticks(range(0, 101, 20))
        ax.set_xlabel("Reasoning-feature score (1–5)", fontweight="bold")
        ax.set_ylabel("Verified real paper rate per focal paper (%)", fontweight="bold")
        style_axis(ax)

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LOW_COLOR,
               markersize=7, label="Low effort"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=HIGH_COLOR,
               markersize=7, label="High effort"),
        Line2D([0], [0], marker="o", color=NEUTRAL_COLOR, markerfacecolor="white",
               markersize=6, label="Median; thick line = IQR"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(f"{JUDGE_LABEL}: Verified-rate distributions at each reasoning-feature score",
                 fontsize=15, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = OUT_DIR / f"{JUDGE_ID}_verified_rate_distributions_by_feature_score.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def grouped_feature_summary(low, high):
    rows = []
    for condition, data in (("Low", low), ("High", high)):
        for feature, label in FEATURES:
            for score_group, selector in (
                ("1–2: Low / absent", data[feature].le(2)),
                ("3–5: Feature present", data[feature].ge(3)),
            ):
                values = data.loc[selector, "verified_rate_pct"].dropna().to_numpy()
                ci_low, ci_high = bootstrap_mean_ci(values)
                rows.append({
                    "condition": condition,
                    "feature": feature,
                    "feature_label": label,
                    "score_group": score_group,
                    "n_focal_papers": len(values),
                    "mean_verified_rate_pct": np.mean(values),
                    "bootstrap_mean_ci_low_pct": ci_low,
                    "bootstrap_mean_ci_high_pct": ci_high,
                    "median_verified_rate_pct": np.median(values),
                    "q1_verified_rate_pct": np.quantile(values, 0.25),
                    "q3_verified_rate_pct": np.quantile(values, 0.75),
                    "sparse_n_lt_10": len(values) < 10,
                })
    return pd.DataFrame(rows)


def plot_grouped_score_distributions(low, high, summary):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), dpi=300, sharex=True, sharey=True)
    rng = np.random.default_rng(126)
    # The 3–5 group carries the wider spread, so give it more horizontal room.
    groups = [
        ("1–2: Low / absent", lambda data, feature: data[feature].le(2), 1.0, 0.30),
        ("3–5: Feature present", lambda data, feature: data[feature].ge(3), 2.25, 0.50),
    ]

    for panel, (ax, (feature, label)) in enumerate(zip(axes.flat, FEATURES)):
        for group_label, selector, center, half_width in groups:
            offset_mag = half_width * 0.47
            for condition, data, sign, side, color in (
                ("Low", low, -1.0, "left", LOW_COLOR),
                ("High", high, 1.0, "right", HIGH_COLOR),
            ):
                values = data.loc[selector(data, feature), "verified_rate_pct"].dropna().to_numpy()
                position = center + sign * offset_mag
                draw_half_violin(ax, values, position, color, side, width=half_width)
                ax.scatter(
                    position + rng.uniform(-0.06, 0.06, len(values)),
                    values,
                    s=14,
                    color=color,
                    alpha=0.38,
                    edgecolors="none",
                    rasterized=True,
                    zorder=3,
                )
                row = summary[
                    summary["condition"].eq(condition)
                    & summary["feature"].eq(feature)
                    & summary["score_group"].eq(group_label)
                ].iloc[0]
                ax.errorbar(
                    position,
                    row["mean_verified_rate_pct"],
                    yerr=[
                        [row["mean_verified_rate_pct"] - row["bootstrap_mean_ci_low_pct"]],
                        [row["bootstrap_mean_ci_high_pct"] - row["mean_verified_rate_pct"]],
                    ],
                    fmt="D",
                    markersize=5.5,
                    markerfacecolor="white",
                    markeredgecolor=color,
                    markeredgewidth=1.5,
                    ecolor=color,
                    elinewidth=2.2,
                    capsize=4,
                    zorder=5,
                )
                sparse = " ⚠" if row["sparse_n_lt_10"] else ""
                ax.text(position, -7, f"n={int(row['n_focal_papers'])}{sparse}",
                        ha="center", va="top", fontsize=8, color=color)

        ax.set_title(f"({chr(65 + panel)}) {label}", fontsize=12,
                     fontweight="bold", loc="left")
        ax.set_xlim(0.4, 3.1)
        ax.set_ylim(-14, 105)
        ax.set_xticks([1.0, 2.25], ["Scores 1–2\nLow / absent", "Scores 3–5\nFeature present"])
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylabel("Verified real paper rate per focal paper (%)", fontweight="bold")
        style_axis(ax)

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LOW_COLOR,
               markersize=7, label="Low effort"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=HIGH_COLOR,
               markersize=7, label="High effort"),
        Line2D([0], [0], marker="D", color=NEUTRAL_COLOR, markerfacecolor="white",
               markersize=6, label="Mean and bootstrap 95% CI"),
        Line2D([0], [0], color="none", label="⚠ n < 10: sparse; interpret descriptively"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(
        f"{JUDGE_LABEL}: Verified-rate distributions for absent/low versus present reasoning features",
        fontsize=15, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = OUT_DIR / f"{JUDGE_ID}_verified_rate_distributions_grouped_scores_1_2_vs_3_5.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    low = load_condition("low")
    high = load_condition("high")
    grouped_summary = grouped_feature_summary(low, high)

    summary_rows = []
    for condition, data in (("Low", low), ("High", high)):
        for feature, label in FEATURES:
            for score, group in data.groupby(feature):
                values = group["verified_rate_pct"].dropna()
                summary_rows.append({
                    "condition": condition,
                    "feature": feature,
                    "feature_label": label,
                    "score": int(score),
                    "n_focal_papers": len(values),
                    "mean_verified_rate_pct": values.mean(),
                    "median_verified_rate_pct": values.median(),
                    "q1_verified_rate_pct": values.quantile(0.25),
                    "q3_verified_rate_pct": values.quantile(0.75),
                })
    summary_path = TABLE_DIR / f"{JUDGE_ID}_verified_rate_by_feature_score_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    grouped_summary_path = TABLE_DIR / f"{JUDGE_ID}_verified_rate_grouped_scores_1_2_vs_3_5_summary.csv"
    grouped_summary.to_csv(grouped_summary_path, index=False)

    print(f"Saved: {plot_score_distributions(low, high)}")
    print(f"Saved: {plot_grouped_score_distributions(low, high, grouped_summary)}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {grouped_summary_path}")


if __name__ == "__main__":
    main()
