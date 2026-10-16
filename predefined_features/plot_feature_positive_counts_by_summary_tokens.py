#!/usr/bin/env python3
"""Predefined-feature presence (score >= 3) by saved-summary length in tokens (GPT-5.5 judge).

load_condition() also supplies the per-paper summary token counts used by Fig 3.
Writes to feature_analysis/predefined_features/outputs/by_judge/gpt_5_5/figures/:
  gpt_5_5_feature_positive_counts_by_summary_tokens.{png,csv}   counts in 50 equal-width bins
  gpt_5_5_feature_prevalence_by_trace_length.png and ..._tokens.csv  % present in 50-token bins
"""

import matplotlib
import numpy as np
import pandas as pd
import tiktoken

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from common.cohort_plot_data import load_cohort_records
from common.judge_config import JUDGES, judge_figure_dir, reasoning_summary


JUDGE_ID = "gpt_5_5"
JUDGE_LABEL = JUDGES[JUDGE_ID]["label"]
OUT_DIR = judge_figure_dir(JUDGE_ID)
TOKENIZER = tiktoken.get_encoding("o200k_base")
FEATURES = [
    ("epistemic_caution", "Epistemic\ncaution"),
    ("temporal_constraint_awareness", "Temporal constraint\nawareness"),
    ("reference_verifiability_checking", "Reference verifiability\nchecking"),
    ("metadata_verification", "Metadata\nverification"),
]
COLORS = {"low": "#E11D48", "high": "#2563EB"}
N_BINS = 50
PREVALENCE_EDGES = np.array([-0.5, 0.5, *np.arange(50.5, 600.5 + 50, 50), np.inf])
PREVALENCE_LABELS = ["0"] + [f"{start}–{start + 49}" for start in range(1, 600, 50)] + ["601+"]


def load_condition(condition):
    """Feature scores and saved-summary length in o200k_base tokens, one row per cohort paper."""
    rows = []
    for record in load_cohort_records(condition):
        summary = reasoning_summary(condition, record["focal_paper_id"], record["prompt_id"])
        rows.append({
            "condition": condition,
            "focal_paper_id": str(record["focal_paper_id"]),
            "summary_tokens": len(TOKENIZER.encode(summary)) if summary else 0,
            **record["classification"],
        })
    return pd.DataFrame(rows)


def summarize(frame, condition, bin_edges):
    rows = []
    for feature, _ in FEATURES:
        all_counts, _ = np.histogram(frame["summary_tokens"], bins=bin_edges)
        positive_counts, _ = np.histogram(frame.loc[frame[feature] >= 3, "summary_tokens"].to_numpy(), bins=bin_edges)
        for bin_index, (total, present) in enumerate(zip(all_counts, positive_counts)):
            rows.append({
                "condition": condition,
                "feature": feature,
                "bin_index": bin_index,
                "bin_left_tokens": bin_edges[bin_index],
                "bin_right_tokens": bin_edges[bin_index + 1],
                "bin_center_tokens": (bin_edges[bin_index] + bin_edges[bin_index + 1]) / 2,
                "n_traces_in_bin": int(total),
                "n_feature_present": int(present),
                "prevalence_pct": 100 * present / total if total else np.nan,
            })
    return rows


def prevalence_table(raw):
    rows = []
    for condition, frame in raw.items():
        bins = pd.cut(frame["summary_tokens"], PREVALENCE_EDGES, labels=PREVALENCE_LABELS, right=True, include_lowest=True)
        for feature, _ in FEATURES:
            for length_bin in PREVALENCE_LABELS:
                group = frame.loc[bins == length_bin]
                rows.append({
                    "condition": condition,
                    "feature": feature,
                    "length_bin": length_bin,
                    "n_traces_in_bin": len(group),
                    "n_feature_present": int((group[feature] >= 3).sum()),
                    "prevalence_pct": 100 * (group[feature] >= 3).mean() if len(group) else np.nan,
                })
    return pd.DataFrame(rows)


def style_axis(ax):
    ax.grid(axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8.5)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = {condition: load_condition(condition) for condition in ("low", "high")}
    bin_edges = {c: np.linspace(0, frame["summary_tokens"].max() + 1, N_BINS + 1) for c, frame in raw.items()}
    table = pd.DataFrame([row for c in ("low", "high") for row in summarize(raw[c], c, bin_edges[c])])

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#94A3B8",
        "axes.linewidth": 0.9,
        "grid.color": "#E2E8F0",
        "grid.linewidth": 0.7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "text.color": "#0F172A",
        "axes.labelcolor": "#0F172A",
        "xtick.color": "#334155",
        "ytick.color": "#334155",
    })
    efforts = (("low", "Low effort"), ("high", "High effort"))

    fig, axes = plt.subplots(2, 4, figsize=(16, 8.0), sharey="row", dpi=300)
    for row_index, (condition, effort_label) in enumerate(efforts):
        edges = bin_edges[condition]
        for index, (feature, title) in enumerate(FEATURES):
            ax = axes[row_index, index]
            data = table.loc[(table["condition"] == condition) & (table["feature"] == feature)].sort_values("bin_index")
            counts = data["n_feature_present"].to_numpy()
            positions = data["bin_center_tokens"].to_numpy()
            ax.bar(positions, counts, width=(edges[1] - edges[0]) * 0.90, color=COLORS[condition],
                   alpha=0.28, edgecolor=COLORS[condition], linewidth=0.9, zorder=2)
            ax.plot(positions, counts, color=COLORS[condition], marker="o", linewidth=2.0, markersize=4.2, zorder=4)
            if row_index == 0:
                ax.set_title(title, fontsize=11, fontweight="bold", pad=9)
            ax.set_xlim(0, edges[-1])
            ax.set_xlabel("Saved-summary length (tokens)", fontsize=9)
            ax.set_ylim(bottom=0)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            style_axis(ax)
            if index == 0:
                ax.set_ylabel(f"{effort_label}\nNumber of traces with feature present (score ≥3)",
                              fontsize=9.5, fontweight="bold")
    fig.suptitle(f"{JUDGE_LABEL}: Feature-Positive Trace Counts by Reasoning-Summary Length",
                 fontsize=14, fontweight="bold", y=0.985)
    fig.text(0.5, 0.012,
             "Each row uses 50 equal-width bins over its own observed range: "
             f"low 0–{int(raw['low']['summary_tokens'].max())} tokens; "
             f"high 0–{int(raw['high']['summary_tokens'].max())} tokens. "
             "Each value counts traces with the named feature scored ≥3; tokens counted with o200k_base.",
             ha="center", va="bottom", fontsize=8.5, color="#475569")
    fig.tight_layout(rect=(0.025, 0.065, 1, 0.95), w_pad=1.2, h_pad=1.7)
    figure_path = OUT_DIR / f"{JUDGE_ID}_feature_positive_counts_by_summary_tokens.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    table.to_csv(figure_path.with_suffix(".csv"), index=False)
    print(f"Saved: {figure_path} and .csv")

    prevalence = prevalence_table(raw)
    fig, axes = plt.subplots(2, 4, figsize=(16, 9.0), sharey=True, dpi=300)
    x = np.arange(len(PREVALENCE_LABELS))
    for row_index, (condition, effort_label) in enumerate(efforts):
        for index, (feature, title) in enumerate(FEATURES):
            ax = axes[row_index, index]
            data = prevalence.loc[(prevalence["condition"] == condition) & (prevalence["feature"] == feature)]
            data = data.set_index("length_bin").reindex(PREVALENCE_LABELS)
            ax.bar(x, data["prevalence_pct"].to_numpy(), width=0.75, color=COLORS[condition], alpha=0.88,
                   edgecolor="none", label=effort_label, zorder=3)
            ax.set_title(title, fontsize=11, fontweight="bold", pad=9)
            ax.set_xticks(x)
            ax.set_xticklabels(PREVALENCE_LABELS, rotation=35, ha="right")
            ax.set_ylim(0, 105)
            ax.set_yticks(np.arange(0, 101, 20))
            ax.set_xlabel("Summary length bin (tokens)", fontsize=9)
            style_axis(ax)
            if index == 0:
                ax.set_ylabel(f"{effort_label}\nFeature present (score ≥3), %", fontsize=9.5, fontweight="bold")
                ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(f"{JUDGE_LABEL}: Predefined Feature Prevalence by Reasoning-Summary Length Bin "
                 f"(N={len(raw['low'])} per condition)", fontsize=14, fontweight="bold", y=0.985)
    fig.text(0.5, 0.012,
             "Both effort conditions use the same 50-token bins. The zero-token bin contains "
             "empty summaries; feature present = score ≥3; bins with no traces are left blank.",
             ha="center", va="bottom", fontsize=8.5, color="#475569")
    fig.tight_layout(rect=(0.025, 0.065, 1, 0.95), w_pad=1.2)
    prevalence_path = OUT_DIR / f"{JUDGE_ID}_feature_prevalence_by_trace_length.png"
    prevalence.to_csv(OUT_DIR / f"{JUDGE_ID}_feature_prevalence_by_trace_length_tokens.csv", index=False)
    fig.savefig(prevalence_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {prevalence_path}")


if __name__ == "__main__":
    main()
