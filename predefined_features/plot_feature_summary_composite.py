#!/usr/bin/env python3
"""Fig 3: predefined reasoning features under low and high effort (GPT-5.5 judge, 1,000 papers).

  (A) score distributions per feature
  (B) verified-reference rate at each score level (box width scales with sqrt(n); lines join the means)
  (C) % of traces with the feature present (score >= 3) by saved-summary length

Output: outputs/final_figures/frontiers/Fig3.{tif,pdf,png}; prints the score means and medians.
"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

from common.cohort_plot_data import load_cohort_frame
from common.frontiers_figures import EFFORT_COLORS, export as frontiers_export
from predefined_features.plot_feature_positive_counts_by_summary_tokens import load_condition
from predefined_features.plot_feature_presence_vs_length_overlay import presence_panel
from predefined_features.plot_verified_rate_distributions import load_condition as load_outcome_condition


LOW = EFFORT_COLORS["low"]
HIGH = EFFORT_COLORS["high"]
DARK = "#0F172A"
GRID = "#E2E8F0"
FEATURES = [
    ("epistemic_caution", "Epistemic caution"),
    ("temporal_constraint_awareness", "Temporal constraint\nawareness"),
    ("reference_verifiability_checking", "Reference verifiability\nchecking"),
    ("metadata_verification", "Metadata verification"),
]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "text.color": DARK,
    "axes.labelcolor": DARK,
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "axes.edgecolor": "#94A3B8",
    "axes.linewidth": 0.6,
})


def style(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.4, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)


def score_distribution(ax, feature, title, paired, show_ylabel=False):
    x = np.arange(1, 6)
    width = 0.36
    low_counts = paired[f"{feature}_low"].value_counts().reindex(x, fill_value=0) / len(paired) * 100
    high_counts = paired[f"{feature}_high"].value_counts().reindex(x, fill_value=0) / len(paired) * 100
    ax.bar(x - width / 2, low_counts, width, color=LOW, alpha=0.86, label="Low effort")
    ax.bar(x + width / 2, high_counts, width, color=HIGH, alpha=0.86, label="High effort")
    ax.set_title(title, fontsize=8, fontweight="bold", loc="center", pad=8, linespacing=1.1)
    low_values = paired[f"{feature}_low"].to_numpy(float)
    high_values = paired[f"{feature}_high"].to_numpy(float)
    print(f"{title.replace(chr(10), ' ')}: low mean {low_values.mean():.2f}, median "
          f"{np.median(low_values):.2f}; high mean {high_values.mean():.2f}, median "
          f"{np.median(high_values):.2f}")
    ax.set_xticks(x, ["1", "2", "3", "4", "5"], fontsize=6.5)
    ax.set_ylim(0, 103)
    ax.set_yticks(range(0, 101, 25))
    ax.set_ylabel("Papers (%)" if show_ylabel else "", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    style(ax)


def dose_response_panel(ax, feature, low, high, show_ylabel=False):
    """Verified-reference rate at each score level: faint variable-width boxes, bold mean lines."""
    groups = []
    for frame, color, offset in ((low, LOW, -0.19), (high, HIGH, 0.19)):
        for score, chunk in frame.groupby(feature)["verified_rate_pct"]:
            values = chunk.dropna().to_numpy()
            if len(values) >= 2:
                groups.append((score + offset, values, color, len(values)))
    widest = max(n for *_, n in groups)
    for position, values, color, n in groups:
        ax.boxplot([values], positions=[position], widths=0.33 * np.sqrt(n / widest), patch_artist=True,
                   showfliers=True, manage_ticks=False, zorder=3,
                   medianprops=dict(color="white", linewidth=0.8),
                   boxprops=dict(facecolor=color, edgecolor=color, alpha=0.28, linewidth=0.6),
                   whiskerprops=dict(color=color, linewidth=0.8, alpha=0.6),
                   capprops=dict(color=color, linewidth=0.8, alpha=0.6),
                   flierprops=dict(marker="o", markersize=1.1, markerfacecolor=color, markeredgecolor="none", alpha=0.16))
    for frame, color, offset, marker in ((low, LOW, -0.19, "o"), (high, HIGH, 0.19, "s")):
        means = frame.groupby(feature)["verified_rate_pct"].mean().dropna()
        ax.plot(means.index.to_numpy(float) + offset, means.to_numpy(float), color=color, linewidth=2.6,
                marker=marker, markersize=4.5, markerfacecolor="white", markeredgecolor=color, markeredgewidth=0.9,
                solid_capstyle="round", zorder=6)
    ax.set_xticks(range(1, 6), ["1", "2", "3", "4", "5"], fontsize=7)
    ax.set_xlim(0.55, 5.45)
    ax.set_ylim(-6, 105)
    ax.set_yticks(range(0, 101, 25))
    ax.set_ylabel("Verified references (%)" if show_ylabel else "", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    style(ax)


def main():
    paired = pd.merge(load_cohort_frame("low"), load_cohort_frame("high"), on="focal_paper_id",
                      suffixes=("_low", "_high"), validate="one_to_one")
    presence_low, presence_high = load_condition("low"), load_condition("high")
    outcome_low, outcome_high = load_outcome_condition("low"), load_outcome_condition("high")

    fig = plt.figure(figsize=(9.6, 9.4), dpi=600)  # wide enough for one-line feature names
    grid = fig.add_gridspec(3, 4, height_ratios=[1.1, 1, 1], hspace=0.30, wspace=0.07)
    top_axes, dose_axes, presence_axes = [], [], []
    for i, (feature, title) in enumerate(FEATURES):
        top_axes.append(fig.add_subplot(grid[0, i]))
        score_distribution(top_axes[-1], feature, title, paired, show_ylabel=i == 0)
        dose_axes.append(fig.add_subplot(grid[1, i]))
        dose_response_panel(dose_axes[-1], feature, outcome_low, outcome_high, show_ylabel=i == 0)
        ax = fig.add_subplot(grid[2, i])
        presence_axes.append(ax)
        presence_panel(ax, presence_low, presence_high, feature, "", "", show_ylabel=i == 0)
        for line in ax.lines:
            line.set_markersize(line.get_markersize() * 1.35)
        for collection in ax.collections:
            if isinstance(collection, LineCollection):
                collection.set_linewidth(1.1)
        # Fewer length ticks so neighbouring labels never touch at 8 pt.
        keep = [(tick, label.get_text()) for tick, label in zip(ax.get_xticks(), ax.get_xticklabels())
                if label.get_text() in {"0", "200", "400", "≥600"}]
        ax.set_xticks([tick for tick, _ in keep], [text for _, text in keep])

    fig.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=LOW, markersize=4, label="Low effort"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=HIGH, markersize=4, label="High effort"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.982), ncol=2, frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.9, bottom=0.05)

    for letter, axes, xlabel in (("(A)", top_axes, None), ("(B)", dose_axes, "Feature score"), ("(C)", presence_axes, None)):
        for col, ax in enumerate(axes):
            if col:
                ax.tick_params(axis="y", left=False, labelleft=False)
                ax.spines["left"].set_visible(False)
            if xlabel:
                ax.set_xlabel(xlabel, fontsize=8)
            ax.spines["bottom"].set_linewidth(1.0)
            ax.spines["bottom"].set_color(DARK)
        fig.text(0.004, axes[0].get_position().y1 + 0.022, letter, fontsize=10, fontweight=700,
                 ha="left", va="top", color=DARK)
        # One x-axis title per row, centred under all four columns.
        row_label = axes[0].get_xlabel()
        if row_label:
            for ax in axes:
                ax.set_xlabel("")
            axes[0].set_xlabel(row_label)
            first, last = axes[0].get_position(), axes[-1].get_position()
            axes[0].xaxis.label.set_x(((first.x0 + last.x1) / 2 - first.x0) / first.width)
    frontiers_export(fig, "Fig3")
    plt.close(fig)


if __name__ == "__main__":
    main()
