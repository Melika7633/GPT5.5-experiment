#!/usr/bin/env python3
"""Fig 4 (A)-(C): the 11 aligned model-defined constructs across 1,000 papers (GPT-5.5 judge).

  (A) papers by score level 3-5, low and high effort
  (B) papers with the construct present (score >= 3), low vs high
  (C) verified-reference rate at each score level (levels with >= 5 papers)

Writes .../model_defined_feature_scoring/outputs/by_judge/gpt_5_5/figures/
gpt_5_5_model_defined_common_feature_prevalence_two_panel.{png,json}; the JSON records the counts
and where the plotting area sits in the image, so plot_model_defined_feature_story.py can embed it.
"""

import json

import matplotlib
import numpy as np

matplotlib.use("Agg")  # the layout (and so the embedded raster) is laid out with the Agg renderer
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.legend_handler import HandlerTuple

from common.figure_style import CATEGORY_PT, LEGEND_PT, PRINT_WIDTH_IN, apply_print_style, emphasize_x_axis, panel_letter
from common.frontiers_figures import EFFORT_COLORS, MIN_FONT_PT, SCORE_SHADES, fit_to_width
from common.full_cohort_aligned_data import FIGURE_DIR, load_aligned_scores, load_alignment
from common.outcome_data import load_paper_metrics


PRESENCE_THRESHOLD = 3
MIN_GROUP_N = 5
SCORE1_LABEL_GAP = 13   # panel (C): minimum separation of the two score-1 labels, in % units of a strip
SCORE1_TICK_LEN = 0.02  # tick length, as a fraction of the strip width
# Short display names for panel (A); the full construct names are in the JSON output.
SHORT_LABELS = {
    "Direct topical relevance": "Topical relevance",
    "Reference-count target": "Reference count",
    "Bibliographic accuracy checking": "Bibliographic accuracy",
    "Thematic / subtopic coverage": "Subtopic coverage",
    "Authenticity / non-fabrication concern": "Authenticity concern",
    "Foundational / seminal source inclusion": "Seminal sources",
    "Methodological source inclusion": "Methodological sources",
    "Citation formatting / style": "Citation formatting",
    "Reference-list organization": "List organization",
    "Temporal source selection": "Temporal selection",
    "Brief relevance annotations": "Relevance notes",
}


def dose_response(frame, construct_id):
    """Mean verified-reference rate at each score level, dropping levels with < MIN_GROUP_N papers."""
    stats = frame[frame["construct_id"] == construct_id].groupby("score")["existence_rate"].agg(["mean", "count"])
    return stats[stats["count"] >= MIN_GROUP_N]


def main():
    apply_print_style()
    plt.rcParams["grid.color"] = "#E2E8F0"
    alignment = load_alignment()
    scores = {condition: load_aligned_scores(condition) for condition in ("low", "high")}
    rows = []
    for pair in alignment:
        low_scores = scores["low"].loc[scores["low"]["construct_id"] == pair["construct_id"], "score"].to_numpy()
        high_scores = scores["high"].loc[scores["high"]["construct_id"] == pair["construct_id"], "score"].to_numpy()
        rows.append((pair["label"], int((low_scores >= PRESENCE_THRESHOLD).sum()),
                     int((high_scores >= PRESENCE_THRESHOLD).sum()), low_scores, high_scores))
    order = sorted(range(len(rows)), key=lambda i: max(rows[i][1], rows[i][2]))
    alignment = [alignment[i] for i in order]
    rows = [rows[i] for i in order]
    labels = [SHORT_LABELS.get(row[0], row[0]) for row in rows]
    low_values = np.asarray([row[1] for row in rows])
    high_values = np.asarray([row[2] for row in rows])
    y = np.arange(len(rows))
    outcome = {condition: scores[condition].merge(load_paper_metrics(condition)[["focal_paper_id", "existence_rate"]],
                                                  on="focal_paper_id")
               for condition in ("low", "high")}

    fig = plt.figure(figsize=(PRINT_WIDTH_IN * 0.98, PRINT_WIDTH_IN * 0.98 / 1.1), dpi=300)
    grid = fig.add_gridspec(1, 2, width_ratios=[2.75, 0.95], wspace=0.13)
    counts = grid[0, 0].subgridspec(1, 2, width_ratios=[1.35, 1.6], wspace=0.075)
    ax_b = fig.add_subplot(counts[0, 0])
    ax = fig.add_subplot(counts[0, 1])

    # (C): one mini panel per construct, stacked on its row, sharing one score axis at the foot.
    strip = grid[0, 1].subgridspec(len(rows), 1, hspace=0.0)
    strip_axes = [fig.add_subplot(strip[position]) for position in range(len(rows))]
    for position, ax_strip in enumerate(strip_axes):
        construct_id = alignment[len(rows) - 1 - position]["construct_id"]
        starts = []  # (verified % at score 1, colour), labelled on the y axis
        for condition in ("low", "high"):
            stats = dose_response(outcome[condition], construct_id)
            ax_strip.plot(stats.index, stats["mean"] * 100, "-o", color=EFFORT_COLORS[condition],
                          linewidth=1.3, markersize=2.6, clip_on=False)
            if 1 in stats.index:
                starts.append((stats.loc[1, "mean"] * 100, EFFORT_COLORS[condition]))
        # Keep the two labels a line apart when the values are close.
        starts.sort()
        positions = [value for value, _ in starts]
        if len(positions) == 2 and positions[1] - positions[0] < SCORE1_LABEL_GAP:
            middle = sum(positions) / 2
            positions = [middle - SCORE1_LABEL_GAP / 2, middle + SCORE1_LABEL_GAP / 2]
        for (value, colour), label_y in zip(starts, positions):
            # Tick at the true value; the label may sit slightly off it.
            ax_strip.plot([-SCORE1_TICK_LEN, 0], [value, value], color=colour, linewidth=0.8,
                          transform=ax_strip.get_yaxis_transform(), clip_on=False)
            ax_strip.text(-SCORE1_TICK_LEN - 0.015, label_y, f"{value:.0f}", color=colour, ha="right", va="center",
                          fontsize=MIN_FONT_PT, transform=ax_strip.get_yaxis_transform(), clip_on=False)
        ax_strip.set_xlim(0.7, 5.3)
        ax_strip.set_ylim(70, 104)
        ax_strip.set_yticks([])
        ax_strip.spines["left"].set_color("#94A3B8")
        ax_strip.spines["left"].set_linewidth(0.5)
        ax_strip.set_xticks([1, 2, 3, 4, 5])
        ax_strip.tick_params(axis="x", length=0, labelbottom=False)
        for side in ("top", "right"):
            ax_strip.spines[side].set_visible(False)
        ax_strip.spines["bottom"].set_visible(False)
        ax_strip.grid(axis="x", alpha=0.8)
        ax_strip.set_axisbelow(True)
    strip_axes[-1].spines["bottom"].set_visible(True)
    strip_axes[-1].tick_params(axis="x", length=2.5, labelbottom=True)
    strip_axes[-1].set_xlabel("Feature score")
    # One shared y label for the stacked strips (each spans 70-104%).
    fig.text(0, 0.5, "Verified references (%)", rotation="vertical", va="center", ha="right",
             fontsize=plt.rcParams["axes.labelsize"], fontweight="bold",
             transform=strip_axes[len(strip_axes) // 2].transAxes
             + mtransforms.ScaledTranslation(-0.2, 0, fig.dpi_scale_trans))
    emphasize_x_axis(strip_axes[-1])
    panel_letter(strip_axes[0], "(C)", y=1.25)

    # (B): papers with the construct present, low vs high.
    for yi, low_value, high_value in zip(y, low_values, high_values):
        ax.plot([low_value, high_value], [yi, yi], color="#94A3B8", linewidth=1.8, zorder=1)
    ax.scatter(low_values, y, s=42, color=EFFORT_COLORS["low"], edgecolor="white", linewidth=0.4,
               label="Low effort", zorder=2)
    ax.scatter(high_values, y, s=42, color=EFFORT_COLORS["high"], edgecolor="white", linewidth=0.4,
               label="High effort", zorder=2)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 1010)
    ticks = np.arange(0, 1001, 500)
    ax.set_xticks(ticks, [str(value) for value in ticks])
    ax.set_xlabel("#Papers")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.8)
    ax.set_axisbelow(True)
    emphasize_x_axis(ax)
    panel_letter(ax, "(B)")

    # (A): papers by score level, stacked 3-4-5.
    score_levels = np.arange(3, 6)
    score_labels = ["3: Implicit", "4: Explicit", "5: Strong"]
    offsets = {"low": -0.2, "high": 0.2}
    for row_index, row in enumerate(rows):
        for condition, condition_scores in (("low", row[3]), ("high", row[4])):
            left = 0.0
            for score_index, count in enumerate([(condition_scores == score).sum() for score in score_levels]):
                ax_b.barh(row_index + offsets[condition], count, left=left, height=0.39,
                          color=SCORE_SHADES[condition][score_index], edgecolor="white", linewidth=0.3,
                          label=("Low" if condition == "low" else "High") + f" {score_labels[score_index]}"
                          if row_index == 0 else None)
                left += count
    ax_b.set_xlabel("#Papers, by score level")
    ax_b.set_yticks(y, labels, fontsize=CATEGORY_PT)
    ax_b.tick_params(axis="y", length=0, pad=6)
    ax_b.set_ylim(-0.6, len(rows) - 0.4)
    ax_b.axvline(0, color="#64748B", linewidth=0.8)
    ax_b.set_xlim(0, 1010)
    count_ticks = np.arange(0, 501, 500)  # 1000 would collide with the 0 of panel (B)
    ax_b.set_xticks(count_ticks, [str(value) for value in count_ticks])
    handles, legend_labels = ax_b.get_legend_handles_labels()
    # Compact key in the empty lower right: one entry per score level, showing its low | high colours.
    ax_b.legend([(handles[index], handles[index + 3]) for index in range(3)],
                [label.split(" ", 1)[1] for label in legend_labels[:3]],
                handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)},
                title="Score (low | high)", loc="lower right", ncol=1,
                fontsize=LEGEND_PT, title_fontsize=LEGEND_PT,
                frameon=True, facecolor="white", edgecolor="#CBD5E1", framealpha=0.95,
                handlelength=1.6, handleheight=0.7, handletextpad=0.4,
                labelspacing=0.25, borderpad=0.35, borderaxespad=0.2)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.grid(axis="x", alpha=0.8)
    ax_b.set_axisbelow(True)
    emphasize_x_axis(ax_b)
    panel_letter(ax_b, "(A)")

    fig.tight_layout()
    # Fig 4 embeds this raster at PRINT_WIDTH_IN * 0.98; size its text and lines to the Frontiers minimums there.
    fit_to_width(fig, PRINT_WIDTH_IN * 0.98, relayout=fig.tight_layout, min_font=MIN_FONT_PT + 0.3)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    output = FIGURE_DIR / "gpt_5_5_model_defined_common_feature_prevalence_two_panel.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")

    # Position of the plotting area inside the saved (tight-cropped) image, for the embedding figure.
    renderer = fig.canvas.get_renderer()
    saved = fig.get_tightbbox(renderer)
    axes_box = ax_b.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
    output.with_suffix(".json").write_text(json.dumps({
        "figure": output.name,
        "mapping": "full_1k_low_high_feature_pairs.json",
        "construct_count": len(rows),
        "matched_paper_count": 1000,
        "presence_threshold": PRESENCE_THRESHOLD,
        "counts": [{"construct": row[0], "low_present": row[1], "high_present": row[2]} for row in rows],
        "plot_area_fraction": {"left": (axes_box.x0 - saved.x0) / saved.width,
                               "right": (axes_box.x1 - saved.x0) / saved.width},
    }, indent=2) + "\n", encoding="utf-8")
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
