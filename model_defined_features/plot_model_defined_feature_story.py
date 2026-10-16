#!/usr/bin/env python3
"""Fig 4: model-defined features under low and high effort (GPT-5.5 judge, 1,000 papers).

Embeds panels (A)-(C) from plot_model_defined_common_feature_two_panel.py (run that first) and adds
  (D) number of aligned constructs present (score >= 3) per trace, by saved-summary length
      in o200k_base tokens (empty summaries are kept in the 0-token bin).
Output: outputs/final_figures/frontiers/Fig4.{tif,pdf,png}, plus a JSON description in
.../model_defined_feature_scoring/outputs/by_judge/gpt_5_5/figures/.
"""

import json

import matplotlib
import numpy as np
import pandas as pd
import tiktoken

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common.figure_style import PRINT_WIDTH_IN, apply_print_style, emphasize_x_axis, panel_letter
from common.frontiers_figures import EFFORT_COLORS, export as frontiers_export
from common.full_cohort_aligned_data import FIGURE_DIR, load_aligned_scores, load_alignment, load_batches
from common.judge_config import reasoning_summary


OUTPUT_STEM = "gpt_5_5_model_defined_feature_story"
PRESENCE_THRESHOLD = 3
CONSTRUCT_COUNT = 11
TOKENIZER = tiktoken.get_encoding("o200k_base")
BIN_EDGES = np.array([-0.5, 0.5, *np.arange(50.5, 600.5 + 50, 50), np.inf])
BIN_LABELS = ["0"] + [f"{start}–{start + 49}" for start in range(1, 600, 50)] + ["601+"]


def paper_data(condition):
    """Per paper: number of aligned constructs present, saved-summary tokens and length bin."""
    scores = load_aligned_scores(condition)
    counts = (scores.assign(present=scores["score"].ge(PRESENCE_THRESHOLD))
              .groupby("focal_paper_id", as_index=False)
              .agg(present_feature_count=("present", "sum")))
    summaries = {paper_id: reasoning_summary(condition, paper_id) for paper_id in sorted(counts["focal_paper_id"])}
    tokens = pd.DataFrame({"focal_paper_id": list(summaries),
                           "summary_tokens": [len(TOKENIZER.encode(s)) if s else 0 for s in summaries.values()]})
    frame = counts.merge(tokens, on="focal_paper_id", validate="one_to_one")
    frame["length_bin"] = pd.cut(frame["summary_tokens"], BIN_EDGES, labels=BIN_LABELS, right=True, include_lowest=True)
    return frame


def style_axis(ax, grid_axis="y"):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color="#E2E8F0", linewidth=0.5)
    ax.set_axisbelow(True)


def main():
    conditions = ("low", "high")
    length_data = {condition: paper_data(condition) for condition in conditions}
    apply_print_style()

    # (A)-(C): embed the source figure as is.
    source = FIGURE_DIR / "gpt_5_5_model_defined_common_feature_prevalence_two_panel.png"
    panel_a_image = plt.imread(source)
    panel_a_plot_area = json.loads(source.with_suffix(".json").read_text())["plot_area_fraction"]

    # Size the embedded row from the source image so it is not rescaled: its text then prints at
    # the same point sizes as panel (D).
    right, top_pad, bottom_pad, gap = 0.98, 0.34, 0.50, 0.16
    image_left, image_right = 0.01, 0.99
    content_width = PRINT_WIDTH_IN * (image_right - image_left)
    row_a_height = content_width * panel_a_image.shape[0] / panel_a_image.shape[1]
    row_b_height = 1.75
    figure_height = top_pad + row_a_height + gap + row_b_height + bottom_pad
    fig = plt.figure(figsize=(PRINT_WIDTH_IN, figure_height), dpi=300)
    rng = np.random.default_rng(55)
    row_a_top = 1 - top_pad / figure_height
    ax_image = fig.add_axes([image_left, row_a_top - row_a_height / figure_height,
                             image_right - image_left, row_a_height / figure_height])
    ax_image.imshow(panel_a_image, interpolation="none", aspect="auto")
    ax_image.axis("off")

    # (D): aligned-construct counts against saved-summary length, y axis aligned with the columns above.
    row_b_left = image_left + panel_a_plot_area["left"] * (image_right - image_left)
    bottom_frame = fig.add_gridspec(1, 1, left=row_b_left, right=right,
                                    top=row_a_top - (row_a_height + gap) / figure_height,
                                    bottom=bottom_pad / figure_height)
    bottom = bottom_frame[0].subgridspec(2, 1, hspace=0.12)
    d_axes = [fig.add_subplot(bottom[index]) for index in range(2)]
    maximum_bin_n = max(int((frame["length_bin"] == length_bin).sum())
                        for frame in length_data.values() for length_bin in BIN_LABELS)
    for index, (ax, condition) in enumerate(zip(d_axes, conditions)):
        frame = length_data[condition]
        color = EFFORT_COLORS[condition]
        distributions, positions, widths = [], [], []
        for position, length_bin in enumerate(BIN_LABELS):
            values = frame.loc[frame["length_bin"] == length_bin, "present_feature_count"].to_numpy()
            if len(values):
                distributions.append(values)
                positions.append(position)
                widths.append(0.12 + 0.66 * len(values) / maximum_bin_n)
                jitter = rng.uniform(-0.24, 0.24, len(values))
                ax.scatter(position + jitter, values + rng.uniform(-0.08, 0.08, len(values)), s=2.4, color=color,
                           alpha=0.35, edgecolors="none", rasterized=True, zorder=2)
        ax.boxplot(distributions, positions=positions, widths=widths, patch_artist=True, showfliers=False,
                   medianprops={"color": "white", "linewidth": 1.0},
                   boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.85, "linewidth": 0.8},
                   whiskerprops={"color": color, "linewidth": 0.9}, capprops={"color": color, "linewidth": 0.9}, zorder=4)
        ax.set_yticks([0, 5, 10])
        ax.set_ylim(-0.5, CONSTRUCT_COUNT + 0.5)
        ax.set_xlim(-0.55, len(BIN_LABELS) - 0.45)
        style_axis(ax)
        # Both rows get the same heavy x axis and ticks; only the bottom one is labelled.
        ax.set_xticks(range(len(BIN_LABELS)), [""] * len(BIN_LABELS) if index == 0 else BIN_LABELS)
        emphasize_x_axis(ax)
    plt.setp(d_axes[-1].get_xticklabels(), rotation=30, ha="right")
    d_axes[-1].set_xlabel("Summary length (tokens)")
    panel_letter(d_axes[0], "(D)")
    # One y label for the stacked pair: at this row height two would collide.
    row_b_bottom = bottom_pad / figure_height
    row_b_top = row_a_top - (row_a_height + gap) / figure_height
    fig.text(row_b_left - 0.035, (row_b_bottom + row_b_top) / 2, "#Features per trace", rotation="vertical",
             va="center", ha="center", fontsize=plt.rcParams["axes.labelsize"], fontweight="bold")
    fig.legend(handles=[Line2D([], [], marker="o", linestyle="none", markersize=5, color=EFFORT_COLORS[condition],
                               label=f"{condition.capitalize()} effort") for condition in conditions],
               loc="upper center", bbox_to_anchor=(0.5, 1 + 0.02 / figure_height), ncol=2,
               frameon=False, handletextpad=0.3, columnspacing=1.6)
    frontiers_export(fig, "Fig4")
    plt.close(fig)

    metadata_path = FIGURE_DIR / f"{OUTPUT_STEM}.json"
    metadata_path.write_text(json.dumps({
        "figure": "outputs/final_figures/frontiers/Fig4.{tif,png,pdf}",
        "matched_paper_count_per_condition": 1000,
        "discovery_batches_per_condition": 5,
        "batch_size": 200,
        "batch_local_feature_counts": {condition: [len(feature_ids) for feature_ids, _ in load_batches(condition).values()]
                                       for condition in conditions},
        "aligned_construct_count": len(load_alignment()),
        "presence_threshold": ">=3",
        "panels": {
            "A": "score composition and prevalence of 11 cross-batch aligned constructs",
            "B": "aligned-construct count distribution by saved reasoning-summary token length",
        },
        "tokenizer": "o200k_base",
        "zero_token_policy": "retain missing/empty summaries in the 0-token bin",
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved: {metadata_path}")


if __name__ == "__main__":
    main()
