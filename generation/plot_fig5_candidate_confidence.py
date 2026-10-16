#!/usr/bin/env python3
"""Fig 6: title- and author-match scores of the high-effort references, with the classification rules.

Each reference is placed at the scores of the candidate behind its classification and coloured by
its final class; the shaded regions and lines show the possible and verified thresholds (verified
also requires a matching first-author surname, which the plot cannot show).
Output: outputs/final_figures/frontiers/Fig6.{tif,pdf,png} (single column).
"""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from common.frontiers_figures import FONT, export as frontiers_export
from generation.resolve_generated_references import (
    AUTHOR_EXISTS_THRESHOLD, POSSIBLE_AUTHOR_THRESHOLD, POSSIBLE_TITLE_THRESHOLD, TITLE_EXISTS_THRESHOLD,
)


HERE = Path(__file__).resolve().parents[1]
CLASS_COLORS = {"verified_real_paper": "#0072B2", "possible_real_paper": "#E69F00", "unresolved": "#7A7A7A"}
CLASS_LABELS = {"verified_real_paper": "Verified real paper", "possible_real_paper": "Possible real paper",
                "unresolved": "Unresolved"}
plt.rcParams.update({
    "font.sans-serif": [FONT, "DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 1.0,
    "grid.color": "#F1F5F9",
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "figure.facecolor": "#FFFFFF",
    "axes.facecolor": "#FFFFFF",
})


def main():
    refs = pd.read_csv(HERE / "outputs" / "high" / "resolved" / "all" / "references.csv")
    # Missing scores mean that no usable candidate was found.
    refs["title_match_score"] = refs["title_match_score"].fillna(0.0)
    refs["author_match_score"] = refs["author_match_score"].fillna(0.0)
    refs["existence_status"] = refs["existence_status"].fillna("unresolved")

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=300)
    # Score regions: everything, possible, verified (verified also needs a first-author match).
    ax.add_patch(Rectangle((0.00, 0.00), 1.00, 1.00, color="#F2F2F2", alpha=0.18, zorder=0))
    ax.add_patch(Rectangle((POSSIBLE_TITLE_THRESHOLD, POSSIBLE_AUTHOR_THRESHOLD), 1.0 - POSSIBLE_TITLE_THRESHOLD,
                           1.0 - POSSIBLE_AUTHOR_THRESHOLD, color="#F6D98B", alpha=0.24, zorder=0))
    ax.add_patch(Rectangle((TITLE_EXISTS_THRESHOLD, AUTHOR_EXISTS_THRESHOLD), 1.0 - TITLE_EXISTS_THRESHOLD,
                           1.0 - AUTHOR_EXISTS_THRESHOLD, color="#9CCBE5", alpha=0.28, zorder=0))
    for status in ("unresolved", "possible_real_paper", "verified_real_paper"):  # unresolved drawn first
        subset = refs[refs["existence_status"] == status]
        ax.scatter(subset["title_match_score"], subset["author_match_score"], color=CLASS_COLORS[status], alpha=0.52,
                   s=12, edgecolors="white", linewidths=0.20, rasterized=True, zorder=2,
                   label=f"{CLASS_LABELS[status]} (n={len(subset):,})")
    ax.plot([POSSIBLE_TITLE_THRESHOLD, POSSIBLE_TITLE_THRESHOLD, 1.00], [1.00, POSSIBLE_AUTHOR_THRESHOLD,
            POSSIBLE_AUTHOR_THRESHOLD], color="#B87800", lw=1.2, ls=":")
    ax.plot([TITLE_EXISTS_THRESHOLD, TITLE_EXISTS_THRESHOLD, 1.00], [1.00, AUTHOR_EXISTS_THRESHOLD,
            AUTHOR_EXISTS_THRESHOLD], color="#005A8D", lw=1.8)
    ax.set_xlim(0.2, 1.02)
    ax.set_ylim(0.2, 1.02)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Title-match score", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Author-match score", fontsize=9.5, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    rule_legend = ax.legend(handles=[
        Line2D([0], [0], color="#B87800", lw=1.2, ls=":",
               label=f"Possible: title > {POSSIBLE_TITLE_THRESHOLD:.2f}, author > {POSSIBLE_AUTHOR_THRESHOLD:.2f}"),
        Line2D([0], [0], color="#005A8D", lw=1.8,
               label=f"Verified: title ≥ {TITLE_EXISTS_THRESHOLD:.2f}, author ≥ {AUTHOR_EXISTS_THRESHOLD:.2f}"),
    ], loc="upper left", bbox_to_anchor=(0.0, -0.40), alignment="left", frameon=False, framealpha=0.92,
        edgecolor="#D7DEE7", fontsize=7.1, title="Classification rules", title_fontsize=8)
    ax.add_artist(rule_legend)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.14), alignment="left", frameon=False, fontsize=7.1,
              markerscale=2, title="Final outcome class", title_fontsize=8)
    frontiers_export(fig, "Fig6", width="single")
    plt.close(fig)


if __name__ == "__main__":
    main()
