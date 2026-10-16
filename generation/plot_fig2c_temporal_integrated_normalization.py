#!/usr/bin/env python3
"""Fig 5: publication years of the generated references relative to their focal papers.

(A) focal-paper years and joint density contours of focal year vs generated-reference year
(B) year difference (reference - focal paper) for low and high effort and for the focal papers'
    recorded bibliographies; bars are one-year histograms, lines Gaussian-smoothed guides.
Output: outputs/final_figures/frontiers/Fig5.{tif,pdf,png}.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde

from common.frontiers_figures import EFFORT_COLORS, FONT, export as frontiers_export


HERE = Path(__file__).resolve().parents[1]
COLORS = {**EFFORT_COLORS, "truth": "#059669", "dark": "#0F172A", "future": "#9CA3AF"}
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": [FONT, "DejaVu Sans"],
    "axes.edgecolor": "#CBD5E1", "axes.linewidth": 1,
    "grid.color": "#E2E8F0", "grid.linewidth": .6,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def load_data():
    generated, ids = {}, set()
    for condition in ("low", "high"):
        df = pd.read_csv(HERE / "outputs" / condition / "resolved" / "all" / "references.csv")
        df["focal_year"] = pd.to_numeric(df["focal_year"], errors="coerce")
        df["generated_year"] = pd.to_numeric(df["generated_year"], errors="coerce")
        df["difference"] = df["generated_year"] - df["focal_year"]
        generated[condition] = df
        ids.update(df["focal_paper_id"].astype(str))
    focal = generated["high"][["focal_paper_id", "focal_year"]].drop_duplicates("focal_paper_id")
    truth = pd.read_csv(HERE / "ground_truth_references_AA.csv", low_memory=False)
    truth = truth[truth["Citing_PaperID"].astype(str).isin(ids)].copy()
    truth["difference"] = pd.to_numeric(truth["Year_x"], errors="coerce") - pd.to_numeric(truth["Year_y"], errors="coerce")
    return generated, focal, truth


def relative_frequency(series, edges, limits):
    values = np.clip(series.dropna().to_numpy(), *limits)
    counts, _ = np.histogram(values, bins=edges)
    return (edges[:-1] + edges[1:]) / 2, counts / len(values) * 100, len(values)


def main():
    generated, focal, truth = load_data()
    fig = plt.figure(figsize=(10.5, 11), dpi=300)
    outer = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[1.65, 1], hspace=.34)
    top = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0], height_ratios=[1, 4.8], hspace=.05)
    ax_hist = fig.add_subplot(top[0])
    ax_joint = fig.add_subplot(top[1], sharex=ax_hist)

    # Focal-paper years, aligned with the joint plot.
    fy = focal["focal_year"].dropna().to_numpy()
    f_edges = np.arange(1998.5, 2022.5)
    f_counts, _ = np.histogram(fy, bins=f_edges)
    f_centers = (f_edges[:-1] + f_edges[1:]) / 2
    ax_hist.bar(f_centers, f_counts / len(fy) * 100, width=.78, color="#94A3B8", alpha=.55,
                edgecolor="#64748B", linewidth=.5)
    ax_hist.plot(f_centers, gaussian_filter1d(f_counts / len(fy) * 100, 1.1), color="#475569", linewidth=1.7,
                 label=f"Focal papers (n={len(fy)})")
    ax_hist.set_ylabel("Focal papers\n(%)", fontsize=9)
    ax_hist.tick_params(axis="x", labelbottom=False)
    ax_hist.spines[["top", "right"]].set_visible(False)
    ax_hist.legend(frameon=False, fontsize=8, loc="upper left")
    ax_hist.set_title("(A)", loc="left", fontweight="bold")

    # Joint density contours; the evaluation grid is padded so contours close off-canvas.
    years = np.linspace(1965, 2030, 300)
    ax_joint.plot(years, years, color=COLORS["dark"], linestyle="--", linewidth=1.2,
                  label="Generated year = focal year", zorder=4)
    ax_joint.fill_between(years, years, 2030, color=COLORS["future"], alpha=.12, label="Future-reference zone")
    for condition in ("low", "high"):
        valid = generated[condition][["focal_year", "generated_year"]].dropna()
        valid = valid[valid["focal_year"].between(1995, 2022) & valid["generated_year"].between(1970, 2025)]
        kde = gaussian_kde(np.vstack([valid["focal_year"], valid["generated_year"]]))
        xi, yi = np.mgrid[1992:2026:135j, 1965:2030:150j]
        zi = kde(np.vstack([xi.ravel(), yi.ravel()])).reshape(xi.shape)
        ax_joint.contour(xi, yi, zi, levels=5, colors=[COLORS[condition]], linewidths=1.25, alpha=.88)
    ax_joint.plot([], [], color=COLORS["low"], label="Low effort")
    ax_joint.plot([], [], color=COLORS["high"], label="High effort")
    ax_joint.set_xlim(1995, 2023)
    ax_joint.set_ylim(1970, 2025)
    ax_joint.set_xlabel("Focal-paper publication year", fontsize=9.5, fontweight="bold")
    ax_joint.set_ylabel("Generated-reference publication year", fontsize=9.5, fontweight="bold")
    ax_joint.grid(alpha=.35)
    ax_joint.spines[["top", "right"]].set_visible(False)
    ax_joint.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=8, loc="upper left")

    # (B) one-year histograms plus thin smoothed guides.
    ax = fig.add_subplot(outer[1])
    edges, limits = np.arange(-40.5, 11.5), (-40, 10)
    centers, low_pct, low_n = relative_frequency(generated["low"]["difference"], edges, limits)
    _, high_pct, high_n = relative_frequency(generated["high"]["difference"], edges, limits)
    _, truth_pct, truth_n = relative_frequency(truth["difference"], edges, limits)
    ax.bar(centers - .18, low_pct, width=.34, color=COLORS["low"], alpha=.28, edgecolor=COLORS["low"],
           linewidth=.45, label=f"Low effort (n={low_n:,})")
    ax.bar(centers + .18, high_pct, width=.34, color=COLORS["high"], alpha=.28, edgecolor=COLORS["high"],
           linewidth=.45, label=f"High effort (n={high_n:,})")
    ax.step(centers, truth_pct, where="mid", color=COLORS["truth"], linewidth=1.35, linestyle="--",
            label=f"Recorded bibliography (n={truth_n:,})")
    for pct, color, linestyle in ((low_pct, COLORS["low"], "-"), (high_pct, COLORS["high"], "-"),
                                  (truth_pct, COLORS["truth"], "--")):
        ax.plot(centers, gaussian_filter1d(pct, 1.35), color=color, linewidth=1.35, linestyle=linestyle, alpha=.95)
    ax.axvline(0, color=COLORS["dark"], linestyle=":", linewidth=1.2, label="Reference year = focal year")
    ax.axvspan(0, 10.5, color=COLORS["future"], alpha=.12)
    ax.text(.4, ax.get_ylim()[1] * .94, "Post-publication\nreferences", color=COLORS["dark"], fontsize=8, va="top")
    ax.set_xticks([-40, -30, -20, -10, 0, 10])
    ax.set_xticklabels(["≤−40", "−30", "−20", "−10", "0", "≥10"])
    ax.set_xlabel("Year difference (reference − focal paper)", fontsize=9.5, fontweight="bold")
    ax.set_ylabel("Relative frequency (%)", fontsize=9.5, fontweight="bold")
    ax.set_title("(B)", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    frontiers_export(fig, "Fig5")
    plt.close(fig)


if __name__ == "__main__":
    main()
