#!/usr/bin/env python3
"""Fig 2: reasoning-summary length and verified references under low and high effort.

  (A) pooled saved-summary word distributions
  (D) paper-level verified-reference raincloud, for all references and for the references
      generated under both efforts (identity-supported shared references)
Bottom rows share one field axis (fields with fewer than 10 papers pooled into Humanities and
Social sciences), sorted by the high-effort mean summary length:
  (B) focal papers per field
  (C) saved-summary words per field (mean +/- SD, mean, median)
  (E) mean verified-reference rate per field, low vs high

Inputs: outputs/<effort>/raw, outputs/<effort>/resolved/all/references.csv,
outputs/low_high_title_overlap_full_1k.csv, focal_papers_AA.csv, cohort.csv.
Output: outputs/final_figures/frontiers/Fig2.{tif,pdf,png}.
"""

import collections
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

from common.frontiers_figures import export as frontiers_export
from generation.plot_trace_length_and_structure import COLORS, CONDITIONS, LABELS, load_condition


HERE = Path(__file__).resolve().parents[1]
CONNECT = "#8A94A3"
SHARED_GREY = "#A78BDA"  # shared low/high references in panel D (purple)
MARKER_S = 230           # scatter size (pt^2) of the mean/median markers in all panels
FIELD_GROUPS = {
    "Art": "Humanities",
    "History": "Humanities",
    "Philosophy": "Humanities",
    "Business": "Social sciences",
    "Political science": "Social sciences",
}


def references(condition):
    with (HERE / "outputs" / condition / "resolved" / "all" / "references.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def verified_rates(condition):
    """Percentage of each paper's references that are verified (possible matches count as not verified)."""
    counts = {}
    for row in references(condition):
        verified, total = counts.get(row["focal_paper_id"], (0, 0))
        counts[row["focal_paper_id"]] = (verified + (row["existence_status"] == "verified_real_paper"), total + 1)
    return {paper: 100.0 * verified / total for paper, (verified, total) in counts.items()}


def shared_verified_rates():
    """Paper rates for identity-supported references verified in each condition.

    The denominator is every reference generated for that focal paper, as in verified_rates().
    Candidate pairs require an exact normalized-title match within the same focal paper. Identity
    support then requires a shared matched DOI or, when a common DOI is unavailable, a shared match
    to the same entry of the focal paper's observed bibliography. High-effort entries in two pairs
    are counted once.
    """
    raw, totals = {}, {}
    for condition in CONDITIONS:
        rows = references(condition)
        raw[condition] = {(r["focal_paper_id"], r["generated_reference_index"]): r for r in rows}
        totals[condition] = collections.Counter(r["focal_paper_id"] for r in rows)

    grouped = collections.defaultdict(list)
    with (HERE / "outputs" / "low_high_title_overlap_full_1k.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["same_focal_paper"] == "True":
                grouped[(row["low_focal_paper_id"], row["low_reference_index"], row["normalized_title"])].append(row)

    supported = []
    for candidates in grouped.values():
        if len(candidates) != 1:
            continue
        match = candidates[0]
        low_key = (match["low_focal_paper_id"], match["low_reference_index"])
        high_key = (match["high_focal_paper_id"], match["high_reference_index"])
        low_row, high_row = raw["low"][low_key], raw["high"][high_key]
        low_doi = low_row["matched_doi"].strip().lower()
        high_doi = high_row["matched_doi"].strip().lower()
        same_doi = bool(low_doi and high_doi and low_doi == high_doi)
        same_observed_record = (
            not (low_doi and high_doi)
            and low_row["ground_truth_match_status"] == "in_focal_reference_list"
            and high_row["ground_truth_match_status"] == "in_focal_reference_list"
            and low_row["ground_truth_matched_paper_id"] == high_row["ground_truth_matched_paper_id"]
        )
        if same_doi or same_observed_record:
            supported.append((low_key, high_key))
    if len(supported) != 7334:
        raise ValueError(f"Expected 7,334 identity-supported pairs, found {len(supported):,}")

    rates = {}
    for condition, pair_pos in (("low", 0), ("high", 1)):
        entry_keys = {pair[pair_pos] for pair in supported}
        verified = collections.Counter(paper for paper, index in entry_keys
                                       if raw[condition][(paper, index)]["existence_status"] == "verified_real_paper")
        rates[condition] = {paper: 100.0 * verified[paper] / total for paper, total in totals[condition].items()}
    return rates


def build_rows(low, high, ids):
    """Per field (after pooling small fields): papers and saved-summary word statistics."""
    with (HERE / "focal_papers_AA.csv").open(encoding="utf-8") as fh:
        fields = {r["PaperID"]: r["top_field"] for r in csv.DictReader(fh)}
    missing = [i for i in ids if not fields.get(i)]
    if missing:
        raise KeyError(f"{len(missing)} papers have no top_field (e.g. {missing[:3]})")
    by_field = {}
    for i in ids:
        by_field.setdefault(FIELD_GROUPS.get(fields[i], fields[i]), []).append(i)
    rows = []
    for field, fids in by_field.items():
        lw = np.array([low[i]["words"] for i in fids], float)
        hw = np.array([high[i]["words"] for i in fids], float)
        rows.append(dict(
            field=field, n=len(fids), paper_ids=fids,
            lo_med=np.median(lw), hi_med=np.median(hw),
            lo_mean=float(lw.mean()), hi_mean=float(hw.mean()),
            lo_sd=float(lw.std(ddof=1)), hi_sd=float(hw.std(ddof=1)),
        ))
    return rows


def panel_counts(ax, rows, y, letter):
    counts = [r["n"] for r in rows]
    ax.barh(y, counts, height=0.85, color="#9AA7B4")
    for yi, count in zip(y, counts):
        ax.text(count + max(counts) * 0.02, yi, str(count), va="center", ha="left",
                fontsize=7, color="#4B5563")
    ax.set_xlim(0, max(counts) * 1.20)
    ax.set_yticks(y, [r["field"] for r in rows], fontsize=9)
    ax.set_xlabel("#papers")
    ax.set_title(letter, loc="left", fontweight="bold")


def panel_words(ax, rows, y, letter):
    for yi, r in zip(y, rows):
        for condition in ("low", "high"):
            prefix = "lo" if condition == "low" else "hi"
            mean = r[f"{prefix}_mean"]
            color = COLORS[condition]
            row_y = yi - 0.16 if condition == "low" else yi + 0.16
            left = max(0.0, mean - r[f"{prefix}_sd"])
            right = mean + r[f"{prefix}_sd"]
            ax.hlines(row_y, left, right, color=color, lw=2.6, alpha=0.9, zorder=3)
            ax.vlines([left, right], row_y - 0.09, row_y + 0.09, color=color, lw=2.0, alpha=0.9, zorder=3)
            ax.scatter([mean], [row_y], s=MARKER_S, color=color, edgecolor="white", lw=1.0, zorder=5)
            ax.scatter([r[f"{prefix}_med"]], [row_y], s=MARKER_S, marker="^",
                       color=color, edgecolor="white", lw=0.8, zorder=6)
    xmax = max(r[f"{p}_mean"] + r[f"{p}_sd"] for r in rows for p in ("lo", "hi"))
    ax.set_xlim(-6, xmax * 1.03)
    ax.set_xticks(np.arange(0, xmax, 200))
    ax.set_xlabel(r"$\mu$ summary length (words)")
    ax.set_title(letter, loc="left", fontweight="bold")


def panel_field_rates(ax, rows, y, rates, letter):
    """Low/high mean verified-reference rate per field, connected, with the difference in pp."""
    for yi, row in zip(y, rows):
        centres = []
        for condition in ("low", "high"):
            values = np.array([rates[condition][p] for p in row["paper_ids"] if p in rates[condition]], dtype=float)
            if not len(values):
                continue
            mean = values.mean()
            ax.scatter([mean], [yi], s=MARKER_S, color=COLORS[condition], edgecolor="white", linewidth=0.7, zorder=5)
            centres.append((mean, yi))
        if len(centres) == 2:
            ax.plot(*zip(*centres), color=CONNECT, lw=5.0, zorder=2, solid_capstyle="round")
            delta = centres[1][0] - centres[0][0]
            ax.text(1.0, yi, f"{delta:+.1f}", transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=7.5, color=COLORS["high"] if delta > 0 else COLORS["low"])
    ax.text(1.0, -0.62, r"$\Delta$ (pp)", transform=ax.get_yaxis_transform(), ha="right",
            va="bottom", fontsize=7.5, color="#475569", fontweight="bold")
    ax.set_xlim(47, 110)  # lowest field mean (Humanities, low effort) is ~53%
    ax.set_xticks(np.arange(50, 101, 10))
    ax.set_xlabel(r"$\mu$ verified references (%)")
    ax.set_title(letter, loc="left", fontweight="bold")


def panel_pooled_words(ax, pooled_lo, pooled_hi):
    """Overlaid low/high histograms (50-word bins) with count-scaled KDEs."""
    width = 50
    max_value = max(float(np.max(v)) for v in (pooled_lo, pooled_hi))
    bins = np.arange(0, max_value + width, width)
    grid = np.linspace(0, bins[-1], 900)
    peak_count = 0
    handles = []
    for condition, values in (("low", pooled_lo), ("high", pooled_hi)):
        color = COLORS[condition]
        counts, _, _ = ax.hist(values, bins=bins, rwidth=0.92, color=mcolors.to_rgba(color, 0.32),
                               edgecolor=color, linewidth=2.0)
        peak_count = max(peak_count, counts.max())
        # KDE of the non-empty summaries, rescaled so its peak meets the tallest bar.
        density = gaussian_kde(values[values > 0])(grid)
        ax.plot(grid, density / density.max() * counts.max(), color=color, linewidth=4.5, zorder=4)
        handles.append(Patch(facecolor=mcolors.to_rgba(color, 0.30), edgecolor=mcolors.to_rgba(color, 0.45),
                             label=LABELS[condition]))
    handles.append(Patch(facecolor=mcolors.to_rgba(SHARED_GREY, 0.30), edgecolor=mcolors.to_rgba(SHARED_GREY, 0.45),
                         label="Shared"))
    ax.set_xlim(-20, 1070)  # the long tail beyond is clipped from view
    ax.set_ylim(0, peak_count * 1.25)
    ax.set_xticks(np.arange(0, 1001, 250))
    ax.set_title("(A)", loc="left", fontweight="bold")
    ax.set_xlabel("Summary length (words)")
    ax.set_ylabel("#papers")
    ax.legend(handles=handles + [
        Line2D([], [], marker="o", linestyle="none", color="#6B7280", markersize=MARKER_S ** 0.5, label="Mean"),
        Line2D([], [], marker="^", linestyle="none", color="#6B7280", markersize=MARKER_S ** 0.5, label="Median"),
    ], loc="upper right", frameon=True, fancybox=False, edgecolor="#64748B", framealpha=1.0, ncol=2)


def panel_raincloud(ax, rates, shared_rates, ylabel="Number of focal papers"):
    """Count-scaled KDEs (papers per 5-pp bin) of all (filled) and shared (dashed) verified rates;
    below, one rain row each for low, high and shared references."""
    rng = np.random.default_rng(42)
    bin_pp = 5.0
    x_vals = np.linspace(0, 120, 290)
    series = {}
    for condition in ("low", "high"):
        series[(condition, "all")] = np.asarray(list(rates[condition].values()), dtype=float)
        series[(condition, "shared")] = np.asarray(list(shared_rates[condition].values()), dtype=float)
    curves = {key: gaussian_kde(values, bw_method=0.25)(x_vals) * len(values) * bin_pp for key, values in series.items()}
    peak = max(curve.max() for curve in curves.values())
    for condition in ("low", "high"):
        color = COLORS[condition]
        ax.fill_between(x_vals, 0, curves[(condition, "all")], color=color, alpha=0.32, lw=0)
        ax.plot(x_vals, curves[(condition, "all")], color=color, lw=4.5)
    ax.fill_between(x_vals, 0, curves[("low", "shared")], color=SHARED_GREY, alpha=0.12, lw=0)
    ax.plot(x_vals, curves[("low", "shared")], color=SHARED_GREY, lw=2.6, linestyle=(0, (4, 2)), zorder=5)

    # Shared low and high rates nearly coincide (the same 7,334 references); show the low-effort one.
    for frac, condition, kind in ((-0.22, "low", "all"), (-0.46, "high", "all"), (-0.74, "low", "shared")):
        row_y = frac * peak
        values = series[(condition, kind)]
        color = COLORS[condition] if kind == "all" else SHARED_GREY
        jitter = rng.uniform(-0.065, 0.065, size=len(values)) * peak
        if kind == "all":
            ax.scatter(values, row_y + jitter, color=color, alpha=0.35, s=18, edgecolor="none", rasterized=True)
        else:
            ax.scatter(values, row_y + jitter, facecolor="none", edgecolor=color, alpha=0.45, s=18,
                       linewidth=0.7, rasterized=True)
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        ax.plot([q25, q75], [row_y, row_y], color="#0F172A", lw=4.0, solid_capstyle="round", zorder=5)
        marker_face = color if kind == "all" else "white"
        ax.scatter(median, row_y, s=MARKER_S * 0.75, marker="^", zorder=6, facecolor=marker_face,
                   edgecolor=color if kind == "shared" else "white", linewidth=1.6 if kind == "shared" else 0.8)
        ax.scatter(values.mean(), row_y, s=MARKER_S * 0.75, marker="o", zorder=7, facecolor=marker_face,
                   edgecolor=color if kind == "shared" else "white", linewidth=1.6 if kind == "shared" else 0.9)
    ax.axhline(0, color="#64748B", lw=0.8)
    # The y-axis covers the distributions only, from the KDE baseline (0) up.
    step = 25 if peak < 150 else 50
    ticks = np.arange(0, np.ceil(peak / step) * step + 1e-9, step)
    ymin, ymax = -0.90 * peak, max(1.15 * peak, ticks[-1])
    ax.set_ylim(ymin, ymax)
    ax.set_yticks(ticks)
    ax.spines["left"].set_bounds(0, ticks[-1])
    ax.set_ylabel(ylabel)
    ax.yaxis.label.set_y((ticks[-1] / 2 - ymin) / (ymax - ymin))  # centre the label on the distributions
    ax.set_xlim(0, 110)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_title("(D)", loc="left", fontweight="bold")
    ax.set_xlabel("Verified references (%)")


def main():
    with (HERE / "cohort.csv").open(encoding="utf-8") as fh:
        cohort_ids = [row["PaperID"] for row in csv.DictReader(fh)][:1000]
    data = {c: load_condition(c, paper_ids=cohort_ids) for c in CONDITIONS}
    low, high = data["low"], data["high"]
    ids = sorted(set(low) & set(high))

    rates = {c: verified_rates(c) for c in CONDITIONS}
    shared_rates = shared_verified_rates()
    rows = build_rows(low, high, ids)
    rows.sort(key=lambda r: (-r["hi_mean"], r["field"]))  # longest high-effort summaries first
    nf = len(rows)
    y = np.arange(nf)
    pooled_lo = np.array([low[i]["words"] for i in ids], dtype=float)
    pooled_hi = np.array([high[i]["words"] for i in ids], dtype=float)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                         "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})
    fig = plt.figure(figsize=(13.5, 16.15))
    # Two columns: A above B-C (word breakdowns); D above E (verified rates).
    outer = fig.add_gridspec(2, 2, height_ratios=[0.65, 1.534], hspace=0.18, width_ratios=[3.77, 2.05], wspace=0.05)
    left_bottom = outer[1, 0].subgridspec(1, 2, width_ratios=[0.92, 2.85], wspace=0.18)
    ax_a = fig.add_subplot(outer[0, 0])
    ax_d = fig.add_subplot(outer[0, 1])
    ax_b = fig.add_subplot(left_bottom[0])
    ax_c = fig.add_subplot(left_bottom[1], sharey=ax_b)
    ax_e = fig.add_subplot(outer[1, 1], sharey=ax_b)

    panel_pooled_words(ax_a, pooled_lo, pooled_hi)
    panel_raincloud(ax_d, rates, shared_rates)
    ax_b.set_ylim(nf - 0.5, -0.5)
    panel_counts(ax_b, rows, y, letter="(B)")
    panel_words(ax_c, rows, y, letter="(C)")
    panel_field_rates(ax_e, rows, y, rates, letter="(E)")
    for ax in (ax_a, ax_d, ax_b, ax_c, ax_e):
        ax.tick_params(axis="x", labelsize=7, length=2, width=0.7, pad=2)
        ax.xaxis.label.set_fontsize(8)
        ax.xaxis.labelpad = 3
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
    # No y-axis line or tick marks on B-E; B keeps the field names as labels.
    for ax in (ax_b, ax_c, ax_e):
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="y", color="#F1F5F9", linewidth=0.8)
        ax.set_axisbelow(True)
    for ax in (ax_c, ax_e):
        ax.tick_params(labelleft=False)

    fig.subplots_adjust(left=0.125, right=0.99, top=0.96, bottom=0.07)
    # Narrow A so D's y-axis (ticks and label) has room; D stays aligned with E.
    pos = ax_a.get_position()
    ax_a.set_position([pos.x0, pos.y0, pos.width - 0.055, pos.height])
    frontiers_export(fig, "Fig2")
    plt.close(fig)


if __name__ == "__main__":
    main()
