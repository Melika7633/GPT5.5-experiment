#!/usr/bin/env python3
"""Feature presence (score >= 3) against saved-summary length, low and high effort overlaid.

presence_panel() draws Fig 3 (C). The script itself draws one panel per feature with binned
presence and Wilson 95% intervals (hollow markers: bins with fewer than SPARSE_N traces) plus a
logistic fit of presence on log(summary tokens) per condition, and writes the length-adjusted
effort odds ratios from logit(present) ~ log(tokens) + effort, fitted on traces with a summary.
"""

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common.judge_config import judge_figure_dir
from predefined_features.plot_feature_positive_counts_by_summary_tokens import load_condition


JUDGE_ID = "gpt_5_5"
OUT_DIR = judge_figure_dir(JUDGE_ID)
LOW = "#4C78A8"   # = frontiers_figures.EFFORT_COLORS
HIGH = "#E45756"
DARK = "#0F172A"
FEATURES = [
    ("epistemic_caution", "Epistemic Caution"),
    ("temporal_constraint_awareness", "Temporal Constraint Awareness"),
    ("reference_verifiability_checking", "Reference Verifiability Checking"),
    ("metadata_verification", "Metadata Verification"),
]
BIN_WIDTH = 50
X_MAX = 650     # traces longer than this are pooled into the last bin
SPARSE_N = 20   # bins below this are drawn hollow
MIN_PLOT_N = 5  # bins below this are not drawn
Z = 1.96

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "text.color": DARK,
    "axes.labelcolor": DARK,
    "xtick.color": "#334155",
    "ytick.color": "#334155",
    "axes.edgecolor": "#94A3B8",
    "axes.linewidth": 0.9,
})


def wilson(k, n):
    p = k / n
    centre = (p + Z**2 / (2 * n)) / (1 + Z**2 / n)
    half = Z * np.sqrt(p * (1 - p) / n + Z**2 / (4 * n**2)) / (1 + Z**2 / n)
    return 100 * p, 100 * (centre - half), 100 * (centre + half)


def binned(frame, feature):
    """Presence with Wilson interval per 50-token bin; traces without a summary form an x=0 bin."""
    tokens = frame["summary_tokens"].clip(upper=X_MAX - 1)
    edges = np.arange(1, X_MAX + BIN_WIDTH, BIN_WIDTH)
    groups = [(0.0, frame["summary_tokens"] == 0)]
    for left, right in zip(edges[:-1], edges[1:]):
        groups.append(((left + right - 1) / 2, (tokens >= left) & (tokens < right)))
    rows = []
    for centre, mask in groups:
        n = int(mask.sum())
        if n >= MIN_PLOT_N:
            rows.append((centre, n, *wilson(int((frame.loc[mask, feature] >= 3).sum()), n)))
    return pd.DataFrame(rows, columns=["x", "n", "pct", "lo", "hi"])


def logistic_curve(frame, feature):
    """Fit presence ~ log(tokens) on traces with a summary; return grid, fit and 95% band (in %)."""
    data = frame[frame["summary_tokens"] > 0]
    y = (data[feature] >= 3).astype(float).to_numpy()
    X = sm.add_constant(np.log(data["summary_tokens"].to_numpy()))
    fit = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    lo_t, hi_t = np.percentile(data["summary_tokens"], [1, 99])
    grid = np.linspace(max(lo_t, 1), min(hi_t, X_MAX), 200)
    Xg = sm.add_constant(np.log(grid))
    eta = Xg @ fit.params
    se = np.sqrt(np.einsum("ij,jk,ik->i", Xg, fit.cov_params(), Xg))

    def expit(v):
        return 100 / (1 + np.exp(-v))
    return grid, expit(eta), expit(eta - Z * se), expit(eta + Z * se)


def adjusted_effort_or(low, high, feature):
    data = pd.concat([low.assign(high_effort=0), high.assign(high_effort=1)])
    data = data[data["summary_tokens"] > 0]
    y = (data[feature] >= 3).astype(float).to_numpy()
    X = sm.add_constant(np.column_stack([np.log(data["summary_tokens"]), data["high_effort"]]))
    fit = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    beta, se = fit.params[2], fit.bse[2]
    return np.exp(beta), np.exp(beta - Z * se), np.exp(beta + Z * se), fit.pvalues[2]


def presence_panel(ax, low, high, feature, title, letter, show_ylabel, show_fit=False):
    """% of traces with the feature present per length bin (optionally with the logistic fit)."""
    for frame, color, marker, offset, label in ((low, LOW, "o", -4, "Low effort"), (high, HIGH, "s", 4, "High effort")):
        if show_fit:
            grid, fit, fit_lo, fit_hi = logistic_curve(frame, feature)
            ax.fill_between(grid, fit_lo, fit_hi, color=color, alpha=0.15, linewidth=0, zorder=1)
            ax.plot(grid, fit, color=color, linewidth=1.3, zorder=3)
        b = binned(frame, feature)
        dense = b["n"] >= SPARSE_N
        for mask, face in ((dense, color), (~dense, "white")):
            value = b.loc[mask, "pct"]
            ax.errorbar(b.loc[mask, "x"] + offset, value, yerr=[value - b.loc[mask, "lo"], b.loc[mask, "hi"] - value],
                        fmt=marker, color=color, markerfacecolor=face, markeredgecolor=color, markersize=3.5,
                        markeredgewidth=0.8, elinewidth=0.7, capsize=1.8, capthick=0.7, zorder=4,
                        label=label if mask is dense else None)
    length_ticks = [0, 100, 200, 300, 400, 500, 600]
    ax.set_xlim(-20, X_MAX + 5)
    ax.set_xticks(length_ticks, [str(t) if t < 600 else "≥600" for t in length_ticks])
    ax.set_xlabel("Summary length (tokens)", fontsize=8.5)
    ax.set_ylim(-3, 103)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Traces with feature (%)" if show_ylabel else "", fontsize=8.5)
    if title:
        ax.set_title(title, fontsize=9.5, loc="left", pad=4)
    if letter:
        ax.text(-0.02, 1.07, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", ha="right", va="bottom")
    ax.tick_params(labelsize=7.5, direction="out", length=3, width=0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    low, high = load_condition("low"), load_condition("high")
    fig = plt.figure(figsize=(12, 3.0))
    grid = fig.add_gridspec(1, 4, wspace=0.18, left=0.055, right=0.995, top=0.82, bottom=0.16)
    or_rows = []
    for col, (feature, title) in enumerate(FEATURES):
        presence_panel(fig.add_subplot(grid[0, col]), low, high, feature, title, "abcd"[col],
                       show_ylabel=col == 0, show_fit=True)
        odds, lo, hi, p = adjusted_effort_or(low, high, feature)
        or_rows.append({"feature": feature, "odds_ratio": odds, "ci_low": lo, "ci_high": hi, "p_value": p})
    fig.legend(handles=[
        Line2D([], [], color=LOW, linewidth=0, marker="o", markersize=3.5, label="Low effort"),
        Line2D([], [], color=HIGH, linewidth=0, marker="s", markersize=3.5, label="High effort"),
    ], loc="upper center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{JUDGE_ID}_feature_presence_vs_length_overlay.png"
    fig.savefig(out, dpi=300)
    table = pd.DataFrame(or_rows)
    table.to_csv(out.with_name(f"{JUDGE_ID}_feature_presence_length_adjusted_effort_or.csv"), index=False)
    print(table.round(3).to_string(index=False))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
