#!/usr/bin/env python3
"""Fig S1: robustness of the feature scores to the judge (GPT-5.5 vs Claude Opus 4.8).

(A) Mean score of the four predefined features, by judge and effort condition.
(B) Papers in which each of the 11 aligned model-defined constructs is present (score >= 3),
    by judge and effort condition.
The GPT-5.5 judge is drawn filled and the Claude judge open; every comparison is restricted to
papers scored by both judges in that condition.
Output: outputs/final_figures/frontiers/FigS1.{tif,pdf,png} and
feature_analysis/outputs/judge_comparison/judge_robustness_stats.json (agreement statistics,
effort shifts and score-outcome correlations per judge).
"""

import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common.cohort_plot_data import load_cohort_records
from common.figure_style import apply_print_style, emphasize_x_axis, panel_letter
from common.frontiers_figures import EFFORT_COLORS, export
from common.full_cohort_aligned_data import BATCHES, batch_dir, load_alignment
from common.judge_config import HERE, judge_output_dir
from common.outcome_data import load_paper_metrics
from model_defined_features.plot_model_defined_common_feature_two_panel import SHORT_LABELS


JUDGES = (("gpt_5_5", "GPT-5.5"), ("claude_opus_4_8", "Claude Opus 4.8"))
CONDITIONS = ("low", "high")
PRESENCE_THRESHOLD = 3
PREDEFINED = (
    ("epistemic_caution", "Epistemic\ncaution"),
    ("temporal_constraint_awareness", "Temporal\nconstraint\nawareness"),
    ("reference_verifiability_checking", "Reference\nverifiability\nchecking"),
    ("metadata_verification", "Metadata\nverification"),
)
STATS_PATH = HERE / "feature_analysis" / "outputs" / "judge_comparison" / "judge_robustness_stats.json"


def read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_predefined(judge, condition):
    """Predefined-feature scores per paper: GPT-5.5 for cohort 0-999, Claude for the papers it scored."""
    records = (load_cohort_records(condition) if judge == "gpt_5_5"
               else read_jsonl(judge_output_dir(judge, condition) / "results.jsonl"))
    return {r["focal_paper_id"]: r["classification"] for r in records}


def load_model_defined(judge, condition):
    """Model-defined feature scores per paper, merged over the five discovery batches."""
    merged = {}
    for batch in BATCHES:
        path = batch_dir(judge, condition, batch) / "results.jsonl"
        if path.exists():
            for r in read_jsonl(path):
                merged.setdefault(r["focal_paper_id"], {}).update(r["scores"])
    return merged


def average_ranks(x):
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = average_ranks(a), average_ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def quadratic_weighted_kappa(a, b, levels=(1, 2, 3, 4, 5)):
    k = len(levels)
    index = {v: i for i, v in enumerate(levels)}
    observed = np.zeros((k, k))
    for x, y in zip(a, b):
        observed[index[int(x)], index[int(y)]] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    weights = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    denom = (weights * expected).sum()
    return float(1 - (weights * observed).sum() / denom) if denom else float("nan")


def construct_scores(judge, condition, alignment):
    """Paper x construct score table, taking each paper's feature from its own batch."""
    side = 0 if condition == "low" else 1
    rows = {}
    for paper_id, paper_scores in load_model_defined(judge, condition).items():
        batch = next((b for b in BATCHES if any(k.startswith(f"{condition.upper()}-{b}-") for k in paper_scores)), None)
        if batch is not None:
            rows[paper_id] = {pair["construct_id"]: paper_scores.get(f"{condition.upper()}-{batch}-{pair[batch][side]}")
                              for pair in alignment}
    return pd.DataFrame.from_dict(rows, orient="index").dropna().astype(int)


def agreement(a, b):
    return {
        "n_items": int(len(a)),
        "exact_agreement": float((a == b).mean()),
        "within_one": float((np.abs(a - b) <= 1).mean()),
        "spearman_rho": spearman(a, b),
        "quadratic_weighted_kappa": quadratic_weighted_kappa(a, b),
        "presence_agreement": float(((a >= PRESENCE_THRESHOLD) == (b >= PRESENCE_THRESHOLD)).mean()),
        "mean_gpt_minus_claude": float((a - b).mean()),
    }


def load_data():
    alignment = load_alignment()
    predef, constructs, stats = {}, {}, {"predefined": {}, "constructs": {}}
    for condition in CONDITIONS:
        frames = {j: pd.DataFrame.from_dict(load_predefined(j, condition), orient="index") for j, _ in JUDGES}
        shared = sorted(set(frames["gpt_5_5"].index) & set(frames["claude_opus_4_8"].index))
        predef[condition] = {j: frames[j].loc[shared] for j, _ in JUDGES}
        cframes = {j: construct_scores(j, condition, alignment) for j, _ in JUDGES}
        shared_c = sorted(set(cframes["gpt_5_5"].index) & set(cframes["claude_opus_4_8"].index))
        constructs[condition] = {j: cframes[j].loc[shared_c] for j, _ in JUDGES}
        for key, store, names in (("predefined", predef, [f for f, _ in PREDEFINED]),
                                  ("constructs", constructs, [p["construct_id"] for p in alignment])):
            g = store[condition]["gpt_5_5"][names].to_numpy().ravel()
            c = store[condition]["claude_opus_4_8"][names].to_numpy().ravel()
            stats[key][condition] = {"n_papers": len(store[condition]["gpt_5_5"]), "pooled": agreement(g, c),
                                     "by_feature": {}}
            for name in names:
                ga = store[condition]["gpt_5_5"][name].to_numpy()
                ca = store[condition]["claude_opus_4_8"][name].to_numpy()
                stats[key][condition]["by_feature"][name] = {
                    **agreement(ga, ca),
                    "mean_gpt": float(ga.mean()), "mean_claude": float(ca.mean()),
                    "present_pct_gpt": float((ga >= PRESENCE_THRESHOLD).mean() * 100),
                    "present_pct_claude": float((ca >= PRESENCE_THRESHOLD).mean() * 100),
                }
    return alignment, predef, constructs, stats


def effort_shift_stats(alignment, predef, stats):
    """High - low differences per judge and whether both judges agree in direction; and the
    within-condition Spearman correlation of predefined scores with the verified-reference rate."""
    for key, names in (("predefined", [f for f, _ in PREDEFINED]), ("constructs", [p["construct_id"] for p in alignment])):
        shifts = {}
        for name in names:
            row = {}
            for judge, _ in JUDGES:
                suffix = "gpt" if judge == "gpt_5_5" else "claude"
                low, high = stats[key]["low"]["by_feature"][name], stats[key]["high"]["by_feature"][name]
                row[judge] = {"delta_mean": high[f"mean_{suffix}"] - low[f"mean_{suffix}"],
                              "delta_present_pct": high[f"present_pct_{suffix}"] - low[f"present_pct_{suffix}"]}
            metric = "delta_mean" if key == "predefined" else "delta_present_pct"
            row["same_direction"] = bool(np.sign(row["gpt_5_5"][metric]) == np.sign(row["claude_opus_4_8"][metric]))
            shifts[name] = row
        stats[key]["effort_shift"] = shifts
    assoc = {}
    for condition in CONDITIONS:
        metrics = load_paper_metrics(condition)
        rate = metrics.set_index(metrics["focal_paper_id"].astype(str))["existence_rate"]
        for judge, _ in JUDGES:
            frame = predef[condition][judge]
            common = [i for i in frame.index if i in rate.index]
            for name, _ in PREDEFINED:
                assoc[f"{condition}|{judge}|{name}"] = spearman(frame.loc[common, name].to_numpy(dtype=float),
                                                                rate.loc[common].to_numpy(dtype=float))
    stats["predefined"]["spearman_score_vs_verified_rate"] = assoc


def panel_a(ax, predef):
    x = np.arange(len(PREDEFINED))
    width = 0.19
    slots = [("low", "gpt_5_5"), ("low", "claude_opus_4_8"), ("high", "gpt_5_5"), ("high", "claude_opus_4_8")]
    for k, (condition, judge) in enumerate(slots):
        frame = predef[condition][judge]
        means = [frame[f].mean() for f, _ in PREDEFINED]
        ci = [1.96 * frame[f].std(ddof=1) / np.sqrt(len(frame)) for f, _ in PREDEFINED]
        colour = EFFORT_COLORS[condition]
        offset = (k - 1.5) * width + (0.03 if k >= 2 else -0.03)
        ax.bar(x + offset, means, width, yerr=ci, capsize=1.6, color=colour if judge == "gpt_5_5" else "white",
               edgecolor=colour, linewidth=0.9, error_kw={"lw": 0.7, "ecolor": "#374151", "capthick": 0.7})
    ax.set_xticks(x, [label for _, label in PREDEFINED])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(1, 3.6)
    ax.set_yticks([1, 1.5, 2, 2.5, 3, 3.5])
    ax.set_ylabel("Mean feature score (1–5)")
    ax.grid(axis="y", alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    emphasize_x_axis(ax)
    panel_letter(ax, "(A)", x=-0.14, y=1.03)


def panel_b(ax, alignment, constructs):
    gpt_counts = {p["construct_id"]: max((constructs[c]["gpt_5_5"][p["construct_id"]] >= PRESENCE_THRESHOLD).mean()
                                         for c in CONDITIONS) for p in alignment}
    # Same ordering as Fig 4: ascending by the larger GPT-5.5 prevalence, bottom to top.
    order = sorted(alignment, key=lambda p: gpt_counts[p["construct_id"]])
    offsets = {"low": -0.18, "high": 0.18}
    for yi, pair in enumerate(order):
        cid = pair["construct_id"]
        for condition in CONDITIONS:
            colour = EFFORT_COLORS[condition]
            g = (constructs[condition]["gpt_5_5"][cid] >= PRESENCE_THRESHOLD).mean() * 100
            c = (constructs[condition]["claude_opus_4_8"][cid] >= PRESENCE_THRESHOLD).mean() * 100
            y = yi + offsets[condition]
            ax.plot([g, c], [y, y], color="#94A3B8", linewidth=1.2, zorder=1, solid_capstyle="butt")
            # Open ring on top, so a coinciding filled GPT-5.5 dot stays visible.
            ax.scatter([g], [y], s=26, color=colour, edgecolor=colour, linewidth=0.8, zorder=3)
            ax.scatter([c], [y], s=26, facecolor="none", edgecolor=colour, linewidth=1.0, zorder=4)
    ax.set_yticks(np.arange(len(order)), [SHORT_LABELS.get(p["label"], p["label"]) for p in order])
    ax.tick_params(axis="y", length=0, pad=4)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Papers with construct present (%)")
    for yi in range(len(order) - 1):
        ax.axhline(yi + 0.5, color="#F1F5F9", linewidth=0.6, zorder=0)
    ax.grid(axis="x", alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    emphasize_x_axis(ax)
    panel_letter(ax, "(B)", x=-0.52, y=1.03)


def main():
    apply_print_style()
    plt.rcParams["grid.color"] = "#F1F5F9"
    alignment, predef, constructs, stats = load_data()
    effort_shift_stats(alignment, predef, stats)

    fig = plt.figure(figsize=(7.4, 4.0), dpi=300)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.55)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    panel_a(ax_a, predef)
    panel_b(ax_b, alignment, constructs)
    fig.legend(handles=[
        Patch(facecolor=EFFORT_COLORS["low"], edgecolor=EFFORT_COLORS["low"], label="Low effort"),
        Patch(facecolor=EFFORT_COLORS["high"], edgecolor=EFFORT_COLORS["high"], label="High effort"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#6B7280", markeredgecolor="#6B7280",
               markersize=5, label="GPT-5.5 judge (filled)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor="#6B7280",
               markersize=5, label="Claude Opus 4.8 judge (open)"),
    ], loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False, handlelength=1.2, columnspacing=1.4)
    fig.subplots_adjust(left=0.09, right=0.99, top=0.86, bottom=0.2)

    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    export(fig, "FigS1")
    plt.close(fig)
    print(f"Saved: {STATS_PATH}")


if __name__ == "__main__":
    main()
