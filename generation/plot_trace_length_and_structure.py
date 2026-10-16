#!/usr/bin/env python3
"""Distributions of saved reasoning-summary length, reasoning tokens and summary blocks.

load_condition() also supplies the per-paper summary lengths used by Fig 2. The figure covers
the 1,000 matched low/high GPT-5.5 responses; missing summaries count as zero words and blocks.
Writes outputs/final_figures/fig_trace_length_and_structure_distributions.{png,csv}.
"""

import json
import re
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tiktoken
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[1] / "outputs"
OUTDIR = ROOT / "final_figures"
CONDITIONS = ("low", "high")
LABELS = {"low": "Low effort", "high": "High effort"}
COLORS = {"low": "#4C78A8", "high": "#E45756"}
# Count Unicode alphanumeric word runs; underscores delimit words.
WORD_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)


def fixed(value, places=2):
    quantum = Decimal(1).scaleb(-places)
    return str(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def reasoning_blocks(payload):
    blocks = []
    output = payload.get("openai_response", {}).get("output", [])
    for item in output:
        if item.get("type") != "reasoning":
            continue
        for block in item.get("summary", []) or []:
            text = block.get("text", "") if isinstance(block, dict) else ""
            if isinstance(text, str) and text.strip():
                blocks.append(text.strip())
    return blocks


def load_condition(condition, paper_ids=None):
    raw_dir = ROOT / condition / "raw"
    files = sorted(raw_dir.glob("*.json"))
    if paper_ids is not None:
        wanted = {str(p) for p in paper_ids}
        files = [path for path in files if path.stem.split("__")[0] in wanted]
    if len(files) != 1000:
        raise ValueError(f"Expected 1000 raw {condition} records; found {len(files)}")

    records = {}
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        paper_id = str(payload["focal_paper_id"])
        blocks = reasoning_blocks(payload)
        summary = payload.get("reasoning_summary") or ""
        if blocks and summary.strip() != "\n\n".join(blocks).strip():
            raise ValueError(f"Saved summary and API blocks disagree: {path}")
        usage = payload.get("openai_response", {}).get("usage", {})
        details = usage.get("output_tokens_details", {})
        model = payload.get("openai_response", {}).get("model") or payload.get("model_name")
        if not model:
            raise ValueError(f"Raw record has no model name: {path}")
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            # Keep newer model names compatible with the project's o200k_base convention.
            encoding = tiktoken.get_encoding("o200k_base")
        records[paper_id] = {
            # Match the established project summary: word-like Unicode tokens.
            "words": len(WORD_RE.findall(summary)),
            # Tokenize the visible saved summary itself. This is distinct from
            # usage.output_tokens_details.reasoning_tokens, which reports
            # internal reasoning compute rather than summary-text length.
            "summary_tokens": len(encoding.encode(summary)),
            "characters": len(summary),
            "blocks": len(blocks),
            "reasoning_tokens": int(details.get("reasoning_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
        }
    if paper_ids is not None and set(records) != wanted:
        raise ValueError(f"Raw {condition} records do not match the requested cohort IDs")
    return records


def values(data, condition, key):
    return np.asarray([row[key] for row in data[condition].values()], dtype=float)


def density_curve(array, grid, bandwidth=None):
    return gaussian_kde(array, bw_method=bandwidth)(grid)


def draw_density(ax, arrays, grid, title, xlabel, tick_values=None, tick_labels=None,
                 bandwidth=None):
    curves = []
    for condition, array in zip(CONDITIONS, arrays):
        density = density_curve(array, grid, bandwidth=bandwidth)
        curves.append(density)
        ax.fill_between(grid, density, color=COLORS[condition], alpha=0.18)
        ax.plot(grid, density, color=COLORS[condition], linewidth=2.5)
        peak = int(np.argmax(density))
        ax.text(grid[peak], density[peak] * 1.07, LABELS[condition],
                color=COLORS[condition], ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.grid(axis="y", alpha=0.20)
    ax.tick_params(axis="y", labelleft=False)
    if tick_values is not None:
        ax.set_xticks(tick_values, tick_labels)
    ax.set_ylim(0, max(np.max(curve) for curve in curves) * 1.28)


def main():
    data = {condition: load_condition(condition) for condition in CONDITIONS}
    if set(data["low"]) != set(data["high"]):
        raise ValueError("Low/high raw records are not the same 1000 focal papers")

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.titlesize": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(3, 1, figsize=(11.8, 10.2))
    ax_words, ax_reasoning, ax_blocks = axes

    # A. Zero-inclusive saved-summary word counts.
    word_arrays = [values(data, c, "words") for c in CONDITIONS]
    word_grid = np.linspace(0, max(a.max() for a in word_arrays), 700)
    draw_density(
        ax_words, word_arrays, word_grid,
        "A  Distribution of saved-summary word counts (zeros retained)",
        "Saved-summary words per paper",
    )
    word_notes = []
    for condition in CONDITIONS:
        array = values(data, condition, "words")
        present = int(np.count_nonzero(array))
        nonempty = array[array > 0]
        q1, q3 = np.quantile(nonempty, [0.25, 0.75])
        chars = values(data, condition, "characters")
        word_notes.append(
            f"{LABELS[condition]}: present {present}/{len(array)}; mean {array.mean():.1f}; "
            f"median {np.median(array):.0f}; IQR {q1:.0f}–{q3:.0f}; "
            f"mean characters {chars.mean():,.0f}"
        )
    ax_words.text(0.38, 0.94, "\n".join(word_notes), transform=ax_words.transAxes,
                  ha="left", va="top", fontsize=8.2)

    # B. Reasoning-token density on log10(tokens + 1), needed for the 121-fold gap.
    raw_reasoning = [values(data, c, "reasoning_tokens") for c in CONDITIONS]
    log_reasoning = [np.log10(a + 1) for a in raw_reasoning]
    reasoning_grid = np.linspace(0, max(a.max() for a in log_reasoning), 700)
    token_ticks = np.asarray([0, 10, 100, 1_000, 10_000, 20_000])
    draw_density(
        ax_reasoning, log_reasoning, reasoning_grid,
        "B  Distribution of API-reported reasoning tokens",
        "Reasoning tokens per paper (log scale)",
        tick_values=np.log10(token_ticks + 1),
        tick_labels=[f"{v:,}" for v in token_ticks],
    )
    token_note = "   |   ".join(
        f"{LABELS[c]}: mean {raw_reasoning[i].mean():,.1f}; median {np.median(raw_reasoning[i]):,.0f}"
        for i, c in enumerate(CONDITIONS)
    )
    ax_reasoning.text(0.02, 0.91, token_note, transform=ax_reasoning.transAxes,
                      ha="left", va="top", fontsize=8.7)

    # C. Smoothed view of the discrete block-count distribution.
    max_blocks = int(max(values(data, c, "blocks").max() for c in CONDITIONS))
    block_arrays = [values(data, c, "blocks") for c in CONDITIONS]
    block_grid = np.linspace(0, max_blocks, 600)
    draw_density(
        ax_blocks, block_arrays, block_grid,
        "C  Distribution of nonempty saved-summary blocks",
        "Number of summary blocks per paper",
        tick_values=np.arange(max_blocks + 1),
        tick_labels=[str(v) for v in range(max_blocks + 1)],
        bandwidth=0.30,
    )
    block_note = []
    for condition in CONDITIONS:
        b = values(data, condition, "blocks")
        nonempty = b[b > 0]
        block_note.append(
            f"{LABELS[condition]}: total {b.sum():.0f}; mean/paper {fixed(b.mean())}; "
            f"when present {fixed(nonempty.mean())} (median {np.median(nonempty):.0f})"
        )
    ax_blocks.text(0.48, 0.91, "\n".join(block_note), transform=ax_blocks.transAxes,
                   ha="left", va="top", fontsize=8.7)

    fig.suptitle("Distributions of saved reasoning-summary length and structure", y=0.985,
                 fontweight="bold")
    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.085, right=0.98, hspace=0.48)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    stem = OUTDIR / "fig_trace_length_and_structure_distributions"
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    plt.close(fig)

    # Machine-readable values used for the figure.
    rows = ["condition,paper_id,summary_words,summary_tokens,summary_characters,summary_blocks,reasoning_tokens,total_output_tokens"]
    for condition in CONDITIONS:
        for paper_id, record in sorted(data[condition].items()):
            rows.append(
                f"{condition},{paper_id},{int(record['words'])},{int(record['summary_tokens'])},"
                f"{int(record['characters'])},"
                f"{int(record['blocks'])},{int(record['reasoning_tokens'])},{int(record['output_tokens'])}"
            )
    stem.with_suffix(".csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(stem.with_suffix(".png"))
    print(stem.with_suffix(".csv"))


if __name__ == "__main__":
    main()
