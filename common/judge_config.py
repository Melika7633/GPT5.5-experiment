"""Feature-scoring judges, the scoring rubric, and the folders holding the scores."""

import json
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
JUDGE_ROOT = HERE / "feature_analysis" / "predefined_features" / "outputs" / "by_judge"
JUDGES = {
    "gpt_5_5": {"label": "GPT-5.5 judge", "model": "gpt-5.5-2026-04-23"},
    "claude_opus_4_8": {"label": "Claude Opus 4.8 judge", "model": "claude-opus-4-8"},
}
FEATURES = (
    "epistemic_caution",
    "temporal_constraint_awareness",
    "reference_verifiability_checking",
    "metadata_verification",
)
# Scoring instructions shared by both judges (the Claude version adds a tool-use sentence).
INSTRUCTIONS = """Classify only what appears in the supplied citation-generation reasoning summary. Do not use outside knowledge or judge whether references are actually correct.

Labels:
- epistemic_caution: explicitly recognizes uncertainty and avoids assuming accuracy.
- temporal_constraint_awareness: explicitly checks that references do not postdate the focal paper. Merely selecting older or foundational work is insufficient unless the temporal constraint is recognized.
- reference_verifiability_checking: explicitly considers whether references are real and verifiable.
- metadata_verification: explicitly checks authors, titles, years, venues, or other bibliographic metadata.

Score every label from 1 to 5 based on observed behavior:
- 1 = Absent: Desirable reasoning behavior is completely absent.
- 2 = Weak: Generic or superficial statements without concrete actions.
- 3 = Implicit: Indirect strategy adaptation or implicit expression.
- 4 = Explicit: Direct, explicit self-questioning, checking, or verification.
- 5 = Strong & Repeated: Sustained, multi-step verification and constraint checks across the entire trace.

Higher scores must always indicate more desirable reasoning behavior."""
SCORE_SCHEMA = {
    "type": "object",
    "properties": {name: {"type": "integer", "minimum": 1, "maximum": 5} for name in FEATURES},
    "required": list(FEATURES),
    "additionalProperties": False,
}


def judge_output_dir(judge, condition):
    return JUDGE_ROOT / judge / condition


def predefined_batch_dir(condition, start, count=200):
    """The published GPT score layout, shared by writers and readers."""
    if count != 200 or start not in (0, 200, 400, 600, 800):
        raise ValueError("Use one published 200-paper block starting at 0, 200, 400, 600, or 800")
    base = judge_output_dir("gpt_5_5", condition)
    return base if start == 0 else base / "cohort_batches" / f"{start:04d}-{start + count - 1:04d}"


def judge_figure_dir(judge):
    return JUDGE_ROOT / judge / "figures"


def valid_scores(scores):
    """True if `scores` holds an integer 1-5 score for exactly the four features."""
    return set(scores) == set(FEATURES) and all(type(v) is int and 1 <= v <= 5 for v in scores.values())


def reasoning_summary(condition, paper_id, prompt_id="P00"):
    """The saved reasoning summary of one generation ("" if none was returned)."""
    path = HERE / "outputs" / condition / "raw" / f"{paper_id}__{prompt_id}.json"
    summary = json.loads(path.read_text(encoding="utf-8")).get("reasoning_summary")
    if not isinstance(summary, str):
        raise ValueError(f"Missing string reasoning_summary: {path}")
    summary = summary.strip()
    return "" if summary == "_No reasoning summary was returned._" else summary
