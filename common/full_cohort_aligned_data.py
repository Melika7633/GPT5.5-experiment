"""Model-defined feature scores and the 11 constructs aligned across discovery batches and efforts.

Features were discovered separately in five 200-paper batches per effort condition, and each batch
was scored against its own frozen feature set (batch-namespaced IDs such as LOW-batch_003-F07).
full_1k_low_high_feature_pairs.json lists, for each of the 11 constructs analysed in the paper, the
matching low-effort and high-effort feature number in every batch.
"""

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parents[1]
MODEL_DEFINED_DIR = HERE / "feature_analysis" / "model_defined_features" / "model_defined_feature_scoring"
SCORE_ROOT = MODEL_DEFINED_DIR / "outputs" / "by_judge"
FIGURE_DIR = SCORE_ROOT / "gpt_5_5" / "figures"
ALIGNMENT_PATH = HERE / "model_defined_features" / "full_1k_low_high_feature_pairs.json"
BATCHES = tuple(f"batch_{number:03d}" for number in range(1, 6))


def batch_dir(judge, condition, batch):
    base = SCORE_ROOT / judge / condition / "feature_scoring"
    return base if batch == "batch_001" else base / "by_discovery_batch" / batch


def load_batches(condition):
    """GPT-5.5 judge scores per batch: {batch: (feature IDs, {paper ID: {feature ID: score}})}."""
    batches = {}
    for batch in BATCHES:
        directory = batch_dir("gpt_5_5", condition, batch)
        features = json.loads((directory / "frozen_feature_set.json").read_text(encoding="utf-8"))["features"]
        feature_ids = [str(feature["feature_id"]) for feature in features]
        with (directory / "results.jsonl").open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if len(rows) != 200 or any(set(r.get("scores") or {}) != set(feature_ids)
                                   or not all(1 <= int(v) <= 5 for v in r["scores"].values()) for r in rows):
            raise ValueError(f"{condition}/{batch}: expected 200 papers with a 1-5 score for every feature")
        papers = {str(r["focal_paper_id"]): r["scores"] for r in sorted(rows, key=lambda r: str(r["focal_paper_id"]))}
        batches[batch] = (feature_ids, papers)
    if len({paper for _, papers in batches.values() for paper in papers}) != 1000:
        raise ValueError(f"{condition}: expected 1,000 different papers across the five batches")
    return batches


def load_alignment():
    """The 11 aligned constructs; each pair maps a batch to its [low, high] feature numbers."""
    pairs = json.loads(ALIGNMENT_PATH.read_text(encoding="utf-8"))["pairs"]
    if len({pair["construct_id"] for pair in pairs}) != 11:
        raise ValueError("Expected 11 distinct constructs")
    for batch in BATCHES:
        for side in (0, 1):
            if len({pair[batch][side] for pair in pairs}) != len(pairs):
                raise ValueError(f"{batch}: a feature is mapped to more than one construct")
    return pairs


def load_aligned_scores(condition):
    """One row per paper and construct: the score of that paper's own batch-specific feature."""
    side = 0 if condition == "low" else 1
    pairs = load_alignment()
    rows = []
    for batch, (_, papers) in load_batches(condition).items():
        for paper_id, scores in papers.items():
            for pair in pairs:
                feature_id = f"{condition.upper()}-{batch}-{pair[batch][side]}"
                rows.append({"focal_paper_id": paper_id, "batch_id": batch, "construct_id": pair["construct_id"],
                             "feature_id": feature_id, "score": int(scores[feature_id])})
    return pd.DataFrame(rows)
