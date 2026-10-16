"""Validated GPT-5.5 predefined-feature scores for cohort indices 0-999.

Indices 0-199 were scored in the first (validation) run and are stored in results.jsonl;
indices 200-999 in four 200-paper batches under cohort_batches/<range>/results.jsonl.
Every file must contain exactly its slice of the cohort, with valid 1-5 scores.
"""

import csv
import json

from common.judge_config import HERE, predefined_batch_dir, valid_scores


RANGES = ((0, 199), (200, 399), (400, 599), (600, 799), (800, 999))


def cohort_ids():
    with (HERE / "cohort.csv").open(newline="", encoding="utf-8") as handle:
        return [row["PaperID"] for row in csv.DictReader(handle)][:1000]


def score_batches(condition, judge="gpt_5_5"):
    """Yield (batch range, results file, records) for the five validated score batches."""
    ids = cohort_ids()
    for start, end in RANGES:
        if judge != "gpt_5_5":
            raise ValueError("This five-block loader is for the GPT-5.5 scores only")
        path = predefined_batch_dir(condition, start, end - start + 1) / "results.jsonl"
        with path.open(encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        if len(records) != 200 or {str(r["focal_paper_id"]) for r in records} != set(ids[start:end + 1]):
            raise ValueError(f"{path} does not contain exactly cohort indices {start}-{end}")
        if not all(valid_scores(r["classification"]) for r in records):
            raise ValueError(f"Invalid feature scores in {path}")
        yield f"{start:04d}-{end:04d}", path, records


def load_cohort_records(condition):
    """One score record per cohort paper, in cohort order."""
    records = {str(r["focal_paper_id"]): r for _, _, batch in score_batches(condition) for r in batch}
    return [records[paper_id] for paper_id in cohort_ids()]


def load_cohort_frame(condition):
    import pandas as pd
    return pd.DataFrame([{"focal_paper_id": str(r["focal_paper_id"]), "sample_index": index, **r["classification"]}
                         for index, r in enumerate(load_cohort_records(condition))])
