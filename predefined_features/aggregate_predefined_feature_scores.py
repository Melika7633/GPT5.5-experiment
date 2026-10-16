"""Combine the GPT-5.5 predefined-feature scores of cohort indices 0-999 into one file per condition.

Writes .../by_judge/gpt_5_5/<condition>/aggregated/cohort_0000_0999_results.jsonl (one record per
paper, in cohort order, with its source record) and a manifest with file checksums and counts.
Papers without a reasoning summary must have all four scores = 1.
"""

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone

from common.cohort_plot_data import cohort_ids, score_batches
from common.judge_config import FEATURES, HERE, judge_output_dir, reasoning_summary


MISSING_SUMMARY_PROCEDURES = {
    "api_judge_placeholder": "Indices 0-199: placeholder text was sent to the judge; returned scores were all 1.",
    "mechanical_missing_summary_rule": "Indices 200-999: no judge request; all four scores assigned 1 locally.",
}


def missing_summary_procedure(batch_range):
    return "api_judge_placeholder" if batch_range == "0000-0199" else "mechanical_missing_summary_rule"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(condition):
    index = {paper_id: i for i, paper_id in enumerate(cohort_ids())}
    rows, sources = [], []
    for batch_range, path, records in score_batches(condition):
        source = str(path.relative_to(HERE))
        sources.append({"batch_range": batch_range, "path": source, "sha256": sha256(path), "record_count": len(records)})
        for record in records:
            paper_id, prompt_id = str(record["focal_paper_id"]), str(record["prompt_id"])
            missing = reasoning_summary(condition, paper_id, prompt_id) == ""
            if missing and set(record["classification"].values()) != {1}:
                raise ValueError(f"Missing-summary record does not have all-1 scores: {paper_id}")
            rows.append({
                "cohort_index": index[paper_id],
                "batch_range": batch_range,
                "condition": condition,
                "focal_paper_id": paper_id,
                "prompt_id": prompt_id,
                "summary_missing": missing,
                "score_source": missing_summary_procedure(batch_range) if missing else "api_judge_reasoning_summary",
                "classification": record["classification"],
                "source_results_file": source,
                "source_record": record,
            })
    rows.sort(key=lambda row: row["cohort_index"])

    out = judge_output_dir("gpt_5_5", condition) / "aggregated"
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "cohort_0000_0999_results.jsonl"
    results_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    ids = {row["focal_paper_id"] for row in rows}
    manifest = {
        "artifact": "predefined_feature_score_aggregate",
        "condition": condition,
        "cohort_indices": [0, 999],
        "features": list(FEATURES),
        "score_range": [1, 5],
        "higher_score_meaning": "more strongly expressed desirable reasoning behavior",
        "judge": {"provider": "OpenAI", "family": "GPT-5.5", "model": "gpt-5.5-2026-04-23"},
        "missing_summary_count": sum(row["summary_missing"] for row in rows),
        "missing_summary_counts_by_procedure": dict(sorted(Counter(
            row["score_source"] for row in rows if row["summary_missing"]).items())),
        "missing_summary_procedures": MISSING_SUMMARY_PROCEDURES,
        "source_files": sources,
        "source_cohort": {"path": "cohort.csv", "sha256": sha256(HERE / "cohort.csv")},
        "validation": {
            "expected_count": 1000,
            "actual_count": len(rows),
            "unique_focal_paper_count": len(ids),
            "duplicate_count": len(rows) - len(ids),
            "missing_id_count": len(set(index) - ids),
            "extra_id_count": len(ids - set(index)),
            "exact_cohort_match": ids == set(index),
            "invalid_score_record_count": 0,
        },
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output": {"path": str(results_path.relative_to(HERE)), "sha256": sha256(results_path)},
    }
    manifest_path = out / "cohort_0000_0999_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{condition}: {len(rows)} records; missing summaries={manifest['missing_summary_count']}")
    print(f"Results: {results_path}\nManifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("low", "high", "both"), default="both")
    args = parser.parse_args()
    for condition in ("low", "high") if args.condition == "both" else (args.condition,):
        aggregate(condition)


if __name__ == "__main__":
    main()
