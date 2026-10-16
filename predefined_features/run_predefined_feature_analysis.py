"""Score the four predefined reasoning features with the GPT-5.5 judge (OpenAI Batch API).

Each request sends one saved reasoning summary with the scoring rubric (judge_config.py) and
asks for the four integer scores under a strict JSON schema. A paper without a summary gets all
four scores = 1 without an API call. Cohort indices 200-999 were scored in four 200-paper
batches per condition (indices 0-199 in the first, validation run; see cohort_plot_data.py):

    python3 -m predefined_features.run_predefined_feature_analysis --condition low --start 200 --count 200 --dry-run
    python3 -m predefined_features.run_predefined_feature_analysis --condition low --start 200 --count 200 --submit
    python3 -m predefined_features.run_predefined_feature_analysis --condition low --start 200 --count 200 --status
    python3 -m predefined_features.run_predefined_feature_analysis --condition low --start 200 --count 200 --collect

Results go to feature_analysis/predefined_features/outputs/by_judge/gpt_5_5/<condition>/
results.jsonl for indices 0-199, and cohort_batches/<range>/results.jsonl for later blocks. Rerunning --submit after --collect scores only the papers
of the range that still have no result. Needs OPENAI_API_KEY.
"""

import argparse
import json

from common.cohort_plot_data import cohort_ids
from common.judge_config import FEATURES, INSTRUCTIONS, JUDGES, SCORE_SCHEMA, predefined_batch_dir, reasoning_summary, valid_scores
from generation.run import (
    download_batch, ensure_no_open_batch, load_jsonl, mark_collected, now,
    response_text, show_status, submit_batch, write_json, write_jsonl,
)


MODEL = JUDGES["gpt_5_5"]["model"]
REASONING_EFFORT = "medium"
GPT_INSTRUCTIONS = INSTRUCTIONS + "\nReturn only the four integer scores."
MISSING_RULE = "missing summary: all predefined features scored 1 (absent)"


def batch_dir(condition, start, count):
    return predefined_batch_dir(condition, start, count)


def pending_papers(condition, start, count, out):
    """Papers of the range without a result yet, split into (with summary, without summary)."""
    done = {row["focal_paper_id"] for row in load_jsonl(out / "results.jsonl")}
    with_summary, without_summary = [], []
    for paper_id in cohort_ids()[start:start + count]:
        if paper_id in done:
            continue
        summary = reasoning_summary(condition, paper_id)
        (with_summary if summary else without_summary).append((paper_id, summary))
    return with_summary, without_summary


def record(condition, paper_id, classification, missing):
    return {
        "condition": condition,
        "focal_paper_id": paper_id,
        "prompt_id": "P00",
        "classifier_model": None if missing else MODEL,
        "classifier_reasoning_effort": None if missing else REASONING_EFFORT,
        "source_reasoning_record": f"outputs/{condition}/raw/{paper_id}__P00.json",
        "classified_at": now(),
        "summary_missing": missing,
        **({"score_assignment_rule": MISSING_RULE} if missing else {}),
        "classification": classification,
    }


def submit(condition, start, count, out):
    ensure_no_open_batch(out)
    with_summary, without_summary = pending_papers(condition, start, count, out)
    if not with_summary:
        raise ValueError("No API requests to submit for this range.")
    requests = [{
        "custom_id": f"{paper_id}__P00",
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": MODEL,
            "reasoning": {"effort": REASONING_EFFORT},
            "instructions": GPT_INSTRUCTIONS,
            "input": summary,
            "text": {"format": {"type": "json_schema", "name": "reasoning_classification", "strict": True,
                                "schema": SCORE_SCHEMA}},
        },
    } for paper_id, summary in with_summary]
    write_json(out / "missing_summaries_scored_absent.json", [paper_id for paper_id, _ in without_summary])
    submit_batch(out, requests, "/v1/responses",
                 {"experiment": "gpt55_1k_features", "condition": condition, "strategy": "predefined", "stage": "classify"},
                 {"condition": condition, "start": start, "count": count, "model": MODEL, "judge_id": "gpt_5_5",
                  "scoring_prompt_version": "predefined-v1", "classifier_reasoning_effort": REASONING_EFFORT,
                  "request_count": len(requests)})


def collect(condition, out):
    state, output = download_batch(out)
    results, errors = [], []
    for item in output:
        paper_id = item["custom_id"].split("__", 1)[0]
        response = item.get("response") or {}
        try:
            if response.get("status_code") != 200:
                raise RuntimeError(f"HTTP {response.get('status_code')}: {response.get('body')}")
            classification = json.loads(response_text(response["body"]))
            if not valid_scores(classification):
                raise ValueError("Invalid four-feature 1-5 classification")
            results.append(record(condition, paper_id, classification, missing=False))
        except Exception as exc:  # record the failure; a later --submit retries this paper
            errors.append({"custom_id": item["custom_id"], "status": "error", "error_message": str(exc)})
    for paper_id in json.loads((out / "missing_summaries_scored_absent.json").read_text(encoding="utf-8")):
        results.append(record(condition, paper_id, {feature: 1 for feature in FEATURES}, missing=True))
    merged = {f"{row['focal_paper_id']}__{row['prompt_id']}": row for row in load_jsonl(out / "results.jsonl")}
    merged.update((f"{row['focal_paper_id']}__{row['prompt_id']}", row) for row in results)
    write_jsonl(out / "results.jsonl", [merged[key] for key in sorted(merged)])
    write_jsonl(out / "collection_errors.jsonl", errors)
    mark_collected(out, state)
    print(f"Collected {len(results)} scores ({len(errors)} errors); {len(merged)} papers of this range are scored.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", required=True, choices=("low", "high"))
    parser.add_argument("--start", type=int, required=True, help="first cohort index")
    parser.add_argument("--count", type=int, required=True, help="number of cohort papers")
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "submit", "status", "collect"):
        action.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    if args.start < 0 or len(cohort_ids()[args.start:args.start + args.count]) != args.count or args.count < 1:
        parser.error("the range must lie within cohort indices 0-999")
    if args.start not in (0, 200, 400, 600, 800) or args.count != 200:
        parser.error("use a 200-paper block starting at 0, 200, 400, 600, or 800")
    out = batch_dir(args.condition, args.start, args.count)
    if args.dry_run:
        with_summary, without_summary = pending_papers(args.condition, args.start, args.count, out)
        print(f"Judge: gpt_5_5 ({MODEL}), condition {args.condition}, indices {args.start}-{args.start + args.count - 1}")
        print(f"Requests to submit: {len(with_summary)}; papers without a summary (scored 1): {len(without_summary)}")
        print(f"Output: {out}\nNo API calls made.")
    elif args.submit:
        submit(args.condition, args.start, args.count, out)
    elif args.status:
        show_status(out)
    else:
        collect(args.condition, out)


if __name__ == "__main__":
    main()
