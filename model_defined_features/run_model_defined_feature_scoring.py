"""Score the model-defined features with the GPT-5.5 judge (OpenAI Batch API).

Each discovery batch is scored against its own discovered features only: every one of its 200
reasoning summaries is sent on its own with that batch's feature list, and the judge returns a
1-5 score per feature under a strict JSON schema. Empty summaries are sent too (both judges
scored them 1 on every feature).

    python3 -m model_defined_features.run_model_defined_feature_scoring --condition low --batch batch_002 --dry-run
    python3 -m model_defined_features.run_model_defined_feature_scoring --condition low --batch batch_002 --submit
    python3 -m model_defined_features.run_model_defined_feature_scoring --condition low --batch batch_002 --status
    python3 -m model_defined_features.run_model_defined_feature_scoring --condition low --batch batch_002 --collect

Results go to feature_analysis/model_defined_features/model_defined_feature_scoring/outputs/
by_judge/gpt_5_5/<condition>/feature_scoring/results.jsonl for batch_001 and to
.../feature_scoring/by_discovery_batch/<batch>/results.jsonl for the others, next to the scored
feature set (frozen_feature_set.json). Rerunning --submit after --collect scores only the papers
still without a result. Needs OPENAI_API_KEY.
"""

import argparse
import hashlib
import json

from common.full_cohort_aligned_data import BATCHES, batch_dir
from common.judge_config import HERE, reasoning_summary
from generation.run import (
    download_batch, ensure_no_open_batch, load_jsonl, mark_collected, now,
    response_text, show_status, submit_batch, write_json, write_jsonl,
)
from model_defined_features.run_model_defined_feature_discovery import discovery_dir


JUDGE_ID = "gpt_5_5"
MODEL = "gpt-5.5-2026-04-23"
REASONING_EFFORT = "medium"

# batch_001 was scored first, in the validation run; its prompt calls the features "frozen" where
# batches 002-005 say "batch-specific". The rubric is the same for every batch.
FIRST_RUN = {
    "use_features": "Use every supplied frozen feature exactly as written. Do not add, remove, merge,\n"
                    "split, rename, or redefine a feature.",
    "word": "frozen", "schema_name": "model_defined_feature_scores", "version": "model-defined-v1",
}
LATER_RUNS = {
    "use_features": "Use every supplied feature exactly as written. Do not add, remove, merge, split,\n"
                    "rename, or redefine a feature.",
    "word": "batch-specific", "schema_name": "batch_specific_feature_scores", "version": "model-defined-batch-specific-v1",
}
RUBRIC = """Score each feature from 1 to 5:
- 1 = Absent: desirable reasoning behavior is completely absent
- 2 = Weak: generic or superficial statements without concrete actions
- 3 = Implicit: indirect strategy adaptation or implicit expression
- 4 = Explicit: direct, explicit self-questioning, checking, or verification
- 5 = Strong & Repeated: sustained, multi-step verification and constraint checks across the trace

Higher scores must always indicate more desirable reasoning. Return only the
integer scores required by the JSON schema."""


def wording(batch):
    return FIRST_RUN if batch == "batch_001" else LATER_RUNS


def instructions(batch):
    return ("Score only what appears in the single supplied citation-generation reasoning summary. Do not use "
            "outside knowledge, compare it with other papers, or judge whether the generated references are "
            f"actually correct.\n\n{wording(batch)['use_features']} {RUBRIC}")


def feature_set(condition, batch):
    """The batch's discovered features (ID, name, definition) and papers, as scored."""
    source = discovery_dir(condition) / "parsed_batch_features" / f"{batch}.json"
    discovered = json.loads(source.read_text(encoding="utf-8"))
    features = [{"feature_id": f["local_feature_id"], "feature_name": f["feature_name"].strip(),
                 "definition": f["definition"].strip()} for f in discovered["features"]]
    digest = hashlib.sha256(json.dumps(features, ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":")).encode("utf-8")).hexdigest()
    return {"condition": condition, "discovery_batch": batch, "discovery_source": str(source.relative_to(HERE)),
            "discovery_model": MODEL, "feature_count": len(features), "features_sha256": digest,
            "paper_ids": discovered["paper_ids"], "features": features}


def score_schema(features):
    return {"type": "object",
            "properties": {f["feature_id"]: {"type": "integer", "minimum": 1, "maximum": 5} for f in features},
            "required": [f["feature_id"] for f in features],
            "additionalProperties": False}


def scoring_input(batch, features, summary):
    header = wording(batch)["word"].capitalize()
    return f"# {header} features\n\n{json.dumps(features, ensure_ascii=False, indent=2)}\n\n# Single reasoning summary\n\n{summary}"


def pending_papers(condition, features, out):
    """(paper ID, summary) for the batch's papers without a result yet."""
    done = {row["focal_paper_id"] for row in load_jsonl(out / "results.jsonl")}
    return [(paper_id, reasoning_summary(condition, paper_id)) for paper_id in features["paper_ids"] if paper_id not in done]


def score_record(condition, batch, paper_id, summary, scores, judge_fields):
    return {"condition": condition, "discovery_batch": batch, "focal_paper_id": paper_id, "prompt_id": "P00",
            "source_response_record": f"outputs/{condition}/raw/{paper_id}__P00.json", **judge_fields,
            "scoring_prompt_version": wording(batch)["version"], "scored_at": now(),
            "summary_missing": not summary, "scores": scores}


def valid(scores, features):
    return set(scores) == {f["feature_id"] for f in features} and all(type(v) is int and 1 <= v <= 5 for v in scores.values())


def merge_results(out, records):
    merged = {row["focal_paper_id"]: row for row in load_jsonl(out / "results.jsonl")}
    merged.update((row["focal_paper_id"], row) for row in records)
    write_jsonl(out / "results.jsonl", [merged[key] for key in sorted(merged)])
    return len(merged)


def submit(condition, batch, out):
    ensure_no_open_batch(out)
    scored = feature_set(condition, batch)
    pending = pending_papers(condition, scored, out)
    if not pending:
        raise ValueError(f"Every paper of {condition}/{batch} already has a result.")
    features = scored["features"]
    requests = [{
        "custom_id": f"{paper_id}__P00",
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": MODEL,
            "reasoning": {"effort": REASONING_EFFORT},
            "instructions": instructions(batch),
            "input": scoring_input(batch, features, summary),
            "text": {"format": {"type": "json_schema", "name": wording(batch)["schema_name"], "strict": True,
                                "schema": score_schema(features)}},
        },
    } for paper_id, summary in pending]
    write_json(out / "frozen_feature_set.json", scored)
    submit_batch(out, requests, "/v1/responses",
                 {"experiment": "gpt55_1k_batch_specific_feature_scoring", "condition": condition, "discovery_batch": batch},
                 {"condition": condition, "discovery_batch": batch, "stage": "feature_scoring", "model": MODEL,
                  "judge_id": JUDGE_ID, "scoring_prompt_version": wording(batch)["version"],
                  "classifier_reasoning_effort": REASONING_EFFORT, "request_count": len(requests),
                  "feature_count": len(features), "features_sha256": scored["features_sha256"]})


def collect(condition, batch, out):
    state, output = download_batch(out)
    features = json.loads((out / "frozen_feature_set.json").read_text(encoding="utf-8"))["features"]
    records, errors = [], []
    for item in output:
        paper_id = item["custom_id"].split("__", 1)[0]
        response = item.get("response") or {}
        try:
            if response.get("status_code") != 200:
                raise RuntimeError(f"HTTP {response.get('status_code')}: {response.get('body')}")
            scores = json.loads(response_text(response["body"]))
            if not valid(scores, features):
                raise ValueError(f"Invalid feature scores: {scores}")
            records.append(score_record(condition, batch, paper_id, reasoning_summary(condition, paper_id), scores,
                                        {"judge_provider": "openai", "judge_id": JUDGE_ID, "classifier_model": MODEL,
                                         "classifier_reasoning_effort": REASONING_EFFORT}))
        except Exception as exc:  # recorded; the paper is resent by the next --submit
            errors.append({"custom_id": item["custom_id"], "status": "error", "error_message": str(exc)})
    total = merge_results(out, records)
    write_jsonl(out / "collection_errors.jsonl", errors)
    mark_collected(out, state)
    print(f"Collected {len(records)} scores ({len(errors)} errors); {total} papers of {condition}/{batch} are scored.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", required=True, choices=("low", "high"))
    parser.add_argument("--batch", required=True, choices=BATCHES)
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "submit", "status", "collect"):
        action.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    out = batch_dir(JUDGE_ID, args.condition, args.batch)
    if args.dry_run:
        scored = feature_set(args.condition, args.batch)
        pending = pending_papers(args.condition, scored, out)
        print(f"Judge: {JUDGE_ID} ({MODEL}), {args.condition}/{args.batch}: {scored['feature_count']} features")
        print(f"Requests to submit: {len(pending)} ({sum(not summary for _, summary in pending)} with an empty summary)")
        print(f"Output: {out}\nNo API calls made.")
    elif args.submit:
        submit(args.condition, args.batch, out)
    elif args.status:
        show_status(out)
    else:
        collect(args.condition, args.batch, out)


if __name__ == "__main__":
    main()
