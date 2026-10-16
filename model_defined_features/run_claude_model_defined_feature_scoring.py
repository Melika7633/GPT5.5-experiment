"""Score the model-defined features with Claude Opus 4.8 (Anthropic Message Batches API).

Second judge for the robustness comparison (Fig S1). Uses the same prompts, feature lists and
papers as the GPT-5.5 judge (run_model_defined_feature_scoring.py) and returns the scores
through a forced tool call:

    python3 -m model_defined_features.run_claude_model_defined_feature_scoring --condition low --batch batch_002 --dry-run
    python3 -m model_defined_features.run_claude_model_defined_feature_scoring --condition low --batch batch_002 --submit
    python3 -m model_defined_features.run_claude_model_defined_feature_scoring --condition low --batch batch_002 --status
    python3 -m model_defined_features.run_claude_model_defined_feature_scoring --condition low --batch batch_002 --collect

Results go to feature_analysis/model_defined_features/model_defined_feature_scoring/outputs/
by_judge/claude_opus_4_8/<condition>/feature_scoring/[by_discovery_batch/<batch>/]results.jsonl.
Rerunning --submit after --collect scores only the papers still without a result (e.g. failed
requests). Needs ANTHROPIC_API_KEY.
"""

import argparse
import json

from common.full_cohort_aligned_data import BATCHES, batch_dir
from common.judge_config import JUDGES, reasoning_summary
from generation.run import now, write_json, write_jsonl
from predefined_features.run_claude_predefined_feature_analysis import anthropic
from model_defined_features.run_model_defined_feature_scoring import (
    feature_set, instructions, merge_results, pending_papers, score_record, score_schema, scoring_input, valid, wording,
)


JUDGE_ID = "claude_opus_4_8"
MODEL = JUDGES[JUDGE_ID]["model"]


def tool_scores(message, features):
    for block in message.get("content") or []:
        if block.get("type") == "tool_use" and block.get("name") == "record_scores":
            scores = block.get("input") or {}
            if not valid(scores, features):
                raise ValueError(f"Invalid feature scores: {scores}")
            return scores
    raise ValueError("Claude response did not contain a record_scores tool call.")


def submit(condition, batch, out):
    state_path = out / "batch.json"
    if state_path.is_file() and not json.loads(state_path.read_text(encoding="utf-8")).get("collected_at"):
        raise ValueError("Collect the existing Claude batch before submitting another.")
    scored = feature_set(condition, batch)
    pending = pending_papers(condition, scored, out)
    if not pending:
        raise ValueError(f"Every paper of {condition}/{batch} already has a result.")
    features = scored["features"]
    tool = {"name": "record_scores",
            "description": f"Record ordinal scores for every {wording(batch)['word']} reasoning feature.",
            "input_schema": score_schema(features)}
    requests = [{
        "custom_id": f"{paper_id}__P00",
        "params": {
            "model": MODEL,
            "max_tokens": max(1024, 64 * len(features)),
            "system": instructions(batch) + "\nUse the record_scores tool exactly once.",
            "messages": [{"role": "user", "content": scoring_input(batch, features, summary)}],
            "tools": [tool],
            "tool_choice": {"type": "tool", "name": "record_scores"},
        },
    } for paper_id, summary in pending]
    batch_state = json.loads(anthropic("messages/batches", {"requests": requests}))
    write_json(out / "frozen_feature_set.json", scored)
    write_json(out / "batches" / batch_state["id"] / "batch_request.json", {"requests": requests})
    write_json(state_path, {**batch_state, "condition": condition, "discovery_batch": batch, "judge_id": JUDGE_ID,
                            "judge_model_requested": MODEL, "scoring_prompt_version": wording(batch)["version"],
                            "request_count": len(requests), "feature_count": len(features),
                            "submitted_at": now(), "collected_at": None})
    print(f"Submitted {len(requests)} Claude judge requests.\nBatch ID: {batch_state['id']}")


def refresh(out):
    state = json.loads((out / "batch.json").read_text(encoding="utf-8"))
    state = {**state, **json.loads(anthropic(f"messages/batches/{state['id']}")), "checked_at": now()}
    write_json(out / "batch.json", state)
    return state


def collect(condition, batch, out):
    state = refresh(out)
    if state.get("processing_status") != "ended":
        raise ValueError(f"Claude batch is {state.get('processing_status')}; results are not ready.")
    raw_text = anthropic(f"messages/batches/{state['id']}/results")
    (out / "batches" / state["id"] / "results_raw.jsonl").write_text(raw_text, encoding="utf-8")
    features = json.loads((out / "frozen_feature_set.json").read_text(encoding="utf-8"))["features"]
    records, errors = [], []
    for item in (json.loads(line) for line in raw_text.splitlines() if line.strip()):
        paper_id = str(item.get("custom_id")).split("__", 1)[0]
        result = item.get("result") or {}
        message = result.get("message") or {}
        try:
            if result.get("type") != "succeeded":
                raise RuntimeError(f"Request {result.get('type')}: {result}")
            scores = tool_scores(message, features)
            records.append({**score_record(condition, batch, paper_id, reasoning_summary(condition, paper_id), scores,
                                           {"judge_provider": "anthropic", "judge_id": JUDGE_ID,
                                            "classifier_model": message.get("model") or MODEL}),
                            "usage": message.get("usage")})
        except Exception as exc:  # recorded; the paper is resent by the next --submit
            errors.append({"custom_id": item.get("custom_id"), "error_message": str(exc), "message": message})
    total = merge_results(out, records)
    write_jsonl(out / "errors.jsonl", errors)
    write_json(out / "batch.json", {**state, "collected_at": now(), "completed_score_count": len(records),
                                    "error_count": len(errors)})
    print(f"Collected {len(records)} Claude scores ({len(errors)} errors); {total} papers of {condition}/{batch} are scored.")


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
    elif args.submit:
        submit(args.condition, args.batch, out)
    elif args.status:
        state = refresh(out)
        print(json.dumps({key: state.get(key) for key in ("id", "processing_status", "request_counts")}, indent=2))
    else:
        collect(args.condition, args.batch, out)


if __name__ == "__main__":
    main()
