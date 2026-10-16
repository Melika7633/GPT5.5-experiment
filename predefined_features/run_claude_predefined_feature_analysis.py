"""Score the predefined features with Claude Opus 4.8 (Anthropic Message Batches API).

Second judge for the robustness comparison (Fig S1). Uses the same rubric as the GPT-5.5 judge
(judge_config.py) and returns the four scores through a forced tool call. All 1,000 cohort papers
of one condition are scored in one batch; papers without a summary get all four scores = 1:

    python3 -m predefined_features.run_claude_predefined_feature_analysis --condition low --dry-run
    python3 -m predefined_features.run_claude_predefined_feature_analysis --condition low --submit
    python3 -m predefined_features.run_claude_predefined_feature_analysis --condition low --status
    python3 -m predefined_features.run_claude_predefined_feature_analysis --condition low --collect

Results go to feature_analysis/predefined_features/outputs/by_judge/claude_opus_4_8/<condition>/
results.jsonl. Rerunning --submit after --collect scores only the papers still without a result
(e.g. failed requests). Needs ANTHROPIC_API_KEY.
"""

import argparse
import json
import os
import urllib.error
import urllib.request

from common.cohort_plot_data import cohort_ids
from common.judge_config import FEATURES, INSTRUCTIONS, JUDGES, SCORE_SCHEMA, judge_output_dir, reasoning_summary, valid_scores
from generation.run import load_jsonl, now, write_json, write_jsonl


JUDGE_ID = "claude_opus_4_8"
MODEL = JUDGES[JUDGE_ID]["model"]
API_URL = "https://api.anthropic.com/v1"
CLAUDE_INSTRUCTIONS = INSTRUCTIONS + " Use the record_scores tool exactly once."
TOOL = {
    "name": "record_scores",
    "description": "Record the four predefined ordinal reasoning-feature scores.",
    "input_schema": SCORE_SCHEMA,
}


def anthropic(path, payload=None):
    """Call the Anthropic API (POST if a payload is given, otherwise GET); return the body text."""
    request = urllib.request.Request(f"{API_URL}/{path}", data=None if payload is None else json.dumps(payload).encode("utf-8"),
                                     headers={"content-type": "application/json", "anthropic-version": "2023-06-01",
                                              "x-api-key": os.environ["ANTHROPIC_API_KEY"]})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Anthropic API error {exc.code}: {exc.read().decode('utf-8', errors='replace')}") from exc


def pending_papers(condition, out):
    """Cohort papers without a result yet, split into (with summary, without summary)."""
    done = {row["focal_paper_id"] for row in load_jsonl(out / "results.jsonl")}
    with_summary, without_summary = [], []
    for paper_id in cohort_ids():
        if paper_id not in done:
            summary = reasoning_summary(condition, paper_id)
            (with_summary if summary else without_summary).append((paper_id, summary))
    return with_summary, without_summary


def record(condition, paper_id, classification, message=None):
    missing = message is None
    return {
        "condition": condition,
        "focal_paper_id": paper_id,
        "prompt_id": "P00",
        "judge_provider": "protocol_assignment" if missing else "anthropic",
        "judge_id": JUDGE_ID,
        "classifier_model": None if missing else message.get("model") or MODEL,
        "scoring_prompt_version": "predefined-v1",
        "source_reasoning_record": f"outputs/{condition}/raw/{paper_id}__P00.json",
        "classified_at": now(),
        "summary_missing": missing,
        **({"score_assignment_rule": "missing summary: all predefined features scored 1 (absent)"} if missing else {}),
        "classification": classification,
        **({} if missing else {"usage": message.get("usage")}),
    }


def tool_scores(message):
    for block in message.get("content") or []:
        if block.get("type") == "tool_use" and block.get("name") == "record_scores":
            scores = block.get("input") or {}
            if not valid_scores(scores):
                raise ValueError(f"Invalid scores: {scores}")
            return scores
    raise ValueError("Claude response did not contain a record_scores tool call.")


def submit(condition, out):
    state_path = out / "batch.json"
    if state_path.is_file() and not json.loads(state_path.read_text(encoding="utf-8")).get("collected_at"):
        raise ValueError("Collect the existing Claude batch before submitting another.")
    with_summary, without_summary = pending_papers(condition, out)
    requests = [{
        "custom_id": f"{paper_id}__P00",
        "params": {
            "model": MODEL,
            "max_tokens": 512,
            "system": CLAUDE_INSTRUCTIONS,
            "messages": [{"role": "user", "content": summary}],
            "tools": [TOOL],
            "tool_choice": {"type": "tool", "name": "record_scores"},
        },
    } for paper_id, summary in with_summary]
    batch = json.loads(anthropic("messages/batches", {"requests": requests}))
    batch_dir = out / "batches" / batch["id"]
    write_json(batch_dir / "batch_request.json", {"requests": requests})
    write_json(batch_dir / "missing_summaries_scored_absent.json", [paper_id for paper_id, _ in without_summary])
    write_json(state_path, {**batch, "judge_id": JUDGE_ID, "judge_model_requested": MODEL, "condition": condition,
                            "request_count": len(requests), "missing_summary_count": len(without_summary),
                            "scoring_prompt_version": "predefined-v1", "submitted_at": now(), "collected_at": None})
    print(f"Submitted {len(requests)} Claude judge requests.\nBatch ID: {batch['id']}")


def refresh(out):
    state = json.loads((out / "batch.json").read_text(encoding="utf-8"))
    state = {**state, **json.loads(anthropic(f"messages/batches/{state['id']}")), "checked_at": now()}
    write_json(out / "batch.json", state)
    return state


def collect(condition, out):
    state = refresh(out)
    if state.get("processing_status") != "ended":
        raise ValueError(f"Claude batch is {state.get('processing_status')}; results are not ready.")
    batch_dir = out / "batches" / state["id"]
    raw_text = anthropic(f"messages/batches/{state['id']}/results")
    (batch_dir / "results_raw.jsonl").write_text(raw_text, encoding="utf-8")
    results, errors = [], []
    for item in (json.loads(line) for line in raw_text.splitlines() if line.strip()):
        paper_id = str(item.get("custom_id")).split("__", 1)[0]
        result = item.get("result") or {}
        message = result.get("message") or {}
        try:
            if result.get("type") != "succeeded":
                raise RuntimeError(f"Request {result.get('type')}: {result}")
            results.append(record(condition, paper_id, tool_scores(message), message))
        except Exception as exc:  # record the failure; a later --submit retries this paper
            errors.append({"custom_id": item.get("custom_id"), "error_message": str(exc), "message": message})
    for paper_id in json.loads((batch_dir / "missing_summaries_scored_absent.json").read_text(encoding="utf-8")):
        results.append(record(condition, paper_id, {feature: 1 for feature in FEATURES}))
    merged = {f"{row['focal_paper_id']}__{row['prompt_id']}": row for row in load_jsonl(out / "results.jsonl")}
    merged.update((f"{row['focal_paper_id']}__{row['prompt_id']}", row) for row in results)
    write_jsonl(out / "results.jsonl", [merged[key] for key in sorted(merged)])
    write_jsonl(out / "errors.jsonl", errors)
    write_json(out / "batch.json", {**state, "collected_at": now(), "completed_classification_count": len(results),
                                    "error_count": len(errors)})
    print(f"Collected {len(results)} Claude classifications with {len(errors)} errors; {len(merged)} papers are scored.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", required=True, choices=("low", "high"))
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "submit", "status", "collect"):
        action.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    out = judge_output_dir(JUDGE_ID, args.condition)
    if args.dry_run:
        with_summary, without_summary = pending_papers(args.condition, out)
        print(f"Judge: {JUDGE_ID} ({MODEL}), condition {args.condition}")
        print(f"Requests to submit: {len(with_summary)}; papers without a summary (scored 1): {len(without_summary)}")
    elif args.submit:
        submit(args.condition, out)
    elif args.status:
        state = refresh(out)
        print(json.dumps({key: state.get(key) for key in ("id", "processing_status", "request_counts")}, indent=2))
    else:
        collect(args.condition, out)


if __name__ == "__main__":
    main()
