"""Discover model-defined reasoning features with GPT-5.5 (OpenAI Batch API).

The 1,000 reasoning summaries of one effort condition are split into five fixed 200-paper
discovery batches, and GPT-5.5 is asked once per batch to name and define the reasoning features
present in that batch's summaries. batch_001 holds cohort indices 0-199 (discovered first, in the
validation run) and batch_002-batch_005 hold indices 200-999; within each part the papers are in
focal-paper-ID order. Papers without a summary belong to their batch but add no text.

    python3 -m model_defined_features.run_model_defined_feature_discovery --condition low --dry-run
    python3 -m model_defined_features.run_model_defined_feature_discovery --condition low --submit
    python3 -m model_defined_features.run_model_defined_feature_discovery --condition low --status
    python3 -m model_defined_features.run_model_defined_feature_discovery --condition low --collect

--submit sends every discovery batch that has no collected feature list yet. Each batch's
features get IDs such as LOW-batch_003-F07 and are written to feature_analysis/model_defined_features/
model_defined_feature_scoring/outputs/by_judge/gpt_5_5/<condition>/feature_discovery/
parsed_batch_features/<batch>.json. Needs OPENAI_API_KEY.
"""

import argparse
import json
import math

from common.cohort_plot_data import cohort_ids
from common.full_cohort_aligned_data import BATCHES, SCORE_ROOT
from common.judge_config import reasoning_summary
from generation.run import (
    download_batch, ensure_no_open_batch, load_jsonl, mark_collected, now,
    response_text, show_status, submit_batch, write_json, write_jsonl,
)


MODEL = "gpt-5.5-2026-04-23"
REASONING_EFFORT = "medium"
BATCH_SIZE = 200
# Each request stays below the 272,000-token long-context price threshold (tokens estimated as UTF-8 bytes / 3).
INPUT_TOKEN_LIMIT = 271_999

INSTRUCTIONS = """You are given a batch of reasoning summaries produced while generating scholarly references for focal papers.

Identify the reasoning features present in the supplied summaries.

For every feature, provide a concise name and an operational definition."""

SCHEMA = {
    "type": "object",
    "properties": {
        "features": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"feature_name": {"type": "string"}, "definition": {"type": "string"}},
                "required": ["feature_name", "definition"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["features"],
    "additionalProperties": False,
}


def discovery_dir(condition):
    return SCORE_ROOT / "gpt_5_5" / condition / "feature_discovery"


def batch_papers():
    """{batch: paper IDs}: cohort indices 0-199 form batch_001, indices 200-999 batch_002-batch_005."""
    ids = cohort_ids()
    rest = sorted(ids[BATCH_SIZE:])
    return {"batch_001": sorted(ids[:BATCH_SIZE]),
            **{batch: rest[i * BATCH_SIZE:(i + 1) * BATCH_SIZE] for i, batch in enumerate(BATCHES[1:])}}


def estimate_tokens(text):
    return max(1, math.ceil(len(text.encode("utf-8")) / 3))


def build_requests(condition):
    """Requests and manifest entries for the batches that have no collected feature list yet."""
    out = discovery_dir(condition)
    requests, manifest = [], []
    for batch, paper_ids in batch_papers().items():
        if (out / "parsed_batch_features" / f"{batch}.json").is_file():
            continue
        summaries = {paper_id: reasoning_summary(condition, paper_id) for paper_id in paper_ids}
        text = "\n\n---\n\n".join(f"## Focal paper ID: {paper_id}\n\n{summary}"
                                  for paper_id, summary in summaries.items() if summary)
        tokens = estimate_tokens(INSTRUCTIONS) + estimate_tokens(text)
        if tokens > INPUT_TOKEN_LIMIT:
            raise ValueError(f"{batch} is estimated at {tokens} input tokens, above the {INPUT_TOKEN_LIMIT}-token limit.")
        manifest.append({"batch_id": batch, "paper_count": len(paper_ids),
                         "summaries_present": sum(bool(summary) for summary in summaries.values()),
                         "estimated_input_tokens": tokens, "paper_ids": paper_ids})
        requests.append({
            "custom_id": f"{condition}__discover__{batch}",
            "method": "POST",
            "url": "/v1/responses",
            "body": {
                "model": MODEL,
                "reasoning": {"effort": REASONING_EFFORT},
                "instructions": INSTRUCTIONS,
                "input": text,
                "text": {"format": {"type": "json_schema", "name": "batch_feature_discovery", "strict": True,
                                    "schema": SCHEMA}},
            },
        })
    return requests, manifest


def submit(condition, out):
    ensure_no_open_batch(out)
    requests, manifest = build_requests(condition)
    if not requests:
        raise ValueError("Every discovery batch already has a collected feature list.")
    write_json(out / "batch_manifest.json", {
        "condition": condition, "batch_size": BATCH_SIZE, "token_count_method": "conservative_utf8_bytes_divided_by_3",
        "effective_input_budget_tokens": INPUT_TOKEN_LIMIT, "created_at": now(), "batches": manifest,
    })
    submit_batch(out, requests, "/v1/responses",
                 {"experiment": "gpt55_1k_feature_discovery", "condition": condition},
                 {"condition": condition, "stage": "feature_discovery", "model": MODEL,
                  "classifier_reasoning_effort": REASONING_EFFORT, "request_count": len(requests)})


def collect(condition, out):
    state, output = download_batch(out)
    papers = batch_papers()
    results = {row["batch_id"]: row for row in load_jsonl(out / "results.jsonl")}
    for item in output:
        batch = item["custom_id"].rsplit("__", 1)[-1]
        response = item.get("response") or {}
        try:
            if response.get("status_code") != 200:
                raise RuntimeError(f"HTTP {response.get('status_code')}: {response.get('body')}")
            write_json(out / "raw_model_outputs" / f"{batch}.json", response["body"])
            features = json.loads(response_text(response["body"]))["features"]
            for index, feature in enumerate(features, start=1):
                if not feature["feature_name"].strip() or not feature["definition"].strip():
                    raise ValueError(f"feature {index} has an empty name or definition")
                feature["local_feature_id"] = f"{condition.upper()}-{batch}-F{index:02d}"
            write_json(out / "parsed_batch_features" / f"{batch}.json",
                       {"condition": condition, "batch_id": batch, "paper_ids": papers[batch],
                        "feature_count": len(features), "features": features})
            results[batch] = {"batch_id": batch, "status": "completed", "feature_count": len(features)}
        except Exception as exc:  # recorded; the batch is resent by the next --submit
            results[batch] = {"batch_id": batch, "status": "error", "error_message": str(exc)}
    write_jsonl(out / "results.jsonl", [results[key] for key in sorted(results)])
    mark_collected(out, state)
    for batch in sorted(results):
        print(f"{batch}: {results[batch]['status']} ({results[batch].get('feature_count', 0)} features)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", required=True, choices=("low", "high"))
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "submit", "status", "collect"):
        action.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args()
    out = discovery_dir(args.condition)
    if args.dry_run:
        for entry in build_requests(args.condition)[1]:
            print(f"{entry['batch_id']}: {entry['paper_count']} papers, {entry['summaries_present']} summaries, "
                  f"~{entry['estimated_input_tokens']} input tokens")
        print("No API calls made.")
    elif args.submit:
        submit(args.condition, out)
    elif args.status:
        show_status(out)
    else:
        collect(args.condition, out)


if __name__ == "__main__":
    main()
