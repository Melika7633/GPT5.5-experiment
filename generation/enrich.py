"""Extract metadata from the generated citations with GPT-4.1 mini via the OpenAI Batch API.

One request per focal paper. The extracted authors, title, year, venue, DOI and arXiv ID are
merged into outputs/<condition>/parsed/*.json (see enrich_parsed_reference_metadata.py for the
prompt and the rank-validated merge).

    python3 -m generation.enrich --condition low --start 0 --count 200 --dry-run   # count papers to enrich
    python3 -m generation.enrich --condition low --start 0 --count 200 --submit
    python3 -m generation.enrich --condition low --status
    python3 -m generation.enrich --condition low --collect
    python3 -m generation.enrich --condition low --summarize                       # extraction success totals

Without --start/--count all papers in the manifest are selected. Papers whose metadata were
already extracted without failures are skipped unless --overwrite is given.
--submit, --status and --collect read the API key from OPENAI_API_KEY.
"""

import argparse
import json
from pathlib import Path

from generation import enrich_parsed_reference_metadata as metadata
from generation.run import (
    OUTPUT_ROOT, download_batch, ensure_no_open_batch, load_jsonl, mark_collected,
    now, show_status, submit_batch, write_json, write_jsonl,
)


MODEL = "gpt-4.1-mini"


def references(payload):
    return (payload.get("parsed_response") or {}).get("references") or []


def parsed_path(condition, row):
    return OUTPUT_ROOT / condition / "parsed" / Path(row["parsed_output_path"]).name


def is_complete(path):
    """True if the file's metadata were extracted with the rank-validated method and no citation failed."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts = payload.get("metadata_enrichment_counts") or {}
    return payload.get("metadata_enrichment_method") == metadata.METHOD and counts.get("metadata_failed", 0) == 0


def pending_paths(condition, start, count, overwrite):
    """Parsed files in the selected cohort range that still need metadata extraction."""
    rows = load_jsonl(OUTPUT_ROOT / condition / "manifest.jsonl")
    if start is not None:
        stop = start + count
        rows = [row for row in rows if start <= row["sample_index"] < stop]
        if len(rows) != count:
            raise ValueError(f"Expected {count} manifest rows for indices {start}-{stop - 1}; found {len(rows)}.")
    paths = [parsed_path(condition, row) for row in rows]
    return paths if overwrite else [path for path in paths if not is_complete(path)]


def dry_run(paths):
    n_references = sum(len(references(json.loads(path.read_text(encoding="utf-8")))) for path in paths)
    print(f"Files requiring enrichment: {len(paths)}")
    print(f"References: {n_references}")
    print("No API calls made.")


def submit(condition, paths):
    if not paths:
        print("No parsed files require metadata enrichment.")
        return
    meta_dir = OUTPUT_ROOT / condition / "meta"
    ensure_no_open_batch(meta_dir)
    requests = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        requests.append({
            "custom_id": payload["focal_paper_id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "messages": [{"role": "user", "content": metadata.build_extraction_prompt(references(payload))}],
                "temperature": 0,
            },
        })
    submit_batch(meta_dir, requests, "/v1/chat/completions",
                 {"experiment": "gpt55_1k_meta", "condition": condition},
                 {"condition": condition, "model": MODEL, "count": len(paths)})


def collect(condition):
    """Download the metadata batch and merge each returned table into its parsed-output file."""
    meta_dir = OUTPUT_ROOT / condition / "meta"
    state, output = download_batch(meta_dir)
    paths = {row["focal_paper_id"]: parsed_path(condition, row)
             for row in load_jsonl(OUTPUT_ROOT / condition / "manifest.jsonl")}
    results = []
    for item in output:
        paper_id = item["custom_id"]
        response = item.get("response") or {}
        try:
            if response.get("status_code") != 200:
                raise RuntimeError(f"HTTP {response.get('status_code')}: {response.get('body')}")
            path = paths[paper_id]
            markdown = response["body"]["choices"][0]["message"]["content"]
            enriched, counts = metadata.enrich_payload(json.loads(path.read_text(encoding="utf-8")), markdown)
            write_json(path, enriched)
            status, error = "completed", None
        except Exception as exc:  # record the failure and continue with the remaining papers
            counts, status, error = {}, "error", str(exc)
        results.append({"focal_paper_id": paper_id, "status": status, "error_message": error, **counts})
    # Keep the latest result per focal paper across all metadata batches.
    latest = {row["focal_paper_id"]: row for row in load_jsonl(meta_dir / "results.jsonl")}
    latest.update((row["focal_paper_id"], row) for row in results)
    write_jsonl(meta_dir / "results.jsonl", [latest[key] for key in sorted(latest)])
    mark_collected(meta_dir, state)
    completed = sum(row["status"] == "completed" for row in results)
    print(f"Collected {len(results)} files: {completed} completed, {len(results) - completed} errors.")
    summarize()


def summarize():
    """Count citation-level metadata extraction outcomes over all parsed files."""
    fields = ("files", "references", "metadata_success", "metadata_failed", "metadata_status_missing")
    conditions, overall = {}, dict.fromkeys(fields, 0)
    for condition in ("low", "high"):
        counts = dict.fromkeys(fields, 0)
        for path in sorted((OUTPUT_ROOT / condition / "parsed").glob("*.json")):
            refs = references(json.loads(path.read_text(encoding="utf-8")))
            counts["files"] += 1
            counts["references"] += len(refs)
            for ref in refs:
                status = ref.get("metadata_parse_status")
                counts[f"metadata_{status}" if status in ("success", "failed") else "metadata_status_missing"] += 1
        conditions[condition] = counts
        for key in fields:
            overall[key] += counts[key]
    for name, counts in (*conditions.items(), ("overall", overall)):
        counts["extractable_metadata_percent"] = (
            round(100 * counts["metadata_success"] / counts["references"], 4) if counts["references"] else None
        )
        print(f"{name}: {counts['files']} files, {counts['references']} references, "
              f"{counts['metadata_success']} extracted ({counts['extractable_metadata_percent']}%), "
              f"{counts['metadata_failed']} failed, {counts['metadata_status_missing']} not yet enriched")
    output_path = OUTPUT_ROOT / "metadata_enrichment_summary.json"
    write_json(output_path, {
        "generated_at": now(),
        "source": "outputs/<condition>/parsed/*.json",
        "conditions": conditions,
        "overall": overall,
    })
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", required=True, choices=("low", "high"), help="reasoning-effort condition")
    parser.add_argument("--start", type=int, help="first cohort index (use with --count; default: all papers)")
    parser.add_argument("--count", type=int, help="number of papers")
    parser.add_argument("--overwrite", action="store_true", help="also re-enrich papers that are already complete")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true", help="count the papers to enrich (no API calls)")
    action.add_argument("--submit", action="store_true", help="submit one metadata batch")
    action.add_argument("--status", action="store_true", help="show the progress of the current batch")
    action.add_argument("--collect", action="store_true", help="download and merge the finished batch")
    action.add_argument("--summarize", action="store_true", help="write metadata_enrichment_summary.json")
    args = parser.parse_args()
    if (args.start is None) != (args.count is None):
        parser.error("use --start and --count together")
    if args.dry_run:
        dry_run(pending_paths(args.condition, args.start, args.count, args.overwrite))
    elif args.submit:
        submit(args.condition, pending_paths(args.condition, args.start, args.count, args.overwrite))
    elif args.status:
        show_status(OUTPUT_ROOT / args.condition / "meta")
    elif args.collect:
        collect(args.condition)
    else:
        summarize()


if __name__ == "__main__":
    main()
