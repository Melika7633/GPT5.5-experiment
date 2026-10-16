"""Resolve the enriched citations of one condition against OpenAlex, Crossref, Semantic Scholar and arXiv.

The 1,000 focal papers are resolved in five blocks of 200 (cohort indices 0, 200, 400, 600, 800):

    python3 -m generation.resolve --condition low --start 0 --count 200    # -> outputs/low/resolved/0000-0199/
    ...
    python3 -m generation.resolve --condition low --merge-blocks           # -> outputs/low/resolved/all/

Sources are queried in this fixed order, one source at a time for all references of the block,
and a reference is no longer searched once it is verified or possible (scoring and classification:
resolve_generated_references.py). Every completed search is checkpointed per reference in
outputs/<condition>/checkpoint/, so an interrupted run is resumed by repeating the same command.
A failed request (HTTP error, timeout) is not a negative search: the source stays pending and the
command exits with status 2 until a rerun completes it. resolver_exceptions/*.json documents
references for which one source stayed unreachable. --no-network rebuilds the block tables from the
saved checkpoints without querying any source.

Needs OPENALEX_API_KEY and SEMANTIC_SCHOLAR_API_KEY (CROSSREF_MAILTO is optional), and
focal_papers_AA.csv and ground_truth_references_AA.csv next to this script.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from generation import resolve_generated_references as core
from generation.enrich_parsed_reference_metadata import METHOD as ENRICHMENT_METHOD
from common.project_utils import write_jsonl
from generation.run import load_jsonl


HERE = Path(__file__).resolve().parents[1]
SOURCES = ("openalex", "crossref", "semantic_scholar", "arxiv")
BLOCK_STARTS = (0, 200, 400, 600, 800)
BLOCK_SIZE = 200


def block_name(start):
    return f"{start:04d}-{start + BLOCK_SIZE - 1:04d}"


# Per-reference checkpoints

def reference_identity(reference):
    """Hash of the generated reference's fields; it keys the checkpoint, so changed inputs never reuse old searches."""
    payload = {
        "focal_paper_id": str(reference.get("focal_paper_id", "")),
        "prompt_id": str(reference.get("prompt_id", "")),
        "generated_reference_index": reference.get("generated_reference_index"),
        "raw_citation": str(reference.get("raw_citation", "")),
        "generated_title": str(reference.get("generated_title") or ""),
        "generated_authors": reference.get("generated_authors") or [],
        "generated_year": reference.get("generated_year"),
        "generated_doi": str(reference.get("generated_doi") or ""),
        "generated_arxiv_id": str(reference.get("generated_arxiv_id") or ""),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), payload


def checkpoint_path(run_dir, reference):
    digest = reference_identity(reference)[0]
    return run_dir / "checkpoint" / "by_reference" / digest[:2] / f"{digest}.json"


def load_state(run_dir, reference):
    path = checkpoint_path(run_dir, reference)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"external_candidates": [], "source_errors": [], "source_candidate_counts": {}, "queried_sources": []}


def save_state(run_dir, reference, state):
    """Write the checkpoint atomically, so an interrupted run never leaves a partial file."""
    path = checkpoint_path(run_dir, reference)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({**state, "reference_identity": reference_identity(reference)[1]},
                                    ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_source_exceptions():
    """Per-reference waivers for a source that stayed unreachable, keyed by reference identity."""
    exceptions = {}
    for path in sorted((HERE / "resolver_exceptions").glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for entry in manifest["references"]:
            exceptions.setdefault(entry["reference_identity_sha256"], []).append(
                {"source": manifest["source"], "reason": manifest["reason"], "manifest": path.name})
    return exceptions


SOURCE_EXCEPTIONS = load_source_exceptions()


def build_record(reference, state, focal, ground_truth):
    record = core.build_resolved_record(
        reference, focal, ground_truth, state["external_candidates"], state["source_errors"],
        state["source_candidate_counts"], state["queried_sources"],
    )
    record["pending_sources"] = sorted(state.get("pending_sources", {}))
    record["source_error_history"] = state.get("source_error_history", [])
    # A waived source no longer counts as pending, but only once every other source was searched.
    for waiver in SOURCE_EXCEPTIONS.get(reference_identity(reference)[0], []):
        source = waiver["source"]
        if (source in record["pending_sources"]
                and all(other in record["queried_sources"] for other in SOURCES if other != source)):
            record["pending_sources"].remove(source)
            record.setdefault("source_exceptions", []).append(waiver)
    return record


# Resolving one block

def search(references, states, focal, ground_truth, run_dir):
    """Query each source in turn for every reference that is not yet verified or possible."""
    for source in SOURCES:
        pending = [i for i, (reference, state) in enumerate(zip(references, states))
                   if source in state.get("pending_sources", {})
                   or not core.should_stop_cascade(build_record(reference, state, focal, ground_truth))]
        if not pending:
            continue
        core.log(f"Querying {source} for {len(pending)} unresolved references.")
        deferred = 0
        for position, index in enumerate(pending, start=1):
            reference, state = references[index], states[index]
            label = f"[{source} {position}/{len(pending)}] {reference['focal_paper_id']}#{reference['generated_reference_index']}"
            if source in state["queried_sources"]:  # already searched in an earlier run
                core.log(f"{label} checkpoint")
                continue
            core.log(f"{label} resolving")
            try:
                candidates, errors = core.QUERY_FUNCTIONS[source](reference)
            except Exception as exc:
                candidates, errors = [], [f"{source}:resolver_error:{type(exc).__name__}:{exc}"]
            if any(core.incomplete_source_error(error) for error in errors):
                # A failed request is not a negative search: keep this source pending for a later run.
                state.setdefault("pending_sources", {})[source] = errors
                history = state.setdefault("source_error_history", [])
                for error in errors:
                    if error not in history:
                        history.append(error)
                deferred += 1
            else:
                state.setdefault("pending_sources", {}).pop(source, None)
                # Replace any earlier failed attempt of this source.
                state["external_candidates"] = [c for c in state["external_candidates"] if c.get("source") != source]
                state["source_errors"] = [e for e in state["source_errors"] if not e.startswith((source + "_", source + ":"))]
                state["queried_sources"].append(source)
                state["source_candidate_counts"][source] = len(candidates)
                state["external_candidates"].extend(candidates)
                state["source_errors"].extend(errors)
            save_state(run_dir, reference, state)
        core.log(f"{source}: {deferred} searches deferred this pass.")


def resolve_block(condition, start, no_network):
    run_dir = HERE / "outputs" / condition
    core.CACHE_ROOT = run_dir / "cache"
    rows = [row for row in load_jsonl(run_dir / "manifest.jsonl") if start <= row["sample_index"] < start + BLOCK_SIZE]
    if len(rows) != BLOCK_SIZE:
        raise ValueError(f"Expected {BLOCK_SIZE} manifest rows for block {block_name(start)}; found {len(rows)}.")
    paths = [run_dir / "parsed" / Path(row["parsed_output_path"]).name for row in rows]
    unenriched = [path.name for path in paths
                  if json.loads(path.read_text(encoding="utf-8")).get("metadata_enrichment_method") != ENRICHMENT_METHOD]
    if unenriched:
        raise ValueError(f"{len(unenriched)} parsed files have no extracted metadata yet; run enrich.py first.")

    references = [reference for path in paths for reference in core.flatten_parsed_references(path)]
    focal = core.load_focal_metadata(HERE / "focal_papers_AA.csv")
    ground_truth = core.load_ground_truth(HERE / "ground_truth_references_AA.csv")
    if no_network:
        missing = [checkpoint_path(run_dir, ref) for ref in references
                   if not checkpoint_path(run_dir, ref).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} checkpoints are absent. Published archives omit checkpoints; "
                "use verify_archived_resolution.py to check retained candidates instead."
            )
    states = [load_state(run_dir, reference) for reference in references]
    core.log(f"Resolving {len(references)} references of block {block_name(start)} ({condition}).")
    if not no_network:
        core.require_source_credentials(SOURCES)
        search(references, states, focal, ground_truth, run_dir)

    records = [build_record(reference, state, focal, ground_truth) for reference, state in zip(references, states)]
    for record in records:
        record["omitted_sources"] = []
        record["requested_sources"] = list(SOURCES)
    output_dir = run_dir / "resolved" / block_name(start)
    write_jsonl(output_dir / "references.jsonl", records)
    core.write_csv(output_dir / "references.csv", records)

    print(f"References: {len(records)}")
    print(f"Existence: {dict(Counter(record['existence_status'] for record in records))}")
    print(f"Temporal: {dict(Counter(record['temporal_status'] for record in records))}")
    print(f"Saved {output_dir / 'references.jsonl'} and references.csv")
    pending = sum(bool(record["pending_sources"]) for record in records)
    if pending:
        print(f"INCOMPLETE: {pending} references have pending searches. Rerun this block later; do not merge yet.")
        raise SystemExit(2)


def merge_blocks(condition):
    """Combine the five blocks into resolved/all/ after checking that they are complete."""
    run_dir = HERE / "outputs" / condition
    records = []
    for start in BLOCK_STARTS:
        block = load_jsonl(run_dir / "resolved" / block_name(start) / "references.jsonl")
        if not block:
            raise FileNotFoundError(f"Block {block_name(start)} has not been resolved.")
        if any(row.get("omitted_sources") or row.get("pending_sources")
               or any(core.incomplete_source_error(error) for error in row.get("source_errors", []))
               for row in block):
            raise ValueError(f"Block {block_name(start)} still has pending searches; rerun it before merging.")
        if len({row["focal_paper_id"] for row in block}) != BLOCK_SIZE:
            raise ValueError(f"Block {block_name(start)} does not contain {BLOCK_SIZE} focal papers.")
        records += block
    keys = [(row["focal_paper_id"], row["prompt_id"], row["generated_reference_index"]) for row in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate generated references across blocks.")
    output_dir = run_dir / "resolved" / "all"
    write_jsonl(output_dir / "references.jsonl", records)
    core.write_csv(output_dir / "references.csv", records)
    waived = sum(bool(record.get("source_exceptions")) for record in records)
    print(f"Merged {len(records)} references from {len({key[0] for key in keys})} focal papers "
          f"({waived} with a waived source) into {output_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--condition", required=True, choices=("low", "high"), help="reasoning-effort condition")
    parser.add_argument("--start", type=int, choices=BLOCK_STARTS, help="first cohort index of the block")
    parser.add_argument("--count", type=int, default=BLOCK_SIZE, choices=(BLOCK_SIZE,), help="block size (fixed)")
    parser.add_argument("--no-network", action="store_true", help="rebuild the tables from saved checkpoints only")
    parser.add_argument("--merge-blocks", action="store_true", help="merge the five resolved blocks into resolved/all/")
    args = parser.parse_args()
    if args.merge_blocks:
        merge_blocks(args.condition)
    elif args.start is None:
        parser.error("give --start (0, 200, 400, 600 or 800) or --merge-blocks")
    else:
        resolve_block(args.condition, args.start, args.no_network)


if __name__ == "__main__":
    main()
