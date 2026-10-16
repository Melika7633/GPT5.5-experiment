"""Verify retained candidate scores and labels without searches or checkpoints.

Reads final JSONL records only; never rewrites the archived evidence. This does
not replay historical retrieval, verify omitted candidates, or confirm source
coverage. References with no candidates retain their recorded search/error
provenance; absence of candidates alone does not establish nonexistence.
"""
import json
from pathlib import Path

from generation import resolve_generated_references as core

HERE = Path(__file__).resolve().parents[1]
EXPECTED = {"low": 20000, "high": 19986}
SCORES = core.CANDIDATE_SCORE_FIELDS


def verify():
    for condition, expected in EXPECTED.items():
        path = HERE / "outputs" / condition / "resolved/all/references.jsonl"
        count, seen, failures = 0, set(), []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                count += 1
                key = (str(record["focal_paper_id"]), record["prompt_id"],
                       record["generated_reference_index"])
                if key in seen:
                    failures.append((key, "duplicate reference"))
                seen.add(key)
                candidates = record.get("top_candidates") or []
                rescored = []
                for candidate in candidates:
                    fresh = core.score_candidate(record, candidate)
                    for field in SCORES:
                        if fresh.get(field) != candidate.get(field):
                            failures.append((key, f"candidate {field}"))
                    rescored.append(fresh)
                # Preserve the saved tie order; classification uses the first three.
                label, matched = core.classify_match(record, rescored, record.get("source_errors") or [])
                for field in ("existence_status", "metadata_status", "resolver_status"):
                    if label[field] != record.get(field):
                        failures.append((key, field))
                if core.temporal_status(record.get("focal_year"), record.get("generated_year")) != record.get("temporal_status"):
                    failures.append((key, "temporal_status"))
        if count != expected or len({key[0] for key in seen}) != 1000:
            raise ValueError(f"{condition}: unexpected reference/cohort dimensions")
        if failures:
            raise ValueError(f"{condition}: {len(failures)} discrepancies; first: {failures[:5]}")
        print(f"{condition}: {count} retained-candidate classifications verified")


if __name__ == "__main__":
    verify()
