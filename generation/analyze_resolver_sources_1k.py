#!/usr/bin/env python3
"""Resolver-source tables for the 1,000-paper cohort.

  outputs/unresolved_source_searches_1k.csv  per source, how the searches for unresolved
                                             references ended (candidates, none, no result)
  outputs/successful_match_sources_1k.csv    which source supplied each verified/possible match
"""

import json
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "outputs"
LABELS = {"openalex": "OpenAlex", "crossref": "Crossref", "semantic_scholar": "Semantic Scholar", "arxiv": "arXiv"}
EXPECTED = {"low": 20_000, "high": 19_986}


def load(condition):
    data = pd.read_csv(OUT / condition / "resolved" / "all" / "references.csv", low_memory=False)
    if len(data) != EXPECTED[condition] or data["focal_paper_id"].nunique() != 1_000:
        raise RuntimeError(f"Unexpected {condition} full-cohort dimensions")
    return data


def unresolved_search_table(data, condition):
    unresolved = data[data["existence_status"] == "unresolved"]
    counts = [json.loads(value) if isinstance(value, str) and value else {} for value in unresolved["source_candidate_counts"]]
    return [{
        "condition": condition,
        "unresolved_reference_denominator": len(unresolved),
        "source": label,
        "returned_candidates": sum(c.get(source, 0) > 0 for c in counts),
        "successful_search_zero_candidates": sum(c.get(source) == 0 for c in counts),
        "source_result_unavailable": sum(source not in c for c in counts),
    } for source, label in LABELS.items()]


def successful_match_table(data, condition):
    rows = []
    for status, status_label in (("verified_real_paper", "Verified"), ("possible_real_paper", "Possible")):
        subset = data[data["existence_status"] == status]
        counts = subset["matched_source"].value_counts()
        row = {"condition": condition, "status": status_label, "status_denominator": len(subset)}
        row.update({label: int(counts.get(source, 0)) for source, label in LABELS.items()})
        if sum(row[label] for label in LABELS.values()) != len(subset):
            raise RuntimeError(f"{condition}/{status}: matched-source counts do not sum")
        rows.append(row)
    return rows


def main():
    unresolved_rows, match_rows = [], []
    for condition in ("low", "high"):
        data = load(condition)
        unresolved_rows += unresolved_search_table(data, condition)
        match_rows += successful_match_table(data, condition)
    unresolved, matches = pd.DataFrame(unresolved_rows), pd.DataFrame(match_rows)
    unresolved.to_csv(OUT / "unresolved_source_searches_1k.csv", index=False)
    matches.to_csv(OUT / "successful_match_sources_1k.csv", index=False)
    print("UNRESOLVED SOURCE SEARCHES")
    print(unresolved.to_string(index=False))
    print("\nSUCCESSFUL MATCH SOURCES")
    print(matches.to_string(index=False))
    print(f"\nWrote {OUT / 'unresolved_source_searches_1k.csv'}\nWrote {OUT / 'successful_match_sources_1k.csv'}")


if __name__ == "__main__":
    main()
