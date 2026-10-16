"""Rebuild the title-pair table used by Fig 2D from final resolver CSVs.

Exact normalized titles generate candidate pairs, including pairs from different
focal papers. Fig 2D subsequently requires the same focal paper, an unambiguous
low-reference pair, and matching DOI or observed-bibliography identity support.
"""
import csv
import re
import unicodedata
from collections import defaultdict
from pathlib import Path



HERE = Path(__file__).resolve().parents[1]
FIELDS = ["normalized_title", "low_title", "high_title", "same_focal_paper",
          "low_focal_paper_id", "high_focal_paper_id", "low_reference_index",
          "high_reference_index", "low_final_label", "high_final_label",
          "low_metadata_status", "high_metadata_status"]


def normalize_title(title):
    """Published overlap normalization; differs from fuzzy-resolver normalization."""
    text = unicodedata.normalize("NFKC", title).lower()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def build_rows():
    references = {}
    for condition in ("low", "high"):
        with (HERE / "outputs" / condition / "resolved/all/references.csv").open(
                newline="", encoding="utf-8") as handle:
            references[condition] = list(csv.DictReader(handle))
    high_by_title = defaultdict(list)
    for row in references["high"]:
        title = normalize_title(row["generated_title"])
        if title:
            high_by_title[title].append(row)
    result = []
    for low in references["low"]:
        title = normalize_title(low["generated_title"])
        for high in high_by_title.get(title, []):
            result.append({
                "normalized_title": title,
                "low_title": low["generated_title"], "high_title": high["generated_title"],
                "same_focal_paper": str(low["focal_paper_id"] == high["focal_paper_id"]),
                "low_focal_paper_id": low["focal_paper_id"],
                "high_focal_paper_id": high["focal_paper_id"],
                "low_reference_index": low["generated_reference_index"],
                "high_reference_index": high["generated_reference_index"],
                "low_final_label": low["existence_status"],
                "high_final_label": high["existence_status"],
                "low_metadata_status": low["metadata_status"],
                "high_metadata_status": high["metadata_status"],
            })
    return result


def main():
    path = HERE / "outputs/low_high_title_overlap_full_1k.csv"
    rows = build_rows()
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Rebuilt {len(rows)} title-pair rows: {path}")


if __name__ == "__main__":
    main()
