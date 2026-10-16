"""Extract structured metadata from generated citation strings.

For each focal paper, its numbered citations are sent to the metadata model in one prompt
(build_extraction_prompt). The model returns a Markdown table with one row per citation.
Rows are merged back by rank, never by position: if the table cannot be parsed, or any rank
is missing, duplicated or unexpected, every citation of that paper is marked as failed.

This is a parser only; no bibliographic database is queried here.
"""

import re

from common.project_utils import utc_now
from generation.resolve_generated_references import clean_arxiv_id, clean_doi, split_author_names


METHOD = "llm_markdown_table_rank_validated_v2"

EXTRACTION_PROMPT = """Below, we share with you a list of references.

Could you for each reference preserve its supplied rank and extract the authors, the number of authors, title, publication year, publication venue, DOI, and arXiv ID?

Please return exactly one row per supplied reference. Copy the supplied integer rank into the first column; do not renumber, omit, duplicate, combine, or add references.

Please only return the extracted information in a markdown table with rank, authors, number of authors, title, publication year, publication venue, DOI, and arXiv ID as columns.

Do not return any additional information or formatting, such as ```markdown:
| Rank | Authors | Number of Authors | Title | Publication Year | Publication Venue | DOI | arXiv ID |
Make sure to respect this format."""

EXPECTED_COLUMNS = [
    "Rank", "Authors", "Number of Authors", "Title",
    "Publication Year", "Publication Venue", "DOI", "arXiv ID",
]
METADATA_FIELDS = (
    "authors", "authors_text", "number_of_authors", "title", "year",
    "venue", "doi", "arxiv_id", "metadata_parse_status",
)
EMPTY_VALUES = {"null", "none", "n/a", "na", "unknown", "unclear", "-"}
EMPTY_AUTHORS = {"null", "none", "n/a", "na", "-"}


def escape_markdown_cell(value):
    text = "" if value is None else str(value)
    text = text.replace("\\", "\\\\").replace("|", "\\|")
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "<br>")


def unescape_markdown_cell(value):
    return value.replace("<br>", "\n").replace("\\|", "|").replace("\\\\", "\\").strip()


def build_extraction_prompt(references):
    lines = [EXTRACTION_PROMPT, "", "References:"]
    lines += [f"{ref.get('rank')}. {escape_markdown_cell(ref.get('raw_citation', ''))}" for ref in references]
    return "\n".join(lines)


# Parsing the returned table

def split_markdown_row(line):
    """Split one table row into cells, keeping escaped pipes inside cells."""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    cells, current, escaped = [], [], False
    for char in stripped:
        if escaped:
            current.append("\\" + char if char != "|" else char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == "|":
            cells.append(unescape_markdown_cell("".join(current)))
            current = []
        else:
            current.append(char)
    if escaped:
        current.append("\\")
    cells.append(unescape_markdown_cell("".join(current)))
    return cells


def is_separator_row(cells):
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.strip()) for cell in cells)


def parse_markdown_table(markdown):
    """Return the table rows below the expected header as dicts keyed by column name."""
    table_lines = [line for line in markdown.splitlines() if line.strip().startswith("|") and line.strip().endswith("|")]
    header = next((i for i, line in enumerate(table_lines) if split_markdown_row(line) == EXPECTED_COLUMNS), None)
    if header is None:
        raise ValueError("LLM response did not contain the expected markdown table header.")
    rows = []
    for line in table_lines[header + 1:]:
        cells = split_markdown_row(line)
        if is_separator_row(cells):
            continue
        if len(cells) > len(EXPECTED_COLUMNS):
            # Extra cells between the venue and the two identifier columns (e.g. volume, pages)
            # are kept as part of the venue.
            venue = ", ".join(cell for cell in cells[5:-2] if cell)
            cells = cells[:5] + [venue] + cells[-2:]
        if len(cells) != len(EXPECTED_COLUMNS):
            rows.append({"__row_parse_error__": f"Markdown table row has {len(cells)} cells; expected {len(EXPECTED_COLUMNS)}."})
            continue
        rows.append(dict(zip(EXPECTED_COLUMNS, cells)))
    if not rows:
        raise ValueError("LLM response markdown table contained no reference rows.")
    return rows


def parse_optional_int(value):
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in EMPTY_VALUES or cleaned.startswith((">", "<", "~")):
        return None
    match = re.search(r"\d+", cleaned)
    return int(match.group(0)) if match else None


def parse_optional_identifier(value, cleaner):
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in EMPTY_VALUES:
        return None
    return cleaner(cleaned)


def split_authors_cell(value):
    cleaned = value.strip()
    if not cleaned or cleaned.lower() in EMPTY_AUTHORS:
        return []
    if ";" in cleaned:
        return split_author_names([part.strip() for part in cleaned.split(";") if part.strip()])
    return split_author_names(cleaned)


def normalize_metadata_row(row):
    authors_text = row["Authors"].strip()
    return {
        "authors": split_authors_cell(authors_text),
        "authors_text": authors_text,
        "number_of_authors": parse_optional_int(row["Number of Authors"]),
        "title": row["Title"].strip(),
        "year": parse_optional_int(row["Publication Year"]),
        "venue": row["Publication Venue"].strip(),
        "doi": parse_optional_identifier(row["DOI"], clean_doi),
        "arxiv_id": parse_optional_identifier(row["arXiv ID"], clean_arxiv_id),
        "metadata_parse_status": "success",
    }


def rows_by_validated_rank(references, rows):
    """Index the table rows by rank; the ranks must match the input ranks exactly."""
    expected = [ref.get("rank") for ref in references]
    if any(not isinstance(rank, int) for rank in expected):
        raise ValueError(f"Input references contain invalid ranks: {expected!r}.")
    if len(set(expected)) != len(expected):
        raise ValueError(f"Input reference ranks are not unique: {expected!r}.")
    indexed = {}
    for row in rows:
        if row.get("__row_parse_error__"):
            raise ValueError(row["__row_parse_error__"])
        value = row["Rank"].strip()
        if not re.fullmatch(r"\d+", value):
            raise ValueError(f"Invalid metadata row rank: {value!r}.")
        if int(value) in indexed:
            raise ValueError(f"LLM returned duplicate metadata rank {int(value)}.")
        indexed[int(value)] = row
    if set(indexed) != set(expected):
        missing = sorted(set(expected) - set(indexed))
        unexpected = sorted(set(indexed) - set(expected))
        raise ValueError(
            "LLM metadata ranks did not exactly match input references: "
            f"missing={missing}, unexpected={unexpected}."
        )
    return indexed


# Merging metadata into a parsed-output file

def count_statuses(references):
    statuses = [ref.get("metadata_parse_status") for ref in references]
    return {
        "references": len(references),
        "metadata_success": statuses.count("success"),
        "metadata_failed": statuses.count("failed"),
    }


def enrich_references(references, markdown):
    """Merge the metadata table into the references, or mark all of them failed on any error."""
    if not references:
        return [], count_statuses([])
    try:
        indexed = rows_by_validated_rank(references, parse_markdown_table(markdown))
        enriched = []
        for ref in references:
            metadata = normalize_metadata_row(indexed[ref["rank"]])
            merged = {**ref, **{field: metadata[field] for field in METADATA_FIELDS}}
            merged.pop("metadata_parse_error", None)
            enriched.append(merged)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        enriched = [{**ref, "metadata_parse_status": "failed", "metadata_parse_error": error} for ref in references]
    return enriched, count_statuses(enriched)


def enrich_payload(payload, markdown):
    """Return the parsed-output payload with metadata added to its references, and the status counts."""
    parsed_response = dict(payload.get("parsed_response") or {})
    references = parsed_response.get("references") or []
    if not isinstance(references, list):
        raise ValueError("parsed_response.references is not a list.")
    parsed_response["references"], counts = enrich_references(references, markdown)
    return {
        **payload,
        "parsed_response": parsed_response,
        "metadata_enriched_at": utc_now(),
        "metadata_enrichment_method": METHOD,
        "metadata_enrichment_counts": counts,
    }, counts
