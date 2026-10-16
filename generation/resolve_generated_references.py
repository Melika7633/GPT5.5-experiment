"""Search scholarly sources for a generated reference, score the candidates and classify existence.

Each source is searched by the reference's DOI or arXiv ID (when one was extracted) and by its
title. Every candidate is scored against the generated title and authors, and candidates are
ranked by

    match_confidence = 0.55 * title_match_score + 0.45 * author_match_score

Only the three best-ranked candidates are used to classify the reference:

    verified_real_paper   title >= 0.90, author >= 0.50 and the same first-author surname
    possible_real_paper   title > 0.80 and author > 0.40
    likely_hallucinated   no candidate at all, and no search failed
    unresolved            everything else, including references without an extracted title or authors

DOI, arXiv ID and year do not affect ranking or existence; they are reported as metadata checks
(metadata_status, *_match_status). temporal_status flags references dated after the focal paper.
"""

import csv
import hashlib
import json
import os
import random
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from rapidfuzz import fuzz

from common.project_utils import project_relative, utc_now


DOI_RE = re.compile(r'\b10\.\d{4,9}/[^\s"<>]+', re.IGNORECASE)
YEAR_RE = re.compile(r"\b(18|19|20|21)\d{2}\b")
ARXIV_ID_RE = re.compile(r"\b(?:arxiv:)?((?:\d{4}\.\d{4,5})|(?:[a-z-]+(?:\.[A-Z]{2})?/\d{7}))(?:v\d+)?\b", re.IGNORECASE)
ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
USER_AGENT = "citation-trace-audit/0.1"
SEARCH_SIZE = 10
TITLE_EXISTS_THRESHOLD = 0.90
AUTHOR_EXISTS_THRESHOLD = 0.50
POSSIBLE_TITLE_THRESHOLD = 0.80
POSSIBLE_AUTHOR_THRESHOLD = 0.40
CLASSIFICATION_CANDIDATE_LIMIT = 3
CACHE_ROOT = Path("outputs/cache")  # set per condition by resolve.py

# Minimum seconds between requests to each host (Semantic Scholar returned HTTP 429 at 1.05 s).
REQUEST_INTERVAL_SECONDS = {
    "api.openalex.org": 0.25,
    "api.crossref.org": 0.11,
    "api.semanticscholar.org": 2.0,
    "export.arxiv.org": 3.1,
}
# On HTTP 429 a request is retried (honouring Retry-After) for at most this many seconds and
# retries; after that the host is skipped for the rest of the run and its searches stay pending.
RATE_LIMIT_WAIT_BUDGET = 60.0
HTTP_MAX_RETRIES = 5
QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "into",
    "is", "of", "on", "or", "the", "to", "using", "with",
}
LAST_REQUEST_AT = {}
RATE_LIMIT_UNTIL = {}


def log(message):
    print(f"[{utc_now()}] {message}", flush=True)


# Normalisation and similarity

def normalize_text(value):
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii").lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[‐-―]", "-", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def clean_doi(value):
    match = DOI_RE.search(str(value)) if value else None
    return match.group(0).rstrip(".,;)]}").lower() if match else None


def clean_arxiv_id(value):
    match = ARXIV_ID_RE.search(str(value)) if value else None
    return match.group(1).lower() if match else None


def extract_year(value):
    match = YEAR_RE.search(str(value)) if value is not None else None
    return int(match.group(0)) if match else None


def split_author_names(value):
    if value is None:
        return []
    if isinstance(value, list):
        raw_authors = [str(author) for author in value if str(author).strip()]
    else:
        text = re.sub(r"\bet\s+al\.?", "", str(value), flags=re.IGNORECASE).replace(";", ",")
        text = re.sub(r"\s+and\s+", ",", text, flags=re.IGNORECASE)
        raw_authors = [part.strip() for part in text.split(",") if part.strip()]
    return [re.sub(r"\s+", " ", author).strip() for author in raw_authors]


def author_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(author) for author in value if str(author).strip())
    return str(value)


def surname(author):
    parts = normalize_text(author).split()
    if not parts:
        return ""
    return next((part for part in reversed(parts) if len(part) > 2), parts[-1])


def first_author_surname(value):
    if isinstance(value, list):
        return next((surname(str(author)) for author in value if str(author).strip()), "")
    authors = split_author_names(value)
    return surname(authors[0]) if authors else ""


def title_similarity(left, right):
    left, right = normalize_text(left), normalize_text(right)
    return fuzz.token_sort_ratio(left, right) / 100.0 if left and right else 0.0


def author_score(generated_authors, candidate_authors):
    """Token-set similarity of the author lists, and whether the first-author surnames agree."""
    generated = normalize_text(author_text(generated_authors))
    candidate = normalize_text(author_text(candidate_authors))
    if not generated or not candidate:
        return 0.0, False
    generated_first = first_author_surname(generated_authors)
    candidate_first = first_author_surname(candidate_authors)
    return (fuzz.token_set_ratio(generated, candidate) / 100.0,
            bool(generated_first) and generated_first == candidate_first)


def raw_citation_similarity(raw_reference, candidate):
    """Similarity of the full citation string to the candidate's authors, year, title, venue and DOI."""
    parts = [" ".join(str(author) for author in candidate.get("authors") or []), str(candidate.get("year") or ""),
             str(candidate.get("title") or ""), str(candidate.get("venue") or ""), str(candidate.get("doi") or "")]
    raw_norm = normalize_text(raw_reference)
    candidate_norm = normalize_text(" ".join(part for part in parts if part).strip())
    return fuzz.token_set_ratio(raw_norm, candidate_norm) / 100.0 if raw_norm and candidate_norm else 0.0


def year_match_status(generated_year, candidate_year):
    if generated_year is None or candidate_year is None:
        return "missing"
    difference = abs(generated_year - candidate_year)
    return "exact" if difference == 0 else "near" if difference <= 1 else "conflicting"


def identifier_status(generated, candidate):
    if generated and candidate:
        return "exact" if generated == candidate else "conflicting"
    if generated:
        return "candidate_missing"
    return "generated_missing" if candidate else "missing"


# Input data

def load_focal_metadata(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["PaperID"]: {
            "focal_paper_id": row["PaperID"],
            "focal_title": row.get("PaperTitle"),
            "focal_year": extract_year(row.get("Year")),
            "focal_doi": clean_doi(row.get("DOI")),
            "focal_venue": row.get("Journal_Name"),
        } for row in csv.DictReader(handle)}


def load_ground_truth(path):
    """The focal papers' real reference lists, keyed by focal-paper ID."""
    by_focal = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_focal.setdefault(row["Citing_PaperID"], []).append({
                "source": "local_ground_truth",
                "source_id": row.get("Cited_PaperID") or row.get("PaperID_x"),
                "title": row.get("PaperTitle") or "",
                "authors": split_author_names(row.get("Author_Name")),
                "year": extract_year(row.get("Year_x")),
                "doi": clean_doi(row.get("DOI")),
                "venue": row.get("Journal_Name") or "",
                "url": None,
            })
    return by_focal


def flatten_parsed_references(path):
    """One record per generated reference in a parsed-output file, with its extracted metadata."""
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    records = []
    for index, reference in enumerate((payload.get("parsed_response") or {}).get("references") or [], start=1):
        raw_citation = reference.get("raw_citation") or reference.get("citation") or ""
        records.append({
            "run_id": payload.get("run_id"),
            "focal_paper_id": str(payload.get("focal_paper_id")),
            "prompt_id": payload.get("prompt_id"),
            "model_name": payload.get("model_name"),
            "parsed_output_path": project_relative(path),
            "generated_reference_index": int(reference.get("rank") or index),
            "generated_title": reference.get("title") or "",
            "generated_authors": split_author_names(reference.get("authors")),
            "generated_authors_text": reference.get("authors_text") or author_text(reference.get("authors")),
            "generated_year": extract_year(reference.get("year")) or extract_year(raw_citation),
            "generated_doi": clean_doi(reference.get("doi")) or clean_doi(raw_citation),
            "generated_arxiv_id": clean_arxiv_id(reference.get("arxiv_id")) or clean_arxiv_id(raw_citation),
            "generated_venue": reference.get("venue") or "",
            "raw_generated_reference": raw_citation,
            "raw_generated_reference_object": reference,
        })
    return records


# HTTP with an on-disk cache, per-host pacing and HTTP 429 handling

def cache_path(kind, url):
    return CACHE_ROOT / kind / f"{hashlib.sha1(url.encode('utf-8')).hexdigest()}.json"


def read_cached_payload(kind, url):
    try:
        return json.loads(cache_path(kind, url).read_text(encoding="utf-8")).get("payload")
    except (OSError, ValueError, AttributeError):  # missing or unreadable cache entry
        return None


def write_cached_payload(kind, url, payload):
    path = cache_path(kind, url)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"url": url, "cached_at": utc_now(), "payload": payload}, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    except OSError:  # the cache is only an optimisation
        pass


def pace_request(url):
    hostname = urllib.parse.urlsplit(url).hostname or ""
    wait = REQUEST_INTERVAL_SECONDS.get(hostname, 0.0) - (time.monotonic() - LAST_REQUEST_AT.get(hostname, 0.0))
    if wait > 0:
        time.sleep(wait)
    LAST_REQUEST_AT[hostname] = time.monotonic()


def wait_for_rate_limit(url, exc, attempt, started):
    """Wait before retrying after HTTP 429; return False (and pause the host) when out of budget."""
    hostname = urllib.parse.urlsplit(url).hostname or "unknown host"
    try:
        wait = max(float(exc.headers.get("Retry-After")), 0.0)
    except (TypeError, ValueError):
        wait = min(2.0 ** (attempt + 1), 300.0) + random.uniform(0.0, 1.0)
    if attempt >= HTTP_MAX_RETRIES or wait > RATE_LIMIT_WAIT_BUDGET - (time.monotonic() - started):
        RATE_LIMIT_UNTIL[hostname] = time.monotonic() + wait
        log(f"{hostname}: HTTP 429; deferring this host. Affected searches remain pending for a later run.")
        return False
    log(f"{hostname}: HTTP 429; waiting {wait:.1f}s before retry {attempt + 1}/{HTTP_MAX_RETRIES}.")
    time.sleep(wait)
    return True


def authenticated_url(url):
    """Add the OpenAlex API key or the Crossref contact address (not part of the cache key)."""
    parsed = urllib.parse.urlsplit(url)
    extra = {"api.openalex.org": ("api_key", os.environ.get("OPENALEX_API_KEY")),
             "api.crossref.org": ("mailto", os.environ.get("CROSSREF_MAILTO"))}.get(parsed.hostname)
    if not extra or not extra[1]:
        return url
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True) + [extra]
    return urllib.parse.urlunsplit(parsed._replace(query=urllib.parse.urlencode(query)))


def request_headers(url, accept_json):
    headers = {"User-Agent": USER_AGENT}
    if accept_json:
        headers["Accept"] = "application/json"
    if urllib.parse.urlsplit(url).hostname == "api.semanticscholar.org" and os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
        headers["x-api-key"] = os.environ["SEMANTIC_SCHOLAR_API_KEY"]
    return headers


def require_source_credentials(sources):
    """OpenAlex and Semantic Scholar need API keys; their anonymous quotas are too small."""
    for source, name in (("openalex", "OPENALEX_API_KEY"), ("semantic_scholar", "SEMANTIC_SCHOLAR_API_KEY")):
        if source in sources and not os.environ.get(name):
            raise ValueError(f"Set {name} before running the resolver.")


def http_get(url, as_json=True, timeout=20):
    """GET a URL (cached); return (payload, None), or (None, error) for a failed request."""
    kind = "json" if as_json else "text"
    cached = read_cached_payload(kind, url)
    if isinstance(cached, dict if as_json else str):
        return cached, None
    hostname = urllib.parse.urlsplit(url).hostname or ""
    if time.monotonic() < RATE_LIMIT_UNTIL.get(hostname, 0.0):
        return None, "http_429:host_cooldown"
    request = urllib.request.Request(authenticated_url(url), headers=request_headers(url, as_json))
    started = time.monotonic()
    for attempt in range(HTTP_MAX_RETRIES + 1):
        try:
            pace_request(request.full_url)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read().decode("utf-8")
            payload = json.loads(payload) if as_json else payload
            write_cached_payload(kind, url, payload)
            return payload, None
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and wait_for_rate_limit(url, exc, attempt, started):
                continue
            return None, f"http_{exc.code}"
        except urllib.error.URLError as exc:
            return None, f"url_error:{exc.reason}"
        except Exception as exc:
            return None, f"{type(exc).__name__}:{exc}"
    return None, "retry_exhausted"


def incomplete_source_error(error):
    """A failed request is not a completed negative search (a 404 for a DOI/ID lookup is)."""
    label, separator, detail = error.partition(":")
    if not separator:
        detail = error
    if detail == "http_404" and label.endswith(("_doi", "_id")):
        return False
    return detail.startswith(("http_", "url_error:", "retry_exhausted", "parse_error:", "resolver_error:")) or "Error:" in detail


# Source queries: each returns (candidates, errors)

def title_query_text(reference):
    return str(reference.get("generated_title") or "").strip()


def openalex_candidate(work, query_kind):
    source = (work.get("primary_location") or {}).get("source") or {}
    return {
        "source": "openalex",
        "query_kind": query_kind,
        "source_id": work.get("id"),
        "title": work.get("title") or "",
        "authors": [a.get("author", {}).get("display_name", "") for a in work.get("authorships", [])
                    if a.get("author", {}).get("display_name")],
        "year": work.get("publication_year"),
        "doi": clean_doi(work.get("doi")),
        "venue": source.get("display_name") or "",
        "url": work.get("id"),
    }


def query_openalex(reference):
    candidates, errors = [], []
    select = "id,doi,title,publication_year,authorships,primary_location"

    def search(filter_value):
        params = {"filter": filter_value, "per-page": SEARCH_SIZE, "select": select}
        return http_get("https://api.openalex.org/works?" + urllib.parse.urlencode(params))

    if reference.get("generated_doi"):
        payload, error = search(f"doi:{reference['generated_doi']}")
        if error:
            errors.append(f"openalex_doi:{error}")
        else:
            candidates += [openalex_candidate(work, "doi") for work in payload.get("results", [])]
    title = title_query_text(reference)
    if title:
        payload, error = search(f"title.search:{title}")
        if error == "http_400":
            # OpenAlex rejects some punctuation; retry once with a punctuation-free title.
            plain_title = " ".join(re.sub(r"[^\w\s-]", " ", title).split())
            if plain_title and plain_title != title:
                log("openalex: title query returned HTTP 400; retrying punctuation-free title search.")
                payload, error = search(f"title.search:{plain_title}")
        if error:
            errors.append(f"openalex_title:{error}")
        else:
            candidates += [openalex_candidate(work, "title") for work in payload.get("results", [])]
    return candidates, errors


def crossref_candidate(item, query_kind):
    year = None
    for key in ("published-print", "published-online", "published", "issued"):
        date_parts = (item.get(key) or {}).get("date-parts") or []
        if date_parts and date_parts[0]:
            year = extract_year(date_parts[0][0])
            break
    return {
        "source": "crossref",
        "query_kind": query_kind,
        "source_id": clean_doi(item.get("DOI")),
        "title": (item.get("title") or [""])[0],
        "authors": [name for name in (" ".join(p for p in (a.get("given") or "", a.get("family") or "") if p).strip()
                                      for a in item.get("author") or []) if name],
        "year": year,
        "doi": clean_doi(item.get("DOI")),
        "venue": (item.get("container-title") or [""])[0],
        "url": item.get("URL"),
    }


def query_crossref(reference):
    candidates, errors = [], []
    if reference.get("generated_doi"):
        payload, error = http_get("https://api.crossref.org/works/" + urllib.parse.quote(reference["generated_doi"], safe=""))
        if error:
            errors.append(f"crossref_doi:{error}")
        elif payload.get("message"):
            candidates.append(crossref_candidate(payload["message"], "doi"))
    title = title_query_text(reference)
    if title:
        payload, error = http_get("https://api.crossref.org/works?" + urllib.parse.urlencode(
            {"query.title": title, "rows": SEARCH_SIZE}))
        if error:
            errors.append(f"crossref_title:{error}")
        else:
            candidates += [crossref_candidate(item, "title") for item in (payload.get("message") or {}).get("items", [])]
    return candidates, errors


def semantic_scholar_candidate(item, query_kind):
    return {
        "source": "semantic_scholar",
        "query_kind": query_kind,
        "source_id": item.get("paperId") or item.get("corpusId"),
        "title": item.get("title") or "",
        "authors": [author.get("name", "") for author in item.get("authors") or [] if author.get("name")],
        "year": item.get("year"),
        "doi": clean_doi((item.get("externalIds") or {}).get("DOI")),
        "venue": item.get("venue") or "",
        "url": item.get("url"),
    }


def query_semantic_scholar(reference):
    candidates, errors = [], []
    fields = "paperId,corpusId,title,authors,year,venue,externalIds,url"
    base = "https://api.semanticscholar.org/graph/v1/paper/"
    if reference.get("generated_doi"):
        payload, error = http_get(base + "DOI:" + urllib.parse.quote(reference["generated_doi"], safe="")
                                  + "?" + urllib.parse.urlencode({"fields": fields}))
        if error:
            errors.append(f"semantic_scholar_doi:{error}")
        elif payload:
            candidates.append(semantic_scholar_candidate(payload, "doi"))
    title = title_query_text(reference)
    if title:
        payload, error = http_get(base + "search?" + urllib.parse.urlencode(
            {"query": title, "limit": SEARCH_SIZE, "fields": fields}))
        if error:
            errors.append(f"semantic_scholar_title:{error}")
        else:
            candidates += [semantic_scholar_candidate(item, "title") for item in payload.get("data") or []]
    return candidates, errors


def arxiv_candidate(entry, query_kind):
    def find(tag):
        return entry.findtext(tag, default="", namespaces=ARXIV_NS)

    authors = [author.findtext("atom:name", default="", namespaces=ARXIV_NS) for author in entry.findall("atom:author", ARXIV_NS)]
    return {
        "source": "arxiv",
        "query_kind": query_kind,
        "source_id": find("atom:id"),
        "title": " ".join(find("atom:title").split()),
        "authors": [author for author in authors if author],
        "year": extract_year(find("atom:published")),
        "doi": clean_doi(find("arxiv:doi")),
        "venue": "arXiv",
        "url": find("atom:id"),
    }


def arxiv_title_query(title, max_terms=8):
    """arXiv title search: the first eight distinct content words, all required."""
    tokens = []
    for token in normalize_text(title).split():
        if len(token) > 2 and not token.isdigit() and token not in QUERY_STOPWORDS and token not in tokens:
            tokens.append(token)
    return " AND ".join(f"ti:{token}" for token in tokens[:max_terms]) or None


def query_arxiv(reference):
    candidates, errors = [], []

    def search(params, kind):
        text, error = http_get("https://export.arxiv.org/api/query?" + urllib.parse.urlencode(params), as_json=False)
        if error:
            errors.append(f"arxiv_{kind}:{error}")
            return
        try:
            candidates.extend(arxiv_candidate(entry, kind) for entry in ET.fromstring(text).findall("atom:entry", ARXIV_NS))
        except ET.ParseError as exc:
            errors.append(f"arxiv_{kind}:parse_error:{exc}")

    if reference.get("generated_arxiv_id"):
        search({"id_list": reference["generated_arxiv_id"], "max_results": SEARCH_SIZE}, "id")
    query = arxiv_title_query(title_query_text(reference))
    if query:
        search({"search_query": query, "start": 0, "max_results": SEARCH_SIZE,
                "sortBy": "relevance", "sortOrder": "descending"}, "title")
    return candidates, errors


QUERY_FUNCTIONS = {
    "openalex": query_openalex,
    "crossref": query_crossref,
    "semantic_scholar": query_semantic_scholar,
    "arxiv": query_arxiv,
}


# Scoring and classification

def dedupe_candidates(candidates):
    seen, unique = set(), []
    for candidate in candidates:
        key = (candidate.get("source"),
               candidate.get("source_id") or candidate.get("doi") or normalize_text(candidate.get("title")))
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def score_candidate(reference, candidate):
    title_score = title_similarity(reference.get("generated_title") or "", candidate.get("title") or "")
    author_score_value, first_author_match = author_score(
        reference.get("generated_authors_text") or reference.get("generated_authors") or [],
        candidate.get("authors") or [],
    )
    candidate_arxiv_id = clean_arxiv_id(candidate.get("source_id")) or clean_arxiv_id(candidate.get("url"))
    doi_status = identifier_status(reference.get("generated_doi"), candidate.get("doi"))
    arxiv_status = identifier_status(reference.get("generated_arxiv_id"), candidate_arxiv_id)
    return {
        **candidate,
        "title_match_score": round(title_score, 4),
        "author_match_score": round(author_score_value, 4),
        "raw_bibliographic_match_score": round(
            raw_citation_similarity(reference.get("raw_generated_reference") or "", candidate), 4),
        "first_author_match": first_author_match,
        "used_raw_title_fallback": False,
        "used_raw_author_fallback": False,
        "year_match_status": year_match_status(reference.get("generated_year"), candidate.get("year")),
        "doi_match_status": doi_status,
        "arxiv_match_status": arxiv_status,
        "identifier_match_status": (
            "exact" if "exact" in (doi_status, arxiv_status)
            else "conflicting" if "conflicting" in (doi_status, arxiv_status) else "missing"
        ),
        "match_confidence": round(0.55 * title_score + 0.45 * author_score_value, 4),
    }


def is_verified_match(candidate):
    return (candidate["title_match_score"] >= TITLE_EXISTS_THRESHOLD
            and candidate["author_match_score"] >= AUTHOR_EXISTS_THRESHOLD
            and candidate.get("first_author_match"))


def is_possible_match(candidate):
    return (candidate["title_match_score"] > POSSIBLE_TITLE_THRESHOLD
            and candidate["author_match_score"] > POSSIBLE_AUTHOR_THRESHOLD)


def first_match(reference, scored, test):
    """The first of the top-ranked candidates that passes the test (needs a title and authors)."""
    if not (reference.get("generated_title") and reference.get("generated_authors")):
        return None
    return next((c for c in scored[:CLASSIFICATION_CANDIDATE_LIMIT] if test(c)), None)


def metadata_status(best):
    major = sum((best["title_match_score"] < TITLE_EXISTS_THRESHOLD,
                 best["author_match_score"] < AUTHOR_EXISTS_THRESHOLD,
                 best["year_match_status"] == "conflicting",
                 best["doi_match_status"] == "conflicting",
                 best.get("arxiv_match_status") == "conflicting"))
    if major:
        return "metadata_major_errors"
    return "metadata_minor_errors" if best["year_match_status"] in {"near", "missing"} else "metadata_correct"


def classify_match(reference, scored, errors):
    """Return (classification, matched candidate) for ranked candidates; see the module docstring."""
    def result(resolver_status, metadata, existence):
        return {"resolver_status": resolver_status, "existence_status": existence, "metadata_status": metadata}

    verified = first_match(reference, scored, is_verified_match)
    if verified:
        return result("title_author_match", metadata_status(verified), "verified_real_paper"), verified
    best = scored[0] if scored else None
    if not reference.get("generated_title") or not reference.get("generated_authors"):
        return result("metadata_insufficient_for_title_author_check", "metadata_insufficient", "unresolved"), best
    if not scored:
        if errors:
            return result("api_error_no_candidate", "metadata_insufficient", "unresolved"), None
        return result("no_candidate_found", "metadata_insufficient", "likely_hallucinated"), None
    possible = first_match(reference, scored, is_possible_match)
    if possible:
        return result("possible_title_author_match", metadata_status(possible), "possible_real_paper"), possible
    return result("low_confidence_candidates", metadata_status(best), "unresolved"), best


def temporal_status(focal_year, generated_year):
    if focal_year is None:
        return "missing_focal_year"
    if generated_year is None:
        return "missing_reference_year"
    return "future_reference" if generated_year > focal_year else "temporally_valid"


CANDIDATE_SCORE_FIELDS = ("match_confidence", "title_match_score", "author_match_score", "raw_bibliographic_match_score",
                          "first_author_match", "used_raw_title_fallback", "used_raw_author_fallback",
                          "year_match_status", "doi_match_status", "arxiv_match_status", "identifier_match_status")


def candidate_summary(candidate):
    if not candidate:
        return {
            **dict.fromkeys(("matched_source", "matched_source_id", "matched_title", "matched_authors", "matched_year",
                             "matched_doi", "matched_venue", "matched_url")),
            "match_confidence": 0.0, "title_match_score": 0.0, "author_match_score": 0.0,
            "raw_bibliographic_match_score": 0.0, "first_author_match": False,
            "used_raw_title_fallback": False, "used_raw_author_fallback": False,
            **dict.fromkeys(("year_match_status", "doi_match_status", "arxiv_match_status", "identifier_match_status"),
                            "missing"),
        }
    return {
        **{f"matched_{key}": candidate.get(key) for key in ("source", "source_id", "title", "authors", "year",
                                                            "doi", "venue", "url")},
        **{key: candidate.get(key) for key in CANDIDATE_SCORE_FIELDS},
    }


def ground_truth_summary(scored):
    """Whether the best-ranked reference of the focal paper's real reference list is a verified match."""
    if not scored:
        return {"ground_truth_match_status": "no_ground_truth_references", "ground_truth_match_confidence": 0.0,
                "ground_truth_matched_paper_id": None, "ground_truth_title_score": 0.0, "ground_truth_author_score": 0.0}
    best = scored[0]
    return {
        "ground_truth_match_status": "in_focal_reference_list" if is_verified_match(best) else "not_in_focal_reference_list",
        "ground_truth_match_confidence": best["match_confidence"],
        "ground_truth_matched_paper_id": best.get("source_id"),
        "ground_truth_title_score": best["title_match_score"],
        "ground_truth_author_score": best["author_match_score"],
    }


def ranked(reference, candidates):
    scored = [score_candidate(reference, candidate) for candidate in dedupe_candidates(candidates)]
    return sorted(scored, key=lambda candidate: candidate["match_confidence"], reverse=True)


def build_resolved_record(reference, focal_metadata, ground_truth, external_candidates, source_errors,
                          source_candidate_counts, queried_sources):
    """Classify one generated reference from all candidates found so far."""
    focal = focal_metadata.get(reference["focal_paper_id"], {})
    scored_external = ranked(reference, external_candidates)
    scored_ground_truth = ranked(reference, ground_truth.get(reference["focal_paper_id"], []))
    classification, matched = classify_match(reference, scored_external, source_errors)
    return {
        **reference,
        "focal_title": focal.get("focal_title"),
        "focal_year": focal.get("focal_year"),
        "focal_doi": focal.get("focal_doi"),
        "resolved_at": utc_now(),
        **classification,
        "temporal_status": temporal_status(focal.get("focal_year"), reference.get("generated_year")),
        **candidate_summary(matched),
        **ground_truth_summary(scored_ground_truth),
        "candidate_count": len(scored_external),
        "candidate_sources": sorted({c.get("source") for c in scored_external if c.get("source")}),
        "queried_sources": list(queried_sources),
        "source_candidate_counts": dict(source_candidate_counts),
        "source_errors": list(source_errors),
        "top_candidates": scored_external[:5],
        "ground_truth_top_candidates": scored_ground_truth[:3],
    }


def should_stop_cascade(record):
    return record.get("existence_status") in {"verified_real_paper", "possible_real_paper"}


CSV_FIELDS = [
    "run_id", "focal_paper_id", "prompt_id", "generated_reference_index", "focal_year", "generated_year",
    "temporal_status", "resolver_status", "existence_status", "metadata_status", "match_confidence",
    "matched_source", "matched_source_id", "title_match_score", "author_match_score",
    "raw_bibliographic_match_score", "used_raw_title_fallback", "used_raw_author_fallback", "year_match_status",
    "doi_match_status", "arxiv_match_status", "identifier_match_status", "generated_title", "generated_authors",
    "generated_authors_text", "generated_doi", "generated_arxiv_id", "generated_venue", "matched_title",
    "matched_authors", "matched_year", "matched_doi", "matched_venue", "ground_truth_match_status",
    "ground_truth_match_confidence", "ground_truth_matched_paper_id", "ground_truth_title_score",
    "ground_truth_author_score", "candidate_count", "queried_sources", "pending_sources", "source_error_history",
    "candidate_sources", "source_candidate_counts", "source_errors", "top_candidates",
]
CSV_JSON_FIELDS = ("generated_authors", "matched_authors", "queried_sources", "pending_sources", "source_error_history",
                   "candidate_sources", "source_candidate_counts", "source_errors", "top_candidates")


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            row = {field: record.get(field) for field in CSV_FIELDS}
            writer.writerow({**row, **{field: json.dumps(row[field], ensure_ascii=False) for field in CSV_JSON_FIELDS}})
