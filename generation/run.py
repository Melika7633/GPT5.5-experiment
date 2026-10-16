"""Generate GPT-5.5 reference lists for the 1,000-paper cohort with the OpenAI Batch API.

Each reasoning-effort condition (low, high) is run separately on the same cohort,
in ranges of cohort indices (the study used five batches of 200 papers):

    python3 -m generation.run --effort low --start 0 --count 200 --dry-run   # render prompts and manifest
    python3 -m generation.run --effort low --start 0 --count 200 --submit    # submit them as one batch
    python3 -m generation.run --effort low --status                          # check the batch
    python3 -m generation.run --effort low --collect                         # download and parse responses

Inputs in the repository root: focal_papers_AA.csv, abstracts/<PaperID>.txt and cohort.csv
(created from the seeded draw if absent); prompt template: generation/P00_vanilla.txt.
--submit, --status and --collect read the API key from OPENAI_API_KEY.
Outputs are written to outputs/<effort>/: prompts/, raw/, parsed/, traces/,
manifest.jsonl, results.jsonl, and batches/<batch id>/ for each submitted batch.
"""

import argparse
import csv
import html
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
FOCAL_PAPERS_PATH = HERE / "focal_papers_AA.csv"
ABSTRACTS_DIR = HERE / "abstracts"
PROMPT_TEMPLATE_PATH = HERE / "generation" / "P00_vanilla.txt"
COHORT_PATH = HERE / "cohort.csv"
OUTPUT_ROOT = HERE / "outputs"
API_URL = "https://api.openai.com/v1"

MODEL = "gpt-5.5-2026-04-23"
REASONING_SUMMARY = "auto"
N_REFERENCES = 20
PROMPT_ID = "P00"
SEED = 42
# cohort.csv holds the original seeded 10,000-paper draw; the experiment uses its first 1,000.
SOURCE_COHORT_SIZE = 10000
TARGET_COHORT_SIZE = 1000
TERMINAL_BATCH_STATUSES = {"failed", "cancelled", "expired"}

TAG_RE = re.compile(r"<[^>]+>")
NUMBER_RE = re.compile(r"^\s*(?:\[\d+\]|\d+[.)])\s+")
BULLET_RE = re.compile(r"^\s*[-*]\s+")
PREFIX_RE = re.compile(r"^\s*(?:[-*]|\[\d+\]|\d+[.)])\s+")
TAIL_RE = re.compile(
    r"^\s*(?:if\s+you\s+want|i\s+can\s+also|would\s+you\s+like|"
    r"these\s+references\s+(?:collectively|cover)|together\s+they\s+form)\b",
    re.IGNORECASE,
)


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")


def load_jsonl(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:  # not str.splitlines(): citation text may contain U+0085
        return [json.loads(line) for line in handle if line.strip()]


def merge_records(path, new_records):
    """Add or replace per-paper rows in a JSONL file, ordered by cohort index."""
    records = {row["focal_paper_id"]: row for row in load_jsonl(path)}
    records.update((row["focal_paper_id"], row) for row in new_records)
    write_jsonl(path, sorted(records.values(), key=lambda row: row["sample_index"]))


# Cohort and prompts

def clean_abstract(text):
    text = html.unescape(TAG_RE.sub(" ", text))
    return " ".join(text.split()).removeprefix("Abstract ").strip()


def load_papers(paper_ids=None):
    """Read the focal papers that have an abstract, in the order of focal_papers_AA.csv."""
    papers = []
    with FOCAL_PAPERS_PATH.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if paper_ids is not None and row["PaperID"] not in paper_ids:
                continue
            abstract_path = ABSTRACTS_DIR / f"{row['PaperID']}.txt"
            if not abstract_path.is_file():
                continue
            papers.append({
                "PaperID": row["PaperID"],
                "PaperTitle": row["PaperTitle"],
                "Author_Name": row["Author_Name"],
                "Year": row["Year"] or None,
                "Journal_Name": row["Journal_Name"] or None,
                "DOI": row["DOI"] or None,
                "Field_Type": row["Field_Type"] or None,
                "top_field": row["top_field"] or None,
                "Abstract": clean_abstract(abstract_path.read_text(encoding="utf-8")),
            })
    return papers


def load_cohort():
    """Return the first 1,000 papers of the seeded draw, identical for both effort conditions.

    The draw is made once with SEED and saved to cohort.csv; later runs reuse that file.
    """
    if COHORT_PATH.is_file():
        with COHORT_PATH.open(newline="", encoding="utf-8") as handle:
            ids = [row["PaperID"] for row in csv.DictReader(handle)][:TARGET_COHORT_SIZE]
        if len(ids) != TARGET_COHORT_SIZE or len(set(ids)) != TARGET_COHORT_SIZE:
            raise ValueError("The published cohort must contain 1,000 unique selected paper IDs")
        by_id = {paper["PaperID"]: paper for paper in load_papers(set(ids))}
        missing = [paper_id for paper_id in ids if paper_id not in by_id]
        if missing:
            raise ValueError(f"Missing metadata or abstracts for {len(missing)} selected papers: {missing[:5]}")
        cohort = [by_id[paper_id] for paper_id in ids]
    else:
        papers = load_papers()
        cohort = random.Random(SEED).sample(papers, SOURCE_COHORT_SIZE)
        fields = ("sample_index", "PaperID", "PaperTitle", "Author_Name", "Year",
                  "top_field", "Field_Type", "DOI", "Journal_Name")
        with COHORT_PATH.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, paper in enumerate(cohort):
                writer.writerow({"sample_index": index, **{key: paper[key] for key in fields[1:]}})
    return cohort[:TARGET_COHORT_SIZE]


def render_prompt(paper):
    return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8").format(
        paper_title=paper["PaperTitle"],
        paper_authors=paper["Author_Name"],
        paper_year=paper["Year"],
        paper_venue=paper["Journal_Name"],
        paper_abstract=paper["Abstract"],
        num_references=N_REFERENCES,
    )


# Response parsing

def response_text(response):
    """Join the text of the final message(s) in a Responses API body."""
    parts = []
    for item in response.get("output") or []:
        if item.get("type") == "message":
            parts.extend(x["text"] for x in item.get("content") or [] if isinstance(x.get("text"), str))
    return "\n".join(parts).strip()


def reasoning_summary(response):
    """Join the reasoning-summary blocks returned with the response."""
    parts = []
    for item in response.get("output") or []:
        if item.get("type") == "reasoning":
            parts.extend(x["text"].strip() for x in item.get("summary") or [] if x.get("text", "").strip())
    return "\n\n".join(parts)


def parse_json(text):
    """Return (object, source text) for a JSON object in the response, or None.

    Tries the whole response, then a fenced code block, then the first embedded object.
    """
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.append(fenced.group(1).strip())
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value, candidate
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, end = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value, text[match.start():end]
    return None


def parse_plain(text):
    """Split a numbered reference list into citations.

    Continuation lines are joined to the current citation and closing remarks end it.
    If no numbered item is found, every line starting with a list marker is a citation.
    """
    references, current = [], None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if NUMBER_RE.match(line):
            if current:
                references.append(current)
            current = NUMBER_RE.sub("", line).strip()
        elif current and TAIL_RE.match(line):
            references.append(current)
            current = None
        elif current and not BULLET_RE.match(line):
            current = f"{current} {line}".strip()
    if current:
        references.append(current)
    if not references:
        references = [PREFIX_RE.sub("", line).strip() for line in text.splitlines() if PREFIX_RE.match(line)]
    return [{"rank": i, "raw_citation": ref} for i, ref in enumerate(references, 1) if ref]


def parse_response(text):
    """Parse the final response as a JSON object if it contains one, otherwise as a plain-text list."""
    parsed = parse_json(text)
    if parsed:
        return parsed[0], parsed[1], "json_object"
    references = parse_plain(text)
    if not references:
        raise ValueError("Could not extract references from model output.")
    return {"references": references}, text.strip(), "plaintext_references"


def write_trace(path, paper_id, effort, model, prompt, summary, final, references):
    """Write a readable Markdown record of one generation."""
    rows = []
    for ref in references:
        citation = str(ref.get("raw_citation", "")).replace("|", "\\|")
        rows.append(f"| {ref.get('rank', '')} | {citation} |")
    rows = "\n".join(rows) or "_No parsed references found._"
    path.write_text(f"""# {paper_id} / {PROMPT_ID}

## Metadata

| Field | Value |
| --- | --- |
| Run ID | gpt55_1k_{effort} |
| Focal paper ID | {paper_id} |
| Prompt ID | {PROMPT_ID} |
| Model | {model} |
| Reasoning effort | {effort} |

## Rendered Prompt

{prompt.strip()}

## Reasoning Summary

{summary.strip() or '_No reasoning summary was returned._'}

## Final Response

{final.strip() or '_No final response was returned._'}

## Parsed References

| Rank | Citation |
| --- | --- |
{rows}
""", encoding="utf-8")


def process_output(record, response, effort, model):
    """Save the raw response, the parsed references and a Markdown trace for one paper."""
    prompt = (HERE / record["prompt_path"]).read_text(encoding="utf-8")
    final = response_text(response)
    summary = reasoning_summary(response)
    run_id = f"gpt55_1k_{effort}"
    # The raw response is saved before parsing, so it is kept even if parsing fails.
    write_json(HERE / record["raw_output_path"], {
        "run_id": run_id,
        "focal_paper_id": record["focal_paper_id"],
        "prompt_id": PROMPT_ID,
        "model_name": model,
        "reasoning_effort": effort,
        "created_at": now(),
        "messages": [{"role": "user", "content": prompt}],
        "rendered_prompt": prompt,
        "openai_request": {"model": model, "reasoning": {"effort": effort, "summary": REASONING_SUMMARY}},
        "openai_response": response,
        "openai_output": final,
        "reasoning_summary": summary,
    })
    parsed, extracted, parse_format = parse_response(final)
    write_json(HERE / record["parsed_output_path"], {
        "run_id": run_id,
        "focal_paper_id": record["focal_paper_id"],
        "prompt_id": PROMPT_ID,
        "model_name": model,
        "reasoning_effort": effort,
        "parsed_at": now(),
        "response_parse_format": parse_format,
        "extracted_response_text": extracted,
        "parsed_response": parsed,
    })
    references = parsed.get("references")
    write_trace(HERE / record["reasoning_trace_path"], record["focal_paper_id"], effort, model,
                prompt, summary, final, references if isinstance(references, list) else [])


# OpenAI Batch API

def api(path, data=None, content_type="application/json"):
    """Send one request to the OpenAI API (POST if data is given, otherwise GET); return the body."""
    request = urllib.request.Request(f"{API_URL}/{path}", data=data, headers={
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
        "Content-Type": content_type,
    })
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OpenAI API error {exc.code}: {exc.read().decode('utf-8', errors='replace')}") from exc


def api_json(path, payload=None):
    return json.loads(api(path, None if payload is None else json.dumps(payload).encode("utf-8")))


def upload_batch_file(path):
    """Upload a JSONL request file and wait until the Batch API can use it; return its file ID."""
    boundary = uuid.uuid4().hex
    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        "Content-Type: application/jsonl\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    file_id = json.loads(api("files", body, f"multipart/form-data; boundary={boundary}"))["id"]
    while (status := api_json(f"files/{file_id}").get("status")) not in (None, "processed"):
        if status == "error":
            raise RuntimeError(f"Uploaded file {file_id} could not be processed.")
        time.sleep(2)
    time.sleep(5)  # give the Batch API's file index time to catch up with the Files API
    return file_id


def save_state(run_dir, state):
    """Save the batch state as the current batch and in the batch's own folder."""
    write_json(run_dir / "batch.json", state)
    write_json(run_dir / "batches" / state["id"] / "batch.json", state)


def refresh_batch(run_dir):
    state = json.loads((run_dir / "batch.json").read_text(encoding="utf-8"))
    state = {**state, **api_json(f"batches/{state['id']}"), "checked_at": now()}
    save_state(run_dir, state)
    return state


def ensure_no_open_batch(run_dir):
    """Allow one open batch per output folder, so no request is submitted (and paid for) twice."""
    state_path = run_dir / "batch.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not state.get("collected_at") and state.get("status") not in TERMINAL_BATCH_STATUSES:
            raise ValueError("The current batch has not been collected. Check or collect it before submitting another.")


def submit_batch(run_dir, requests, endpoint, metadata, details):
    """Upload the requests as one batch; save its state and input under run_dir."""
    input_path = run_dir / "input.jsonl"
    write_jsonl(input_path, requests)
    batch = api_json("batches", {
        "input_file_id": upload_batch_file(input_path),
        "endpoint": endpoint,
        "completion_window": "24h",
        "metadata": metadata,
    })
    save_state(run_dir, {**batch, **details, "submitted_at": now(), "collected_at": None})
    input_path.rename(run_dir / "batches" / batch["id"] / "input.jsonl")
    print(f"Submitted {len(requests)} requests.")
    print(f"Batch ID: {batch['id']}")
    print(f"Status: {batch['status']}")


def show_status(run_dir):
    state = refresh_batch(run_dir)
    counts = state.get("request_counts") or {}
    print(f"Batch ID: {state['id']}")
    print(f"Status: {state['status']}")
    for key in ("completed", "failed", "total"):
        print(f"{key.capitalize()}: {counts.get(key, 0)}")


def download_batch(run_dir):
    """Download the output (and any error file) of the current batch; return its state and output rows."""
    state = refresh_batch(run_dir)
    if not state.get("output_file_id"):
        raise ValueError(f"Batch is {state['status']}; output is not ready yet.")
    batch_dir = run_dir / "batches" / state["id"]
    output_path = batch_dir / "output.jsonl"
    output_path.write_bytes(api(f"files/{state['output_file_id']}/content"))
    if state.get("error_file_id"):
        (batch_dir / "errors.jsonl").write_bytes(api(f"files/{state['error_file_id']}/content"))
    return state, load_jsonl(output_path)


def mark_collected(run_dir, state):
    state["collected_at"] = now()
    save_state(run_dir, state)


# Actions

def prepare(effort, start, count):
    """Render the prompts for cohort indices [start, start + count) and add them to the manifest."""
    stop = start + count
    if start < 0 or count < 1 or stop > TARGET_COHORT_SIZE:
        raise ValueError(f"Select a positive range within the {TARGET_COHORT_SIZE}-paper cohort.")
    run_dir = OUTPUT_ROOT / effort
    for name in ("prompts", "raw", "traces", "parsed"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    records = []
    for index, paper in enumerate(load_cohort()[start:stop], start=start):
        stem = f"{paper['PaperID']}__{PROMPT_ID}"
        record = {
            "sample_index": index,
            "focal_paper_id": paper["PaperID"],
            "prompt_id": PROMPT_ID,
            "prompt_path": f"outputs/{effort}/prompts/{stem}.txt",
            "raw_output_path": f"outputs/{effort}/raw/{stem}.json",
            "reasoning_trace_path": f"outputs/{effort}/traces/{stem}.md",
            "parsed_output_path": f"outputs/{effort}/parsed/{stem}.json",
        }
        (HERE / record["prompt_path"]).write_text(render_prompt(paper), encoding="utf-8")
        records.append(record)
    merge_records(run_dir / "manifest.jsonl", records)
    print(f"Prepared {len(records)} prompts in {run_dir}")


def submit(effort, start, count, overwrite, model):
    """Submit the prepared prompts for [start, start + count) as one batch."""
    run_dir = OUTPUT_ROOT / effort
    stop = start + count
    records = [row for row in load_jsonl(run_dir / "manifest.jsonl") if start <= row["sample_index"] < stop]
    if len(records) != count:
        raise ValueError(
            f"Expected {count} prepared prompts for indices {start}-{stop - 1}; "
            f"found {len(records)}. Run --dry-run first."
        )
    ensure_no_open_batch(run_dir)
    pending = [row for row in records if overwrite or not (HERE / row["raw_output_path"]).is_file()]
    if not pending:
        print("No requests to submit; all selected raw outputs already exist.")
        return
    requests = [{
        "custom_id": row["focal_paper_id"],
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "input": [{"role": "user", "content": (HERE / row["prompt_path"]).read_text(encoding="utf-8")}],
            "reasoning": {"effort": effort, "summary": REASONING_SUMMARY},
        },
    } for row in pending]
    submit_batch(run_dir, requests, "/v1/responses", {"experiment": "gpt55_1k", "effort": effort}, {
        "effort": effort,
        "model": model,
        "start": start,
        "count": len(pending),
        "references": N_REFERENCES,
        "summary": REASONING_SUMMARY,
    })


def collect(effort):
    """Download the current batch and save the raw, parsed and trace files for each paper."""
    run_dir = OUTPUT_ROOT / effort
    state, output = download_batch(run_dir)
    manifest = {row["focal_paper_id"]: row for row in load_jsonl(run_dir / "manifest.jsonl")}
    results = []
    for item in output:
        record = manifest[item["custom_id"]]
        response = item.get("response") or {}
        try:
            if response.get("status_code") != 200:
                raise RuntimeError(f"Batch request returned HTTP {response.get('status_code')}: {response.get('body')}")
            process_output(record, response["body"], effort, state["model"])
            status, error = "completed", None
        except Exception as exc:  # record the failure and continue with the remaining papers
            status, error = "error", str(exc)
        results.append({
            **record,
            "run_id": f"gpt55_1k_{effort}",
            "model_name": state["model"],
            "reasoning_effort": effort,
            "batch_id": state["id"],
            "status": status,
            "error_message": error,
            "finished_at": now(),
        })
    merge_records(run_dir / "results.jsonl", results)
    mark_collected(run_dir, state)
    completed = sum(row["status"] == "completed" for row in results)
    print(f"Collected {len(results)} responses: {completed} completed, {len(results) - completed} errors.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--effort", required=True, choices=("low", "high"), help="reasoning-effort condition")
    parser.add_argument("--start", type=int, default=0, help="first cohort index (default: 0)")
    parser.add_argument("--count", type=int, default=200, help="number of papers (default: 200)")
    parser.add_argument("--model", default=MODEL, help=f"generation model (default: {MODEL})")
    parser.add_argument("--overwrite", action="store_true", help="also resubmit papers that already have a raw output")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true", help="render prompts and the manifest (no API calls)")
    action.add_argument("--submit", action="store_true", help="submit the prepared prompts as one batch")
    action.add_argument("--status", action="store_true", help="show the progress of the current batch")
    action.add_argument("--collect", action="store_true", help="download and parse the finished batch")
    args = parser.parse_args()
    if args.dry_run:
        prepare(args.effort, args.start, args.count)
    elif args.submit:
        submit(args.effort, args.start, args.count, args.overwrite, args.model)
    elif args.status:
        show_status(OUTPUT_ROOT / args.effort)
    else:
        collect(args.effort)


if __name__ == "__main__":
    main()
