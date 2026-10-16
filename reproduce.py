"""Reproduce reported analyses and figures from archived data, without API calls."""
import argparse
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "generation.verify_archived_resolution",
    "predefined_features.aggregate_predefined_feature_scores",
    "generation.build_title_overlap",
    "generation.analyze_metadata_status_1k",
    "generation.analyze_resolver_sources_1k",
    "predefined_features.run_paired_nonparametric_analysis",
    "predefined_features.analyze_verification_extremes",
]
ARCHIVES = {"low.zip": HERE / "outputs", "high.zip": HERE / "outputs", "feature_analysis.zip": HERE}


def install_archives(directory):
    planned = {}
    for name, destination in ARCHIVES.items():
        with zipfile.ZipFile(directory / name) as archive:
            for member in archive.infolist():
                parts = PurePosixPath(member.filename).parts
                if member.is_dir() or "__MACOSX" in parts or ".DS_Store" in parts or any(p.startswith("._") for p in parts):
                    continue
                if not parts or PurePosixPath(member.filename).is_absolute() or ".." in parts or "\\" in member.filename:
                    raise ValueError(f"Unsafe archive path: {member.filename}")
                target = destination.joinpath(*parts)
                if target in planned:
                    raise ValueError(f"Duplicate archive destination: {target}")
                planned[target] = (directory / name, member.filename)
    with zipfile.ZipFile(directory / "derived_tables.zip") as archive:
        for name in ("cohort.csv", "low_high_title_overlap_full_1k.csv",
                     "metadata_status_1k.csv", "successful_match_sources_1k.csv",
                     "unresolved_source_searches_1k.csv"):
            destination = HERE / name if name == "cohort.csv" else HERE / "outputs" / name
            planned[destination] = (directory / "derived_tables.zip", "derived_tables/" + name)
    from contextlib import ExitStack
    with ExitStack() as stack:
        archives = {path: stack.enter_context(zipfile.ZipFile(path))
                    for path in {path for path, _ in planned.values()}}
        # Validate all destinations before writing; stream entries to bound memory use.
        import shutil
        for target, (archive_path, member) in planned.items():
            if target.exists():
                with archives[archive_path].open(member) as inp, target.open("rb") as existing:
                    while True:
                        left, right = inp.read(1024 * 1024), existing.read(1024 * 1024)
                        if left != right:
                            raise FileExistsError(f"Existing file differs from archive: {target}; use a fresh checkout")
                        if not left:
                            break
        for target, (archive_path, member) in planned.items():
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".installing")
                try:
                    with archives[archive_path].open(member) as inp, temporary.open("wb") as out:
                        shutil.copyfileobj(inp, out)
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
        print(f"Installed/checked {len(planned)} archived files; existing differing files were not overwritten", flush=True)


def validate_inputs(tables_only=False):
    from common.cohort_plot_data import cohort_ids, load_cohort_records
    from common.full_cohort_aligned_data import load_aligned_scores
    ids = cohort_ids()
    if len(ids) != 1000 or len(set(ids)) != 1000:
        raise ValueError("Expected 1,000 unique published cohort IDs")
    # Figs 2 and S1 need these two tables from Algaba et al. (2025), doi:10.5281/zenodo.15124610.
    required = [] if tables_only else ["focal_papers_AA.csv", "ground_truth_references_AA.csv"]
    for name in required:
        if not (HERE / name).is_file():
            raise FileNotFoundError(f"{name} is absent; copy it from doi:10.5281/zenodo.15124610 (see README)")
    for condition in ("low", "high"):
        load_cohort_records(condition)
        aligned = load_aligned_scores(condition)
        if len(aligned) != 11000 or set(aligned["focal_paper_id"]) != set(ids):
            raise ValueError(f"Unexpected aligned-score cohort: {condition}")
        for paper_id in ids:
            if not (HERE / "outputs" / condition / "raw" / f"{paper_id}__P00.json").is_file():
                raise FileNotFoundError(f"Missing raw response: {condition}/{paper_id}")
    print("Cohort, raw-response coverage and GPT score coverage validated", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, help="install the four data ZIPs from this directory")
    parser.add_argument("--tables-only", action="store_true", help="skip rendering figures")
    args = parser.parse_args()
    if args.archives:
        install_archives(args.archives)
    validate_inputs(args.tables_only)
    scripts = SCRIPTS + ([] if args.tables_only else ["make_frontiers_figures"])
    for script in scripts:
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, "-B", "-m", script], cwd=HERE, check=True)
    print("Archived-data reproduction completed; no model calls or bibliographic searches were made")


if __name__ == "__main__":
    main()
