# Higher Reasoning Effort Improves Reference Verifiability and Changes Reasoning Summaries in GPT-5.5


## Repository layout

| Folder | Contents |
|---|---|
| `generation/` | GPT-5.5 reference generation (`run.py`, prompt template `P00_vanilla.txt`), GPT-4.1 mini metadata extraction (`enrich*.py`), reference resolution (`resolve*.py`), HPC job files, and the reference analyses: Fig 2, Figs S1–S2, metadata and source tables |
| `predefined_features/` | GPT-5.5 and Claude scoring of the four predefined features, paired statistics, Table S2 and Fig 3 |
| `model_defined_features/` | Feature discovery and scoring, the fixed mapping of the 11 constructs, Fig 4 and the judge comparison (Fig S3) |
| `common/` | Shared helpers: data paths, cohort and score loaders, outcome metrics, figure style |
| `figures/` | Submitted figure files (TIFF) |
| `tests/` | Regression tests |
| `reproduce.py`, `make_frontiers_figures.py` | Run all table analyses / regenerate all figures |

Run every command from the repository's top folder. Scripts inside the folders run as modules,
for example `python3 -m generation.run`. Data installed from the archives goes to `outputs/`,
`feature_analysis/` and `cohort.csv` in the top folder.

## Reproduce the tables and figures from archived data

Use Python 3.12 and a fresh checkout. Install the dependencies:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Download the four data archives of the study record https://doi.org/10.5281/zenodo.23361321
(`low.zip`, `high.zip`, `feature_analysis.zip` and `derived_tables.zip`) into a directory
outside the checkout, then rebuild the tables:

```sh
python3 reproduce.py --archives /path/to/downloaded/archives --tables-only
```

The installer skips macOS metadata and refuses to overwrite differing files.
The tables-only workflow checks score/cohort coverage, verifies retained resolver
candidates, rebuilds title-overlap and source/metadata tables, aggregates predefined
scores, and runs paired statistics and verification-extreme analysis. It does not
render figures. No API credentials, network searches, retrieval caches or execution
checkpoints are needed.
After installation, `python3 reproduce.py --tables-only` repeats these analyses.
Use a fresh checkout when reinstalling archives after regenerated files have changed.

### Figures

Fig 2 and Supplementary Fig S1 also need two tables from the source dataset of
Algaba et al. (2025), https://doi.org/10.5281/zenodo.15124610 (`data.zip`):
Fig 2 takes the research field of each focal paper from the focal-paper table, and
Fig S1 the publication years of the observed references. Copy both into the top
folder under the names the code uses, then run `reproduce.py` without `--tables-only`
(add `--archives /path/to/downloaded/archives` if the archives are not installed yet):

```sh
unzip /path/to/data.zip data/focal_papers.csv data/ground_truth_references.csv -d /path/to/algaba
cp /path/to/algaba/data/focal_papers.csv focal_papers_AA.csv
cp /path/to/algaba/data/ground_truth_references.csv ground_truth_references_AA.csv
python3 reproduce.py
```

This runs the table analyses and then `make_frontiers_figures.py`, which regenerates all figures.

### Inputs and outputs

- `cohort.csv`: original 10,000-paper seeded draw; first 1,000 rows are the fixed study cohort.
- `outputs/{low,high}/`: archived generation outputs and final reference CSV/JSONL.
- `feature_analysis/`: archived GPT-5.5 and Claude scores and discovered definitions.
- `model_defined_features/full_1k_low_high_feature_pairs.json`: fixed mapping of eleven constructs, supplied with the code.
- `focal_papers_AA.csv`, `ground_truth_references_AA.csv` and `abstracts/`: `data/focal_papers.csv`,
  `data/ground_truth_references.csv` and `data/abstracts/` from Algaba et al. (2025), not
  redistributed here. Figs 2 and S1 need the two tables; new generation runs also need the abstracts.

The focal papers and observed bibliographies come from Algaba et al. (2025),
doi:10.5281/zenodo.15124610, CC BY 4.0. Rendered generation prompts are retained in the evidence archives.

The bootstrap resamples paired focal papers in the published cohort order:
10,000 resamples, seed 20260917. Possible and unresolved references remain in the
observed-reference denominator. The high-effort paper with six references keeps
its observed denominator; the existing 999-pair sensitivity analysis is retained.

| Paper element | Reproduction code / output |
|---|---|
| Main +6.53 pp, 95% CI 5.78–7.28 | `predefined_features/run_paired_nonparametric_analysis.py`; GPT predefined `analysis_tables/` |
| Fig 2 | `generation/plot_effort_inflation.py`; `generation/build_title_overlap.py` |
| Fig 3 | `predefined_features/plot_feature_summary_composite.py` and imported plotting helpers |
| Fig 4 | `model_defined_features/plot_model_defined_common_feature_two_panel.py`, then `plot_model_defined_feature_story.py` |
| Supplementary metadata/source counts | `generation/analyze_metadata_status_1k.py`, `generation/analyze_resolver_sources_1k.py` |
| Table S2 | `predefined_features/analyze_verification_extremes.py` |
| Table S3 | Fixed mapping JSON and archived batch-specific feature definitions |
| Table S4 | Illustrative excerpts in the manuscript; raw summaries and scores are archived |
| Fig S1 | `generation/plot_fig2c_temporal_integrated_normalization.py` produces `Fig5.*` |
| Fig S2 | `generation/plot_fig5_candidate_confidence.py` produces `Fig6.*` |
| Fig S3 | `model_defined_features/plot_judge_robustness_frontiers.py` produces `FigS1.*` |

Figure files are written to `outputs/final_figures/frontiers/` as PDF, TIFF and PNG.
Figure 1 was manually prepared in PowerPoint; the included TIFF (`figures/Fig1.tif`) is a reference
artifact and is not regenerated by this workflow. Arial is used for submission;
other fonts/rendering libraries can alter appearance and file bytes. Pinned
versions are an environment specification, not a claim of byte-for-byte identity.

### Classification evidence and harmonization

`python3 -m generation.verify_archived_resolution` recomputes retained-candidate scores,
existence/metadata classifications and temporal labels without writing to the
archive. Only retained candidates are available; this does not replay historical
retrieval or demonstrate that the searches exhausted all possible publications.
Source failures/waivers remain part of the archived record's provenance.
The historical resolver can emit `likely_hallucinated` when no candidate was found
and no search failed; that status is not used in the published cohort and is not
proof that a paper does not exist. Published analyses use the recorded labels.

The mapping JSON records the semantic matching rule, batch-local IDs and scope
notes. Eleven constructs are retained when a corresponding low/high pair exists
in all five discovery batches. Each paper uses its own condition/batch feature
definition; no paper is re-scored or assigned absence for features from other
batches. Treat the mapping as a frozen analysis input.

Judge-comparison panel A uses mean ± 1.96 standard errors, computed across the
papers included in each condition/judge comparison. Missing summaries are scored
1; Claude refusals remain missing. Scores ≥3 indicate presence.

## Optional new API experiments

Run new experiments in a separate checkout/output tree so the published evidence
is preserved. Model responses and present-day bibliographic databases may differ;
new calls reproduce the procedure, not the exact recorded observations.

Generation requires the fixed cohort (`cohort.csv`), `focal_papers_AA.csv` and `abstracts/`
(see Inputs and outputs):

```sh
python3 -m generation.run --effort low --start 0 --count 200 --dry-run
python3 -m generation.run --effort low --start 0 --count 200 --submit
python3 -m generation.run --effort low --status
python3 -m generation.run --effort low --collect
```

Repeat for starts 200, 400, 600, 800 and condition high. `--submit` makes paid
requests using `OPENAI_API_KEY`. Metadata extraction uses `python3 -m generation.enrich`
(`--condition` and `--submit/--status/--collect`).
Live resolution uses `python3 -m generation.resolve --condition low --start 0 --count 200`,
then `--merge-blocks` after all blocks finish; it requires source credentials
described in that script. Local caches/checkpoints are generated for new runs.
`--no-network` requires complete local checkpoints and fails if they are absent;
it is not the public archived-data reproduction command.

Predefined GPT scoring uses the same five 200-paper blocks:

```sh
python3 -m predefined_features.run_predefined_feature_analysis --condition low --start 0 --count 200 --dry-run
```

Switch to `--submit/--status/--collect` for new scoring.
The first block writes directly to the condition's score directory; later blocks
write to `cohort_batches/<range>/`, matching the published data layout.
Discovery uses `python3 -m model_defined_features.run_model_defined_feature_discovery --condition low`;
model-defined scoring uses `python3 -m model_defined_features.run_model_defined_feature_scoring --condition low --batch batch_001`.
Both support `--dry-run/--submit/--status/--collect`. Claude equivalents use the
two `run_claude_*.py` scripts (in `predefined_features/` and `model_defined_features/`) and `ANTHROPIC_API_KEY`.
New discoveries require a new reviewed harmonization mapping before comparison.

## License and tests

This release uses CC BY 4.0, the license of the study data record (see LICENSE).
Cite the paper, the study archive and the source dataset.

Run `python3 -m unittest discover -s tests` for path, input and checkpoint regression
checks, and `python3 reproduce.py --tables-only` for archived table reproduction. Numerical
reference checks include 20,000/19,986 references, 15,651/16,946 verified,
138/101 empty summaries, 7,334 identity-supported shared pairs, and the paired
effect and interval above.
