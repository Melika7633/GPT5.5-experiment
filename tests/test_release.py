"""Regression checks for release paths, fixed-cohort inputs and offline safety."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generation import build_title_overlap
from common import cohort_plot_data
from common import judge_config
from common import outcome_data
from generation import resolve
from generation import run
from predefined_features import run_predefined_feature_analysis


class ReleaseTests(unittest.TestCase):
    def test_overlap_normalization_preserves_published_unicode_rules(self):
        self.assertEqual(build_title_overlap.normalize_title("S&P 100: NH₃–H₂O"), "s p 100 nh3 h2o")
        self.assertEqual(build_title_overlap.normalize_title("Théorie"), "th orie")

    def test_archive_installation_skips_macos_metadata_and_does_not_overwrite(self):
        import zipfile
        import reproduce
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            downloads, root = base / "downloads", base / "checkout"
            downloads.mkdir()
            root.mkdir()
            for name, member in (("low.zip", "low/sample.txt"), ("high.zip", "high/sample.txt"),
                                 ("feature_analysis.zip", "feature_analysis/sample.txt")):
                with zipfile.ZipFile(downloads / name, "w") as archive:
                    archive.writestr(member, "evidence")
                    archive.writestr("__MACOSX/._sample", "metadata")
            with zipfile.ZipFile(downloads / "derived_tables.zip", "w") as archive:
                for name in ("cohort.csv", "low_high_title_overlap_full_1k.csv", "metadata_status_1k.csv",
                             "successful_match_sources_1k.csv", "unresolved_source_searches_1k.csv"):
                    archive.writestr("derived_tables/" + name, "header\n")
            destinations = {"low.zip": root / "outputs", "high.zip": root / "outputs", "feature_analysis.zip": root}
            with patch.multiple(reproduce, HERE=root, ARCHIVES=destinations):
                reproduce.install_archives(downloads)
                self.assertTrue((root / "cohort.csv").is_file())
                self.assertFalse((root / "outputs/__MACOSX").exists())
                (root / "cohort.csv").write_text("local modification")
                with self.assertRaises(FileExistsError):
                    reproduce.install_archives(downloads)
                self.assertEqual((root / "cohort.csv").read_text(), "local modification")

    def test_predefined_writer_and_reader_share_published_layout(self):
        for condition in ("low", "high"):
            for start in (0, 200, 400, 600, 800):
                out = run_predefined_feature_analysis.batch_dir(condition, start, 200)
                self.assertEqual(out, judge_config.predefined_batch_dir(condition, start))
                self.assertEqual(out.name, condition if start == 0 else f"{start:04d}-{start+199:04d}")
        for start, count in ((1, 200), (0, 100), (1000, 200)):
            with self.assertRaises(ValueError):
                run_predefined_feature_analysis.batch_dir("low", start, count)

    def test_fixed_cohort_does_not_require_unselected_abstracts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            abstracts = root / "abstracts"
            abstracts.mkdir()
            fields = ["PaperID", "PaperTitle", "Author_Name", "Year", "Journal_Name",
                      "DOI", "Field_Type", "top_field"]
            focal = root / "focal.csv"
            with focal.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for index in range(1000):
                    writer.writerow({**dict.fromkeys(fields, "value"), "PaperID": str(index)})
                    (abstracts / f"{index}.txt").write_text("Abstract text")
            cohort = root / "cohort.csv"
            cohort.write_text("PaperID\n" + "\n".join(map(str, range(1001))) + "\n")
            with patch.multiple(run, COHORT_PATH=cohort, FOCAL_PAPERS_PATH=focal, ABSTRACTS_DIR=abstracts):
                rows = run.load_cohort()
            self.assertEqual([r["PaperID"] for r in rows], list(map(str, range(1000))))

    def test_metrics_do_not_need_focal_covariates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "outputs/low/resolved/all/references.csv"
            path.parent.mkdir(parents=True)
            path.write_text("focal_paper_id,existence_status,temporal_status,focal_year\n"
                            "123,verified_real_paper,temporally_valid,2000\n"
                            "123,possible_real_paper,future_reference,2000\n"
                            "456,unresolved,missing_reference_year,2001\n")
            with patch.object(outcome_data, "HERE", root):
                frame = outcome_data.load_paper_metrics("low").set_index("focal_paper_id")
            self.assertEqual(frame.loc["123", "existence_rate"], 0.5)
            self.assertEqual(frame.loc["123", "future_rate"], 0.5)
            self.assertEqual(frame.loc["456", "existence_rate"], 0.0)
            self.assertEqual(frame.loc["456", "temporal_validity_rate"], 1.0)

    def test_missing_checkpoints_fail_before_any_output_or_search(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parsed = root / "outputs/low/parsed"
            parsed.mkdir(parents=True)
            (parsed / "paper.json").write_text(json.dumps({"metadata_enrichment_method": resolve.ENRICHMENT_METHOD}))
            rows = [{"sample_index": i, "parsed_output_path": "paper.json"} for i in range(200)]
            reference = {"focal_paper_id": "paper", "prompt_id": "P00", "generated_reference_index": 1}
            with patch.object(resolve, "HERE", root), patch.object(resolve, "load_jsonl", return_value=rows), \
                 patch.object(resolve.core, "flatten_parsed_references", return_value=[reference]), \
                 patch.object(resolve.core, "load_focal_metadata", return_value={}), \
                 patch.object(resolve.core, "load_ground_truth", return_value={}), \
                 patch.object(resolve, "search") as search:
                with self.assertRaisesRegex(FileNotFoundError, "checkpoints are absent"):
                    resolve.resolve_block("low", 0, True)
                search.assert_not_called()
            self.assertFalse((root / "outputs/low/resolved").exists())


if __name__ == "__main__":
    unittest.main()
