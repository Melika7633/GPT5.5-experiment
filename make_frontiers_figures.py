#!/usr/bin/env python3
"""Regenerate the manuscript figures (main Figs 2-4 and supplementary Figs S1-S3) and check their size and resolution.

Each figure script runs in its own process (they set global matplotlib styles). Fig 4 embeds the
panels (A)-(C) written by plot_model_defined_common_feature_two_panel.py, so that script runs first.
Output: outputs/final_figures/frontiers/FigN.tif (LZW), FigN.pdf (vector) and FigN.png (preview).
Fig 1, the study-flow diagram, was drawn in PowerPoint.

    python3 make_frontiers_figures.py
"""

import subprocess
import sys
from pathlib import Path

from PIL import Image

from common.frontiers_figures import DPI, FRONTIERS_DIR, MAX_HEIGHT_MM, MM_PER_IN, WIDTHS_MM


HERE = Path(__file__).resolve().parent
SCRIPTS = [
    "generation.plot_effort_inflation",                                  # Fig 2
    "predefined_features.plot_feature_summary_composite",                # Fig 3
    "model_defined_features.plot_model_defined_common_feature_two_panel",  # Fig 4 (A)-(C)
    "model_defined_features.plot_model_defined_feature_story",           # Fig 4
    "generation.plot_fig2c_temporal_integrated_normalization",           # Supplementary Fig S1 (output Fig5)
    "generation.plot_fig5_candidate_confidence",                         # Supplementary Fig S2 (output Fig6)
    "model_defined_features.plot_judge_robustness_frontiers",            # Supplementary Fig S3 (output FigS1)
]
FIGURES = ("Fig2", "Fig3", "Fig4", "Fig5", "Fig6", "FigS1")


def check(name):
    image = Image.open(FRONTIERS_DIR / f"{name}.tif")
    width_mm, height_mm = image.width / DPI * MM_PER_IN, image.height / DPI * MM_PER_IN
    dpi = round(image.info["dpi"][0])
    ok = dpi == DPI and round(width_mm) in {round(w) for w in WIDTHS_MM.values()} and height_mm <= MAX_HEIGHT_MM
    print(f"{'OK ' if ok else 'FAIL'} {name}: {width_mm:.0f} x {height_mm:.0f} mm, {dpi} dpi")
    return ok


def main():
    for script in SCRIPTS:
        print(f"\n=== {script}", flush=True)
        subprocess.run([sys.executable, "-m", script], cwd=HERE, check=True)
    print()
    if not all([check(name) for name in FIGURES]):
        raise SystemExit("Some figures do not meet the Frontiers size/resolution requirements")


if __name__ == "__main__":
    main()
