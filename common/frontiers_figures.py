"""Frontiers submission export for matplotlib figures.

A figure keeps its designed layout; it is only scaled to the journal width.
Before scaling, every text is raised to MIN_FONT_PT and every data line to
MIN_LINE_PT *as measured at the final printed size*, so the saved file meets:

* width 85 mm (one column) or 180 mm (two columns), no taller than a page
* 300 dpi at final size, TIFF (LZW) plus a PNG preview
* smallest text >= 8 pt, data lines >= 2 pt at final size

Grid lines, tick marks, axis spines and patch outlines (bar/box edges) are
structural, not data lines, and are left as designed.
"""

import io
import warnings
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
import matplotlib.lines as mlines
import matplotlib.text as mtext
from PIL import Image

MM_PER_IN = 25.4
WIDTHS_MM = {"single": 85.0, "double": 180.0}
MAX_HEIGHT_MM = 230.0  # printable page height
DPI = 300
MIN_FONT_PT = 8.0
MIN_LINE_PT = 2.0
PAD_IN = 0.04
PANEL_LETTERS = {f"({c})" for c in "ABCDEFGH"}  # Frontiers style: (A), (B), ...
TEXT_COLOR = "#4B5563"  # dark grey: readable, but quieter than the data
FONT = "Arial"  # typeface of all Frontiers figures (Fig 1's PowerPoint uses it too)
# Effort colours shared by every Frontiers figure: low effort blue, high effort red.
EFFORT_COLORS = {"low": "#4C78A8", "high": "#E45756"}
# Light / base / dark shade of each effort colour for score levels 3, 4, 5.
SCORE_SHADES = {"low": ["#A9C3E0", "#4C78A8", "#223F63"],
                "high": ["#F4AEAD", "#E45756", "#8F1F24"]}

FRONTIERS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "final_figures" / "frontiers"


def _structural_lines(fig):
    lines = set()
    for ax in fig.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            lines.update(axis.get_gridlines())
            lines.update(axis.get_ticklines(minor=False))
            lines.update(axis.get_ticklines(minor=True))
    return lines


def enforce_minimums(fig, scale, min_font=MIN_FONT_PT, min_line=MIN_LINE_PT):
    """Set uniform text and minimum line sizes as they will print after scaling by `scale`.

    Every text prints at exactly `min_font` in regular weight; only panel
    letters stay bold, one point larger.
    """
    font_size, line_floor = min_font / scale, min_line / scale
    for ax in fig.get_axes():
        ax.tick_params(axis="both", labelsize=font_size, labelcolor=TEXT_COLOR)
    for text in fig.findobj(mtext.Text):
        label = text.get_text().strip()
        if not label:
            continue
        if label in PANEL_LETTERS:
            text.set_fontsize((min_font + 1) / scale)
            text.set_fontweight("bold")
            continue
        text.set_fontsize(font_size)
        text.set_fontweight("normal")
        # Mute neutral (black/grey) text so the data carries the visual weight;
        # colour-coded labels keep their colour.
        red, green, blue = mcolors.to_rgb(text.get_color())
        if max(red, green, blue) - min(red, green, blue) < 0.12:
            text.set_color(TEXT_COLOR)
    structural = _structural_lines(fig)
    for line in fig.findobj(mlines.Line2D):
        # Only data curves/trends (3+ points); 2-point segments are whiskers,
        # caps and reference lines, which stay as designed.
        if (line in structural or len(line.get_xdata()) < 3
                or line.get_linestyle() in ("None", "none", "", " ")):
            continue
        if 0 < line.get_linewidth() < line_floor:
            line.set_linewidth(line_floor)


def fit_to_width(fig, width_in, relayout=None, **minimums):
    """Iterate minimum enforcement until the tight bbox scale is stable; return scale."""
    scale = None
    for _ in range(6):
        if relayout:
            relayout()
        fig.canvas.draw()
        bbox = fig.get_tightbbox(fig.canvas.get_renderer())
        new_scale = width_in / (bbox.width + 2 * PAD_IN)
        if scale is not None and abs(new_scale - scale) < 1e-3:
            break
        scale = new_scale
        enforce_minimums(fig, scale, **minimums)
    check_font(fig)
    return scale


def check_font(fig):
    """Warn when any text would fall back to another typeface (e.g. Arial is not installed).

    A fallback font has different widths, so the figure would no longer match
    the submitted files.
    """
    used = {Path(font_manager.findfont(text.get_fontproperties())).name
            for text in fig.findobj(mtext.Text) if text.get_text().strip()}
    fallbacks = sorted(name for name in used if not name.lower().startswith(FONT.lower()))
    if fallbacks:
        warnings.warn(f"{FONT} is not used for all text (found {', '.join(fallbacks)}); "
                      f"install {FONT} to reproduce the submitted figures exactly.", stacklevel=2)


def save_pdf(fig, path, width_in, height_in):
    """Save `fig` as a vector PDF whose page is exactly the printed size of the TIFF.

    The figure is laid out larger than print (layout scale), so the PDF is
    rescaled onto a page of the final size; fonts stay embedded as TrueType.
    """
    import pymupdf

    buffer = io.BytesIO()
    with matplotlib.rc_context({"pdf.fonttype": 42}):
        fig.savefig(buffer, format="pdf", bbox_inches="tight", pad_inches=PAD_IN, facecolor="white")
    source = pymupdf.open("pdf", buffer.getvalue())
    out = pymupdf.open()
    page = out.new_page(width=width_in * 72, height=height_in * 72)
    page.show_pdf_page(page.rect, source, 0)
    out.save(path, garbage=4, deflate=True)


def export(fig, name, width="double", relayout=None, out_dir=FRONTIERS_DIR):
    """Save `fig` as <name>.tif (LZW), .png and vector .pdf at the Frontiers width and 300 dpi."""
    width_mm = WIDTHS_MM[width]
    width_in = width_mm / MM_PER_IN
    scale = fit_to_width(fig, width_in, relayout=relayout)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI * scale, bbox_inches="tight",
                pad_inches=PAD_IN, facecolor="white")
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    width_px = round(width_in * DPI)
    height_px = round(image.height * width_px / image.width)
    image = image.resize((width_px, height_px), Image.LANCZOS)
    height_mm = height_px / DPI * MM_PER_IN
    if height_mm > MAX_HEIGHT_MM:
        raise ValueError(f"{name}: {height_mm:.0f} mm tall exceeds one page ({MAX_HEIGHT_MM} mm)")
    out_dir.mkdir(parents=True, exist_ok=True)
    tif = out_dir / f"{name}.tif"
    image.save(tif, dpi=(DPI, DPI), compression="tiff_lzw")
    image.save(out_dir / f"{name}.png", dpi=(DPI, DPI))
    save_pdf(fig, out_dir / f"{name}.pdf", width_in, height_mm / MM_PER_IN)
    print(f"Saved: {tif} ({width_mm:.0f} x {height_mm:.0f} mm, {width_px} x {height_px} px, "
          f"{DPI} dpi, layout scale {scale:.2f})")
    return tif
