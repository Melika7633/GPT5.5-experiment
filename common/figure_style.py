"""Print-size typography shared by Fig 4 and Fig S1.

Sizes are points at the final printed width, so figures using this module are laid out at
PRINT_WIDTH_IN (180 mm, double column).
"""

import matplotlib as mpl


PRINT_WIDTH_IN = 7.09  # 180 mm
FONT_STACK = ["Arial", "Helvetica", "DejaVu Sans"]
TITLE_PT = 10
PANEL_LETTER_PT = 10
AXIS_LABEL_PT = 8
TICK_PT = 7
CATEGORY_PT = 7
LEGEND_PT = 7
AXIS_LINE_WIDTH = 0.8
AXIS_LINE_COLOR = "#4B5563"


def panel_letter(ax, letter, x=0.0, y=1.015):
    """Place a panel letter just above the top-left corner of an axes."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=PANEL_LETTER_PT,
            fontweight="bold", color="black", ha="left", va="bottom")


def emphasize_x_axis(ax):
    """Draw the x axis line heavier, to match the bold axis label."""
    ax.spines["bottom"].set_linewidth(AXIS_LINE_WIDTH)
    ax.spines["bottom"].set_color(AXIS_LINE_COLOR)
    ax.spines["bottom"].set_visible(True)


def apply_print_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "font.size": TICK_PT,
        "axes.titlesize": TITLE_PT,
        "axes.titleweight": "bold",
        "axes.labelsize": AXIS_LABEL_PT,
        "axes.labelweight": "bold",
        "xtick.labelsize": TICK_PT,
        "ytick.labelsize": TICK_PT,
        "legend.fontsize": LEGEND_PT,
        "figure.titlesize": TITLE_PT,
        "figure.titleweight": "bold",
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.6,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    })
