"""Post-processing for AI-generated matplotlib figures.

Call tidy_figure(fig) AFTER the generated code has run successfully, outside the
exec/timeout block, so a cosmetic problem can never turn a good chart into an
"execution error" that triggers the Debugger.

Legend strategy (chosen per axes by how many series there are and what they are):
  1-12 series            -> normal legend, outside the plot on the right
  13-80 series (text)    -> compact multi-column legend, small font, outside on the right
  8+ ordered numeric     -> legend replaced by a colorbar (a sequential ramp), because
                            matplotlib's default 10-colour cycle repeats, so a 74-entry
                            legend cannot actually tell the entries apart
  81+ series (text)      -> legend dropped, with a note suggesting binning or grouping
Legends built from explicit handles (custom Patch lists, scatter.legend_elements())
are read from the legend itself and rebuilt, never silently deleted.
"""
import math
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.container import BarContainer
from matplotlib.ticker import FuncFormatter

MAX_SIMPLE_LEGEND = 12
MAX_COMPACT_LEGEND = 80
MAX_LEGEND_COLUMNS = 6
MIN_COLORBAR_SERIES = 8
ROWS_PER_LEGEND_COLUMN = 14
MAX_FIG_INCHES = (12, 8)
SEQUENTIAL_CMAP = "viridis"


def _numeric_ordered(labels) -> Optional[List[float]]:
    """Return labels as floats if they are all numeric and strictly increasing."""
    try:
        vals = [float(l) for l in labels]
    except (TypeError, ValueError):
        return None
    if all(b > a for a, b in zip(vals, vals[1:])):
        return vals
    return None


_UNIT_WORDS = ("thousand", "million", "billion", "(k)", "(m)", " k)", " m)")


def _apply_unit_scaling(ax) -> Optional[str]:
    """Rescale the y axis to thousands/millions when values are large.

    Skipped when the generated code already formatted the ticks itself (a custom
    formatter is present), when the axis is log-scaled, when there is no y label to
    carry the unit, or when the label already names a unit.
    """
    label = ax.get_ylabel()
    if not label or ax.get_yscale() != "linear":
        return None
    if isinstance(ax.yaxis.get_major_formatter(), FuncFormatter):
        return None  # the Coder already formatted the ticks; don't double-scale
    lo, hi = ax.get_ylim()
    top = max(abs(lo), abs(hi))
    low = label.lower()

    # Case 1: the label (written from the user's request) already names a unit.
    # Make the ticks agree with it. If the data is already small, assume the code
    # divided it itself and leave everything alone.
    stated = None
    if "million" in low:
        stated, divisor = "millions", 1e6
    elif "thousand" in low:
        stated, divisor = "thousands", 1e3
    if stated:
        if top < divisor:
            return None
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v / divisor:g}"))
        return f"Y ticks rescaled to {stated} to match the axis title."

    # Case 2: no unit named; pick one from the data range and say so in the title.
    if top >= 1e6:
        divisor, word = 1e6, "millions"
    elif top >= 1e4:
        divisor, word = 1e3, "thousands"
    else:
        return None
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v / divisor:g}"))
    if label.rstrip().endswith(")"):
        label = label.rstrip()[:-1] + f", {word})"
    else:
        label = f"{label} ({word})"
    ax.set_ylabel(label)
    return f"Y axis shown in {word}."


def _recolor_sequential(ax, labels, cmap) -> bool:
    """Recolor the artists behind each legend label along a sequential ramp."""
    n = len(labels)
    colors = {str(lab): cmap(i / max(n - 1, 1)) for i, lab in enumerate(labels)}
    touched = 0
    for c in ax.containers:
        if isinstance(c, BarContainer) and str(c.get_label()) in colors:
            for patch in c.patches:
                patch.set_facecolor(colors[str(c.get_label())])
            touched += 1
    for line in ax.lines:
        if str(line.get_label()) in colors:
            line.set_color(colors[str(line.get_label())])
            touched += 1
    return touched > 0


def _outside_legend(ax, handles, labels, title, ncol, fontsize, compact=False):
    kwargs = dict(
        title=title or None,
        fontsize=fontsize,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),   # outside the plot area, never on top of data
        borderaxespad=0.0,
        ncol=ncol,
    )
    if compact:
        kwargs.update(handlelength=1.0, handletextpad=0.4,
                      labelspacing=0.25, columnspacing=0.8, title_fontsize=fontsize)
    ax.legend(handles, labels, **kwargs)


def tidy_figure(fig, n_rows: Optional[int] = None) -> List[str]:
    """Make a figure readable. Returns human-readable notes about what was changed.

    n_rows: total records in the dataset. When given, a small "n = 10,432 records"
    footer is added so a bare "Count" axis is anchored to the size of the data.
    """
    notes: List[str] = []

    # 1. Cap the canvas size so a runaway figure can't produce a giant image.
    w, h = fig.get_size_inches()
    max_w, max_h = MAX_FIG_INCHES
    if w > max_w or h > max_h:
        fig.set_size_inches(min(w, max_w), min(h, max_h))

    skip_axes = []  # axes we created ourselves (colour bars) must not be unit-scaled

    # 2. Rebuild legends. Legend.set_ncols() only stores a number; it does not
    #    re-pack an existing legend, so we recreate the legend instead.
    for ax in list(fig.get_axes()):
        legend = ax.get_legend()
        if legend is None:
            continue
        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            # The legend was built from explicit/proxy handles (custom Patch list,
            # scatter.legend_elements(), ...). Read the entries from the legend itself.
            handles = list(getattr(legend, "legend_handles", None)
                           or getattr(legend, "legendHandles", None) or [])
            labels = [t.get_text() for t in legend.get_texts()]
        title = legend.get_title().get_text()
        n = len(handles)
        if n == 0:
            continue  # nothing we can rebuild, so leave the original legend untouched

        # Build the replacement first; ax.legend() swaps out the old one. We only call
        # legend.remove() in branches that deliberately do not draw a legend.
        if n <= MAX_SIMPLE_LEGEND:
            _outside_legend(ax, handles, labels, title,
                            ncol=1 if n <= 8 else 2, fontsize="small")
            continue

        vals = _numeric_ordered(labels)
        if vals is not None and n >= MIN_COLORBAR_SERIES:
            cmap = plt.get_cmap(SEQUENTIAL_CMAP)
            if _recolor_sequential(ax, labels, cmap):
                legend.remove()
                sm = ScalarMappable(norm=Normalize(vmin=min(vals), vmax=max(vals)), cmap=cmap)
                cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, shrink=0.8)
                skip_axes.append(cbar.ax)
                if title:
                    cbar.set_label(title, fontsize="small")
                cbar.ax.tick_params(labelsize="x-small")
                notes.append(f"{n} ordered series: legend replaced by a colour scale.")
                continue

        if n <= MAX_COMPACT_LEGEND:
            ncol = min(MAX_LEGEND_COLUMNS, math.ceil(n / ROWS_PER_LEGEND_COLUMN))
            _outside_legend(ax, handles, labels, title, ncol=ncol,
                            fontsize="x-small" if n <= 40 else "xx-small", compact=True)
            notes.append(f"{n} series: legend shown in compact multi-column form.")
            continue

        legend.remove()
        notes.append(
            f"Legend hidden: {n} series is too many to read. "
            "Try grouping or binning that column (for example, age bands)."
        )

    # 2b. Scale large y values to thousands/millions and say so in the axis title,
    #     so the title and the tick numbers can never disagree.
    for ax in list(fig.get_axes()):
        if ax in skip_axes:
            continue
        note = _apply_unit_scaling(ax)
        if note:
            notes.append(note)

    # 3. Rotate x tick labels only when they would collide (text-like or crowded).
    fig.canvas.draw()  # tick label text is empty for numeric axes until a draw
    for ax in list(fig.get_axes()):
        texts = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
        if texts and (max(len(t) for t in texts) > 5 or len(texts) > 10):
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    try:
        if n_rows is not None:
            fig.tight_layout(rect=[0, 0.04, 1, 1])  # reserve a thin footer strip
            fig.text(0.99, 0.01, f"n = {n_rows:,} records", ha="right", va="bottom",
                     fontsize="x-small", color="0.4")
        else:
            fig.tight_layout()
    except Exception:
        pass  # layout is cosmetic; never fail the run over it

    return notes