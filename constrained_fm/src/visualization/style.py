# -*- coding: utf-8 -*-
"""Shared Matplotlib typography for the paper figures.

One dict, so every figure module renders the same glyphs. Import it and either apply it
globally (``plt.rcParams.update``) or scope it to one figure (``plt.rc_context``).
"""

from __future__ import annotations

from typing import Any

# Computer Modern, matching the LaTeX body text of the paper.
SERIF_RC: dict[str, Any] = {
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "cmr10", "DejaVu Serif"],
    # cmr10 has no U+2212 MINUS SIGN, so Matplotlib's default unicode minus renders as a
    # tofu box on every negative tick and colorbar label. ASCII hyphen instead.
    "axes.unicode_minus": False,
    "mathtext.fontset": "cm",
    "axes.grid": False,
}

# Figures are authored near 7.6x5.2in but land in a two-column paper about 3.3in wide, so
# every glyph is reduced to roughly 45% before a reader sees it. These sizes are chosen to
# stay legible after that reduction rather than to look balanced at authoring size.
PAPER_RC: dict[str, Any] = {
    **SERIF_RC,
    "font.size": 16,
    "axes.labelsize": 19,
    "axes.titlesize": 19,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "legend.title_fontsize": 15,
    "figure.labelsize": 19,
    "lines.linewidth": 2.2,
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
}

__all__ = ["SERIF_RC", "PAPER_RC"]
