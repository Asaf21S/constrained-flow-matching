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

__all__ = ["SERIF_RC"]
