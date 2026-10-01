# -*- coding: utf-8 -*-
"""Figures for the decay6d importance-sampling study; every function consumes saved arrays only."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogFormatterSciNotation, MaxNLocator, NullLocator

from constrained_fm.src.visualization.style import PAPER_RC

ESTIMATOR_STYLE = {
    "q_raw": {"label": r"CFM $q$ (raw)", "color": "tab:orange", "ls": ":", "marker": "v"},
    "q_filtered": {"label": r"CFM $q$ (in-box)", "color": "tab:red", "ls": ":", "marker": "^"},
    "is_learned": {"label": r"IS, learned $p$", "color": "tab:blue", "ls": "-", "marker": "o"},
    "is_exact": {"label": r"IS, exact $p$", "color": "tab:cyan", "ls": "--", "marker": "s"},
    "rej_equal_n": {"label": "Rejection, equal $N$", "color": "tab:gray", "ls": "-",
                    "marker": "x"},
    "rej_equal_time": {"label": "Rejection, equal time", "color": "tab:green", "ls": "-.",
                       "marker": "d"},
    "rej_equal_nfe": {"label": "Rejection, equal NFE", "color": "tab:olive", "ls": "-.",
                      "marker": "P"},
}
WEIGHT_STYLE = {"learned": ESTIMATOR_STYLE["is_learned"], "exact": ESTIMATOR_STYLE["is_exact"]}
GT_COLOR = "black"
PANEL_SIZE = (4.2, 3.6)
HIST_BINS = 80
MAP_BINS = 120
MAP_PERCENTILES = (0.5, 99.5)
WEIGHT_BINS = 80
MAX_TICKS = 5
DATASET_BINS = 200
DATASET_PERCENTILE = 99.9
# Readable both on the dark end of viridis and on white.
BOX_COLORS = ("tab:red", "tab:orange", "magenta", "tab:cyan", "limegreen", "tab:brown")
AXIS_NAMES = "xyz"


def _normalized_weights(log_w: np.ndarray) -> np.ndarray:
    finite = np.isfinite(log_w)
    w = np.zeros_like(log_w, dtype=np.float64)
    w[finite] = np.exp(log_w[finite] - log_w[finite].max())
    return w / w.sum()


def plot_error_vs_n(n_values: np.ndarray, estimates: np.ndarray, gt_mean: np.ndarray,
                    gt_se: np.ndarray, box_names: list[str], estimators: list[str],
                    f_index: int, f_label: str) -> Figure:
    """Top: RMSE vs ``N`` with a ``1/sqrt(N)`` guide. Bottom: mean +- std of the estimate vs GT.

    ``estimates`` is ``(boxes, estimators, f, N, reps)``.
    """
    num_b = len(box_names)
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(2, num_b, figsize=(PANEL_SIZE[0] * num_b, 2 * PANEL_SIZE[1]),
                                 sharex=True, sharey="row", squeeze=False)
        for b, name in enumerate(box_names):
            gt = gt_mean[b, f_index]
            for i, est in enumerate(estimators):
                vals = estimates[b, i, f_index]
                style = ESTIMATOR_STYLE[est]
                kw = {"color": style["color"], "ls": style["ls"], "marker": style["marker"]}
                rmse = np.sqrt(np.nanmean((vals - gt) ** 2, axis=-1))
                axes[0, b].plot(n_values, rmse, label=style["label"], **kw)
                axes[1, b].errorbar(n_values, np.nanmean(vals, -1), yerr=np.nanstd(vals, -1),
                                    capsize=3, **kw)
            ref = np.sqrt(np.nanmean((estimates[b, estimators.index("is_exact"), f_index, 0]
                                      - gt) ** 2))
            axes[0, b].plot(n_values, ref * np.sqrt(n_values[0] / n_values), color=GT_COLOR,
                            lw=1.0, ls="--", label=r"$\propto N^{-1/2}$")
            axes[1, b].axhspan(gt - 2 * gt_se[b, f_index], gt + 2 * gt_se[b, f_index],
                               color=GT_COLOR, alpha=0.15, lw=0)
            axes[1, b].axhline(gt, color=GT_COLOR, lw=1.0, label="Ground truth")
            axes[0, b].set(title=name.replace("_", " "), xscale="log", yscale="log")
            axes[1, b].set(xscale="log", xlabel="$N$")
        axes[0, 0].set_ylabel(f"RMSE of {f_label}")
        axes[1, 0].set_ylabel(f"Estimate of {f_label}")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.13))
        fig.tight_layout()
    return fig


def plot_weight_histograms(log_w: dict[str, list[np.ndarray]], box_names: list[str]) -> Figure:
    """``log10(N w_norm)`` of in-box samples: 0 is the mean weight, the right tail drives ESS."""
    num_b = len(box_names)
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, num_b, figsize=(PANEL_SIZE[0] * num_b, PANEL_SIZE[1]),
                                 sharey=True, squeeze=False)
        for b, name in enumerate(box_names):
            ax = axes[0, b]
            for kind, arrays in log_w.items():
                w = _normalized_weights(arrays[b])
                scaled = np.log10(w[w > 0] * np.count_nonzero(w))
                ax.hist(scaled, bins=WEIGHT_BINS, density=True, histtype="step", lw=1.8,
                        color=WEIGHT_STYLE[kind]["color"], label=WEIGHT_STYLE[kind]["label"])
            ax.set(title=name.replace("_", " "), yscale="log",
                   xlabel=r"$\log_{10}(N_{\mathcal{B}}\,\bar w)$")
            ax.xaxis.set_major_locator(MaxNLocator(MAX_TICKS))
        axes[0, 0].set_ylabel("Density")
        axes[0, 0].legend(frameon=False)
        fig.tight_layout()
    return fig


def plot_ess(n_values: np.ndarray, ess_frac: np.ndarray, max_weight: np.ndarray,
             box_names: list[str], weights: list[str]) -> Figure:
    """ESS / N and the max normalized weight vs ``N``; arrays are ``(boxes, weights, N, reps)``."""
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 2, figsize=(2 * PANEL_SIZE[0] * 1.3, PANEL_SIZE[1] * 1.2))
        for b, name in enumerate(box_names):
            for i, kind in enumerate(weights):
                style = WEIGHT_STYLE[kind]
                kw = {"color": style["color"], "ls": style["ls"], "marker": markers[b]}
                label = f"{name.replace('_', ' ')}, {kind}"
                axes[0].plot(n_values, np.nanmean(ess_frac[b, i], -1), label=label, **kw)
                axes[1].plot(n_values, np.nanmean(max_weight[b, i], -1), **kw)
        axes[1].plot(n_values, 1.0 / n_values, color=GT_COLOR, lw=1.0, ls="--", label="$1/N$")
        axes[0].set(xscale="log", xlabel="$N$", ylabel="ESS / $N$", ylim=(0.0, 1.05))
        axes[1].set(xscale="log", yscale="log", xlabel="$N$", ylabel=r"max $\bar w$")
        axes[1].legend(frameon=False)
        axes[0].legend(frameon=False, fontsize="x-small", ncol=2)
        fig.tight_layout()
    return fig


def _hist_1d(ax, values: list[np.ndarray], weights: list[np.ndarray | None], styles: list[dict],
             bins: np.ndarray) -> None:
    for v, w, style in zip(values, weights, styles):
        ax.hist(v, bins=bins, weights=w, density=True, histtype="step", lw=1.8, **style)


def plot_p2_marginals(gt_p2: np.ndarray, q_p2: np.ndarray, q_inside: np.ndarray,
                      log_w: dict[str, np.ndarray], box_name: str) -> Figure:
    """1D marginals of ``|p2|``, ``p2_x``, ``p2_z`` and ``(p2_x, p2_z)`` maps for one box.

    ``raw q`` uses every proposal sample, violators included, so leakage stays visible; the IS
    series reweight the same samples with learned and exact ``p``, zero weight outside the box.
    """
    series = [("Ground truth", gt_p2, None, {"color": GT_COLOR}),
              (r"CFM $q$ (raw)", q_p2, None, {"color": ESTIMATOR_STYLE["q_raw"]["color"]})]
    for kind, lw in log_w.items():
        w = _normalized_weights(lw)
        series.append((WEIGHT_STYLE[kind]["label"], q_p2[q_inside], w[q_inside],
                       {"color": WEIGHT_STYLE[kind]["color"]}))
    marginals = [(r"$|\vec p_2|$", lambda p: np.linalg.norm(p, axis=1)),
                 (r"$p_{2x}$", lambda p: p[:, 0]), (r"$p_{2z}$", lambda p: p[:, 2])]

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(2, len(series), figsize=(len(series) * PANEL_SIZE[0],
                                                          2 * PANEL_SIZE[1]),
                                 layout="constrained")
        for ax, (label, fn) in zip(axes[0], marginals):
            lo, hi = np.percentile(fn(gt_p2), MAP_PERCENTILES)
            pad = 0.1 * (hi - lo)
            bins = np.linspace(lo - pad, hi + pad, HIST_BINS + 1)
            _hist_1d(ax, [fn(p) for _, p, _, _ in series], [wt for _, _, wt, _ in series],
                     [{**s, "label": n} for n, _, _, s in series], bins)
            ax.set(xlabel=label)
            ax.xaxis.set_major_locator(MaxNLocator(MAX_TICKS))
        axes[0, 0].set_ylabel("Density")
        for ax in axes[0, len(marginals):]:
            ax.axis("off")
        axes[0, -1].legend(*axes[0, 0].get_legend_handles_labels(), frameon=False, loc="center")

        (x_lo, z_lo), (x_hi, z_hi) = (np.percentile(gt_p2[:, [0, 2]], q, axis=0)
                                      for q in MAP_PERCENTILES)
        extent = [x_lo, x_hi, z_lo, z_hi]
        maps = [np.histogram2d(p[:, 0], p[:, 2], bins=MAP_BINS, range=[extent[:2], extent[2:]],
                               weights=wt, density=True)[0]
                for _, p, wt, _ in series]
        # Anchored on the ground truth so a few heavy IS weights cannot wash out the scale.
        vmax = maps[0].max()
        for ax, (name, _, _, _), density in zip(axes[1], series, maps):
            image = ax.imshow(density.T, origin="lower", extent=extent, aspect="auto",
                              interpolation="nearest", cmap="viridis", vmin=0.0, vmax=vmax)
            ax.set(title=name, xlabel=r"$p_{2x}$")
        axes[1, 0].set_ylabel(r"$p_{2z}$")
        for ax in axes[1, 1:]:
            ax.sharey(axes[1, 0])
            ax.tick_params(labelleft=False)
        fig.colorbar(image, ax=axes[1].tolist(), shrink=0.9, label="Density", extend="max")
        fig.suptitle(box_name.replace("_", " "))
    return fig


def _log_density_image(ax, a: np.ndarray, b: np.ndarray, extent: list[float], aspect: str):
    density = np.histogram2d(a, b, bins=DATASET_BINS, range=[extent[:2], extent[2:]],
                             density=True)[0]
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(cmap(0))
    return ax.imshow(np.ma.masked_equal(density.T, 0.0), origin="lower", extent=extent,
                     aspect=aspect, interpolation="nearest", cmap=cmap, norm=LogNorm())


def _box_label(box: dict) -> str:
    return f"{box['name'].replace('_', ' ')} ({100 * box['gt_mass']:.1f}%)"


def plot_dataset_p1_boxes(x: np.ndarray, boxes: list[dict]) -> Figure:
    """``p1`` projections with the evaluation boxes; a projected box also spans events that
    miss it along the hidden axis."""
    p1 = x[:, :3]
    lim = float(np.percentile(np.abs(p1), DATASET_PERCENTILE))
    extent = [-lim, lim, -lim, lim]
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 3, figsize=(3 * PANEL_SIZE[0], 1.1 * PANEL_SIZE[1]),
                                 layout="constrained")
        for ax, (i, j) in zip(axes, [(0, 1), (0, 2), (1, 2)]):
            image = _log_density_image(ax, p1[:, i], p1[:, j], extent, "equal")
            for color, box in zip(BOX_COLORS, boxes):
                ax.add_patch(Rectangle((box["lo"][i], box["lo"][j]),
                                       box["hi"][i] - box["lo"][i], box["hi"][j] - box["lo"][j],
                                       fill=False, ec=color, lw=1.6, label=_box_label(box)))
            ax.set(xlabel=f"$p_{{1{AXIS_NAMES[i]}}}$", ylabel=f"$p_{{1{AXIS_NAMES[j]}}}$")
        fig.colorbar(image, ax=axes.tolist(), shrink=0.9, label=r"Density of $\vec p_1$")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=len(boxes),
                   bbox_to_anchor=(0.5, 1.0), frameon=False)
    return fig


def plot_dataset_structure(x: np.ndarray) -> Figure:
    """How ``p2`` is tied to ``p1``: momentum split, back-to-back z, and opening angle."""
    p1, p2 = x[:, :3], x[:, 3:]
    n1, n2 = np.linalg.norm(p1, axis=1), np.linalg.norm(p2, axis=1)
    cos = (p1 * p2).sum(1) / (n1 * n2)
    hi = float(np.percentile(np.concatenate([n1, n2]), DATASET_PERCENTILE))
    lim = float(np.percentile(np.abs(np.concatenate([p1[:, 2], p2[:, 2]])), DATASET_PERCENTILE))
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(1, 3, figsize=(3 * PANEL_SIZE[0], 1.1 * PANEL_SIZE[1]),
                                 layout="constrained")
        for ax, (a, b, extent, xl, yl, title) in zip(axes, [
                (n1, n2, [0.0, hi, 0.0, hi], r"$|\vec p_1|$", r"$|\vec p_2|$", "Momentum split"),
                (p1[:, 2], p2[:, 2], [-lim, lim, -lim, lim], r"$p_{1z}$", r"$p_{2z}$",
                 "Back-to-back")]):
            fig.colorbar(_log_density_image(ax, a, b, extent, "equal"), ax=ax, shrink=0.8)
            ax.set(xlabel=xl, ylabel=yl, title=title)
        axes[2].hist(cos, bins=HIST_BINS, range=(-1.0, 1.0), density=True, color="0.5")
        axes[2].set(yscale="log", xlabel=r"$\cos\angle(\vec p_1, \vec p_2)$", ylabel="Density",
                    title="Opening angle")
    return fig


def plot_dataset_p2_given_box(x: np.ndarray, boxes: list[dict]) -> Figure:
    """What each box does to ``p2``: ``|p2|`` against all events, and the in-box ``(p2x, p2z)``."""
    p1, p2 = x[:, :3], x[:, 3:]
    n2 = np.linalg.norm(p2, axis=1)
    bins = np.linspace(0.0, float(np.percentile(n2, DATASET_PERCENTILE)), HIST_BINS + 1)
    lim = float(np.percentile(np.abs(p2[:, [0, 2]]), DATASET_PERCENTILE))
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(2, len(boxes), figsize=(len(boxes) * PANEL_SIZE[0],
                                                         2 * PANEL_SIZE[1]),
                                 layout="constrained", sharey="row", squeeze=False)
        for b, (color, box) in enumerate(zip(BOX_COLORS, boxes)):
            inside = ((p1 >= box["lo"]) & (p1 <= box["hi"])).all(1)
            ax = axes[0, b]
            ax.hist(n2, bins=bins, density=True, color="0.85", label="All events")
            ax.hist(n2[inside], bins=bins, density=True, histtype="step", lw=1.8, color=color,
                    label=r"$\vec p_1 \in \mathcal{B}$")
            ax.axvline(box["tail_threshold"], color=color, ls="--", lw=1.0,
                       label=r"$\tau_{\mathcal{B}}$")
            ax.set(title=f"{_box_label(box)}\n{int(inside.sum())} events",
                   xlabel=r"$|\vec p_2|$")
            ax.xaxis.set_major_locator(MaxNLocator(MAX_TICKS))
            _log_density_image(axes[1, b], p2[inside, 0], p2[inside, 2], [-lim, lim, -lim, lim],
                               "equal")
            axes[1, b].set(xlabel=r"$p_{2x}$")
        axes[0, 0].set_ylabel("Density")
        axes[0, 0].legend(frameon=False, fontsize="small")
        axes[1, 0].set_ylabel(r"$p_{2z}$ (in-box events)")
    return fig


def plot_rmse_vs_mass(masses: np.ndarray, rmse: dict[str, np.ndarray], box_names: list[str],
                      y_label: str, n: int) -> Figure:
    """RMSE at one budget ``n`` against ``P(B)``, rarest box on the right of a reversed log axis.

    ``rmse[estimator]`` is ``(boxes,)``, aligned with ``masses`` and ``box_names``.
    """
    order = np.argsort(masses)[::-1]
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(7.5, 5.2), layout="constrained")
        for estimator, values in rmse.items():
            style = ESTIMATOR_STYLE[estimator]
            ax.plot(masses[order], values[order], color=style["color"], ls=style["ls"],
                    marker=style["marker"], ms=9, label=style["label"])
        ax.set(xscale="log", yscale="log", xlabel=r"True constraint mass $P(\mathcal{B})$",
               ylabel=y_label, title=f"$N={n:,}$ samples per estimate")
        ax.set_xticks(masses[order], [f"{box_names[i]}\n$P={m:.3g}$" for i, m in
                                      zip(order, masses[order])], fontsize="x-small",
                      rotation=25, ha="right", rotation_mode="anchor")
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_formatter(LogFormatterSciNotation(labelOnlyBase=False,
                                                             minor_thresholds=(2, 0.5)))
        ax.tick_params(axis="y", which="minor", labelsize="x-small")
        ax.invert_xaxis()
        ax.grid(True, which="major", alpha=0.3)
        ax.legend(frameon=False, fontsize="small")
    return fig


def plot_rmse_vs_mass_grid(masses: np.ndarray, rmse: dict[str, dict[int, dict[str, np.ndarray]]],
                           row_labels: dict[str, str], title: str) -> Figure:
    """RMSE against ``P(B)``: one row per observable, one column per budget ``N``.

    ``rmse[observable][n][estimator]`` is ``(boxes,)``, aligned with ``masses``; the mass axis is
    reversed so rarer boxes sit to the right.
    """
    order = np.argsort(masses)[::-1]
    n_values = list(next(iter(rmse.values())))
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(len(rmse), len(n_values), sharex=True, sharey="row",
                                 squeeze=False, layout="constrained",
                                 figsize=(4.4 * len(n_values), 3.4 * len(rmse)))
        for row, (observable, by_n) in zip(axes, rmse.items()):
            for ax, n in zip(row, n_values):
                for estimator, values in by_n[n].items():
                    style = ESTIMATOR_STYLE[estimator]
                    ax.plot(masses[order], values[order], color=style["color"], ls=style["ls"],
                            marker=style["marker"], ms=6, label=style["label"])
                ax.set(xscale="log", yscale="log")
                ax.grid(True, which="major", alpha=0.3)
            row[0].set_ylabel(row_labels[observable])
        for ax, n in zip(axes[0], n_values):
            ax.set_title(f"$N={n:,}$")
        for ax in axes[-1]:
            ax.set_xticks(masses[order], [f"{100 * m:.{1 if m < 0.01 else 0}f}%"
                                          for m in masses[order]], fontsize="small")
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlabel(r"Constraint mass $P(\mathcal{B})$")
        axes[0, 0].invert_xaxis()
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center",
                   ncol=len(rmse[next(iter(rmse))][n_values[0]]), frameon=False)
        fig.suptitle(title)
    return fig


def _count_label(count: float) -> str:
    if count >= 1e6:
        return f"{count / 1e6:.3g}M"
    return f"{count / 1e3:.3g}k"


def plot_rmse_vs_time(seconds: dict[str, np.ndarray], rmse: dict[str, np.ndarray],
                      draws: dict[str, np.ndarray], y_label: str, box_name: str) -> Figure:
    """Error-vs-cost frontier: one connected curve per estimator, one point per IS budget ``N``.

    Points are labelled with the estimator's own model draws per estimate; marker size grows with ``N``.
    """
    sizes = np.linspace(6, 14, len(next(iter(draws.values()))))
    with plt.rc_context(PAPER_RC):
        fig, ax = plt.subplots(figsize=(7.5, 5.2), layout="constrained")
        for estimator, time in seconds.items():
            style = ESTIMATOR_STYLE[estimator]
            ax.plot(time, rmse[estimator], color=style["color"], ls=style["ls"], lw=1.8,
                    label=style["label"])
            for t, err, size, count in zip(time, rmse[estimator], sizes, draws[estimator]):
                ax.plot(t, err, color=style["color"], marker=style["marker"], ms=size)
                ax.annotate(_count_label(count), (t, err), textcoords="offset points",
                            xytext=(7, 5), fontsize="x-small", color=style["color"])
        ax.set(xscale="log", yscale="log", xlabel="Mean time per estimate [s]",
               ylabel=y_label, title=box_name)
        ax.grid(True, which="major", alpha=0.3)
        ax.legend(frameon=False, fontsize="small", title="labels: draws per estimate",
                  title_fontsize="x-small")
    return fig


__all__ = ["ESTIMATOR_STYLE", "plot_error_vs_n", "plot_weight_histograms", "plot_ess",
           "plot_p2_marginals", "plot_dataset_p1_boxes", "plot_dataset_structure",
           "plot_dataset_p2_given_box", "plot_rmse_vs_mass", "plot_rmse_vs_mass_grid",
           "plot_rmse_vs_time"]
