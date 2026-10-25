"""Matplotlib figures for the README / report, sharing one quiet visual style.

Colours follow a validated categorical order (slot 1 = our method) so the same
family keeps the same colour in every figure; text never uses series colours.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

FAMILY_COLORS = {
    "Deep RL": SERIES[0],
    "Contextual bandit": SERIES[1],
    "Heuristic": SERIES[2],
    "Oracle": SERIES[3],
}

DIVERGING = LinearSegmentedColormap.from_list(
    "red_gray_blue", ["#b52f2f", "#e88584", "#f0efec", "#6da7ec", "#184f95"])

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 10,
    "text.color": INK,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "axes.titleweight": "semibold",
    "axes.titlesize": 12,
    "axes.titlelocation": "left",
    "legend.frameon": False,
})


def _style(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def policy_comparison(rows, path, title):
    """Horizontal bars: mean session return with 95% CI, coloured by policy family."""
    rows = sorted(rows, key=lambda r: r["mean"])
    fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(rows) + 1.4))
    y = np.arange(len(rows))
    colors = [FAMILY_COLORS[r["family"]] for r in rows]
    ax.barh(y, [r["mean"] for r in rows], height=0.55, color=colors, edgecolor=SURFACE,
            linewidth=2)
    ax.errorbar([r["mean"] for r in rows], y, xerr=[r["ci95"] for r in rows], fmt="none",
                ecolor=INK_2, elinewidth=1, capsize=2)
    span = max(abs(r["mean"]) + r["ci95"] for r in rows)
    for yi, r in zip(y, rows):
        offset = span * 0.03 + r["ci95"]
        x = r["mean"] + offset if r["mean"] >= 0 else r["mean"] - offset
        ax.text(x, yi, f"{r['mean']:+.2f}", va="center",
                ha="left" if r["mean"] >= 0 else "right", color=INK, fontsize=9)
    ax.set_yticks(y, [r["name"] for r in rows])
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_xlim(-span * 1.25 if min(r["mean"] for r in rows) < 0 else 0, span * 1.25)
    ax.set_xlabel("Mean return per 20-turn session (higher is better)")
    ax.set_title(title)
    _style(ax, grid_axis="x")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLORS.values()]
    present = [f for f in FAMILY_COLORS if any(r["family"] == f for r in rows)]
    ax.legend([handles[list(FAMILY_COLORS).index(f)] for f in present], present,
              loc="lower right", fontsize=9)
    _save(fig, path)


def learning_curves(curves, references, path, title):
    """curves: name -> (timesteps, mean, std). references: name -> value (thin rules)."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, (name, (steps, mean, std)) in enumerate(curves.items()):
        color = SERIES[i]
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.plot(steps, mean, color=color, linewidth=2, label=name, solid_capstyle="round")
    for name, value in references.items():  # labelled at the right end, clear of early curves
        ax.axhline(value, color=MUTED, linewidth=1)
        ax.text(ax.get_xlim()[1], value, f"{name} ({value:+.2f}) ", va="bottom", ha="right",
                color=INK_2, fontsize=8.5)
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean evaluation return")
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K" if v else "0"))
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    _style(ax)
    _save(fig, path)


def heatmap(matrix, row_labels, col_labels, path, title):
    """Diverging heatmap centred on 0 (red = worse than doing nothing, blue = better)."""
    matrix = np.asarray(matrix)
    limit = np.nanmax(np.abs(matrix))
    fig, ax = plt.subplots(figsize=(1.5 * len(col_labels) + 3.5, 0.45 * len(row_labels) + 1.6))
    im = ax.imshow(matrix, cmap=DIVERGING, vmin=-limit, vmax=limit, aspect="auto")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            dark = abs(v) / limit > 0.55
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=9,
                    color="#ffffff" if dark else INK)
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelcolor=INK_2)
    cbar.set_label("Mean return", color=INK_2)
    _save(fig, path)


def line_with_bands(x, series, path, title, xlabel, ylabel, xticks=None, xticklabels=None):
    """series: name -> (mean array, ci array). Markers on every point, labels at the end."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for i, (name, (mean, ci)) in enumerate(series.items()):
        color = SERIES[i]
        mean, ci = np.asarray(mean), np.asarray(ci)
        ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)
    if xticks is not None:
        ax.set_xticks(xticks, xticklabels if xticklabels is not None else xticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    _style(ax)
    _save(fig, path)


def confusion(matrix, labels, path, title):
    """Row-normalised confusion matrix on a single-hue sequential scale."""
    matrix = np.asarray(matrix, dtype=float)
    norm = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1e-9)
    cmap = LinearSegmentedColormap.from_list(
        "blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95"])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{norm[i, j] * 100:.0f}%", ha="center", va="center", fontsize=8,
                    color="#ffffff" if norm[i, j] > 0.55 else INK)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelcolor=INK_2)
    _save(fig, path)
