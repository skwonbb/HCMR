from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, MUTED = "#222222", "#555555", "#9a988f"


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 11.5, "axes.titlesize": 13.5, "axes.labelsize": 12,
        "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 12,
        "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": INK, "axes.linewidth": 0.8, "axes.spines.top": True, "axes.spines.right": True,
        "axes.grid": True, "grid.color": "#b0b0b0", "grid.alpha": 0.3, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "legend.frameon": False, "lines.linewidth": 1.8,
    })


def grid(nrows, ncols, height_per_row=2.3, width=17, **kw):
    return plt.subplots(nrows, ncols, figsize=(width, height_per_row * nrows), squeeze=False, **kw)


def label_grid(axes, col_titles=None, row_names=None, ylabel="", xlabel=None):
    nr, nc = axes.shape
    for j in range(nc):
        if col_titles:
            axes[0, j].set_title(col_titles[j], fontweight="bold")
        if xlabel:
            axes[-1, j].set_xlabel(xlabel)
    for i in range(nr):
        bold = r"$\bf{" + row_names[i].replace(" ", r"\ ") + "}$" if row_names else ""
        axes[i, 0].set_ylabel(f"{bold}\n{ylabel}" if row_names else ylabel)


def dots(ax, x, vals, width, color):
    import numpy as np
    from matplotlib.colors import to_rgb
    rgb = np.array(to_rgb(color))
    lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    tone = rgb + (1 - rgb) * 0.55 if lum < 0.35 else rgb * 0.55
    vals = np.asarray(vals, dtype=float)
    half = 0.2 if width >= 0.6 else 0.1
    xs = x + np.linspace(-half, half, len(vals)) * width if len(vals) > 1 else np.full(len(vals), x)
    ax.scatter(xs, vals, s=6, color=tone, alpha=0.9, lw=0, zorder=4)


def finish(fig, path, handles=None, labels=None, ncol=6, top=0.92):
    path = Path(path)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, top))
    else:
        fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path.with_suffix(".png"))
