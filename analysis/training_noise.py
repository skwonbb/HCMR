import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as S

S.apply()
ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
D = Path(__file__).resolve().parent.parent / "results" / "noise" / "train"
GROUPS = [("hcmr", "10"), ("hcmr", "60"), ("hcmr_cnn", "10"), ("hcmr_cnn", "60")]
MNAME = {"hcmr": "HCMR (linear)", "hcmr_cnn": "HCMR (CNN)"}
CONDS = [("shots", "shot noise", "#c9c7c0"), ("device", "device noise", "#e8a33d"), ("both", "both", "#b2182b")]
plt.rcParams["hatch.linewidth"] = 0.7
LOSS = dict(facecolor="#f4c7c3", edgecolor="#b2182b", hatch="//////", lw=0)
GAIN = dict(facecolor="#d6e3f3", edgecolor="#2c5d9e", hatch="\\\\\\\\\\\\", lw=0)


def acc(cond, m, j, reg):
    v = [np.mean([100 * json.loads((D / cond / f"amplitude_{ds}_8q4c" / m / f"seed_{s}.json").read_text())
                  ["J"][j]["ideal"][reg] for ds in ("mnist", "fmnist")]) for s in range(5)]
    return np.mean(v), np.std(v), v


def change_bar(ax, xi, ref, val, width, color):
    ax.bar(xi, val, width, color=color)
    if val < ref:
        ax.bar(xi, ref - val, width, bottom=val, **LOSS)


def captions(fig, axes, groups, path, h, l, ncol):
    import matplotlib.pyplot as plt
    fig.legend(h, l, loc="upper center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    r = fig.canvas.get_renderer()
    y = min(a.get_tightbbox(r).transformed(fig.transFigure.inverted()).y0 for a in axes.flat) - 0.015
    for lab, grp in groups:
        x = (grp[0].get_position().x0 + grp[-1].get_position().x1) / 2
        fig.text(x, y, lab, ha="center", va="top", fontsize=14)
    for ext in ("png", "pdf"):
        fig.savefig(Path(path).with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig); print("saved", Path(path).with_suffix(".png"))


CE_C, PE_C = "#e58ca8", "#3a6fb0"
ENC_C = {"hcmr": "#4b2c86", "hcmr_cnn": "#1b7837"}
x = np.arange(len(CONDS))
ROWS = [("acc", [(m, "acc", ENC_C[m]) for m in ("hcmr", "hcmr_cnn")], "accuracy"),
        ("hcmr", [("hcmr", "acc_ce", CE_C), ("hcmr", "acc_pe", PE_C)], "HCMR (linear)"),
        ("hcmr_cnn", [("hcmr_cnn", "acc_ce", CE_C), ("hcmr_cnn", "acc_pe", PE_C)], "HCMR (CNN)")]
COLS = [(j, k) for j in ("10", "60") for k in range(3)]
TITLES = ["overall", "HCMR (linear)", "HCMR (CNN)"]
fig, axes = S.grid(1, 6, height_per_row=2.8, width=14)
for c, (j, r) in enumerate(COLS):
    ax = axes[0, c]; bars = ROWS[r][1]
    for i, (cond, *_) in enumerate(CONDS):
        for k, (m, reg, col) in enumerate(bars):
            ref = acc("none", m, j, reg)[0]; mu, sd, per_seed = acc(cond, m, j, reg)
            xi = i + (k - 0.5) * 0.28
            change_bar(ax, xi, ref, mu, 0.26, col)
            ax.errorbar(xi, mu, yerr=sd, fmt="none", lw=0.7, capsize=1.5, ecolor=S.INK2)
            S.dots(ax, xi, per_seed, 0.26, col)
    ax.set_xticks(x); ax.set_xticklabels(["shot", "device", "both"]); ax.grid(axis="x", visible=False)
    ax.set_xlim(-0.55, 2.55); ax.set_ylim(0, 100); ax.set_title(TITLES[r], fontweight="bold")
axes[0, 0].set_ylabel("test accuracy [%]")
h = [Patch(color=ENC_C["hcmr"]), Patch(color=ENC_C["hcmr_cnn"]), Patch(color=CE_C), Patch(color=PE_C), Patch(**LOSS)]
l = ["HCMR (linear)", "HCMR (CNN)", "CE accuracy", "PE accuracy", "lost vs. noiseless training"]
captions(fig, axes, [("(a) J = 10", list(axes[0, :3])), ("(b) J = 60", list(axes[0, 3:]))],
         FIG / "training_noise", h, l, ncol=5)
