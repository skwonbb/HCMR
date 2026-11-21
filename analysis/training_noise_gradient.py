import json, os, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as S

S.apply()
ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
G = Path(__file__).resolve().parent.parent / "results" / "noise" / "grad"
CONDS = [("shots", "shot noise", "#9a988f"), ("device", "device noise", "#e8a33d"), ("both", "both", "#b2182b")]
JOINT = "#4b2c86"
R = {(m, ds, s): json.loads((G / f"amplitude_{ds}_8q4c" / m / f"seed_{s}.json").read_text())
     for m in ("hcmr", "hcmr_cnn") for ds in ("mnist", "fmnist") for s in range(5)}


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


def curve(m, path, cond, part):
    per_seed = []
    for s in range(5):
        v = []
        for ds in ("mnist", "fmnist"):
            pts = [p for p in R[(m, ds, s)][path] if f"cos_{part}" in p[cond]]
            v.append([np.mean(p[cond][f"cos_{part}"]) for p in pts]); ep = [p["epoch"] for p in pts]
        per_seed.append(np.mean(v, 0))
    per_seed = np.array(per_seed)
    return np.array(ep), per_seed.mean(0), per_seed.std(0)


COLS = [("hcmr", "J10", 10), ("hcmr", "J60", 60), ("hcmr_cnn", "J10", 10), ("hcmr_cnn", "J60", 60)]
fig, axes = S.grid(2, 4, height_per_row=2.4, width=13)
for r, part in enumerate(("pqc", "enc")):
    for c, (m, path, j) in enumerate(COLS):
        ax = axes[r, c]
        ax.axvspan(60 - j + 0.5, 60.5, color=JOINT, alpha=0.07, lw=0)
        for cond, _, col in CONDS:
            ep, mu, sd = curve(m, path, cond, part)
            ax.plot(ep, mu, color=col, lw=1.2); ax.fill_between(ep, mu - sd, mu + sd, color=col, alpha=0.2, lw=0)
        ax.axhline(0, color=S.INK2, lw=0.5)
        ax.set_xlim(0.5, 60.5); ax.set_ylim(-0.15, 1.05); ax.set_xticks([1, 10, 20, 30, 40, 50, 60])
S.label_grid(axes, ["J = 10", "J = 60", "J = 10", "J = 60"], ["PQC gradient", "encoder gradient"], "cosine similarity",
             xlabel="epoch")
h = [Line2D([], [], color=col, lw=1.3) for _, _, col in CONDS] + [Patch(facecolor=JOINT, alpha=0.07, edgecolor=S.INK2, lw=0.5)]
l = [f"{n}" for _, n, _ in CONDS] + ["shaded: joint training"]
captions(fig, axes, [("(a) HCMR (linear)", list(axes[1, :2])), ("(b) HCMR (CNN)", list(axes[1, 2:]))],
         FIG / "training_noise_gradient", h, l, ncol=4)
