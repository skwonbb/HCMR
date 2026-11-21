import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as S

S.apply()
ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
GROUPS = [("hcmr", "amplitude", "linear, amplitude", "#4b2c86"), ("hcmr", "angle", "linear, angle", "#a996cc"),
          ("hcmr_cnn", "amplitude", "CNN, amplitude", "#1b7837"), ("hcmr_cnn", "angle", "CNN, angle", "#7fbf7b")]
WOP = {"hcmr": "wo_pretrain", "hcmr_cnn": "cnn_wo_pretrain"}
T, L = 60, 4


def runs(root, methods, field):
    rows = []
    for m in methods:
        for f in sorted(root.glob(f"*/{m}/seed_*.json")):
            r = json.loads(f.read_text())
            if field in r:
                rows += [dict(method=m, enc=r["encoding"], setup=f.parent.parent.name, seed=r["seed"], k=int(j), acc=v * 100)
                         for j, v in r[field].items()]
    return pd.DataFrame(rows, columns=["method", "enc", "setup", "seed", "k", "acc"])


def curve(df, m, enc):
    per_seed = df[(df.method == m) & (df.enc == enc)].groupby(["seed", "k"]).acc.mean().unstack()
    return per_seed.mean(), per_seed.std()


def circuits(enc, n, J):
    pqc, emb = 1 + 2 * 3 * L * n, 2 * (n if enc == "angle" else 2 ** n - 1)
    return T * pqc + J * emb


sweep = runs(ROOT / "results" / "j_sweep", ("hcmr", "hcmr_cnn", "wo_pretrain", "cnn_wo_pretrain"), "joint_test_acc")
window = runs(ROOT / "results" / "joint_window", ("hcmr", "hcmr_cnn"), "window_test_acc")
window = window[window.setup.str.contains("_6q")]

fig, axes = plt.subplots(1, 4, figsize=(13, 2.9))
ax, bx, cx, dx = axes
for m, enc, name, col in GROUPS:
    mu, sd = curve(sweep, m, enc)
    ax.plot(T - mu.index, mu.values, "-o", color=col, ms=3.5)
    ax.fill_between(T - mu.index, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
CAPS = ["(a) when joint training starts"]; ax.set_ylabel("test accuracy [%]")
ax.set_xlabel("joint start epoch s"); ax.set_xticks(range(0, 61, 10))

if len(window):
    for m, enc, name, col in GROUPS:
        mu, sd = curve(window, m, enc)
        bx.plot(mu.index, mu.values, "-o", color=col, ms=3.5)
        bx.fill_between(mu.index, mu - sd, mu + sd, color=col, alpha=0.15, lw=0)
else:
    bx.text(0.5, 0.5, "pending", transform=bx.transAxes, ha="center", color=S.MUTED)
CAPS.append("(b) J = 10, window position"); bx.set_ylabel("test accuracy [%]")
bx.set_xlabel("window start s"); bx.set_xlim(-3, 53); bx.set_xticks(range(0, 51, 10))

Js = np.arange(0, T + 1, 5)
for n, shade in zip((4, 6, 8, 10), ("#c9c7c0", "#b3b0a8", "#8a877f", "#3d3b37")):
    cx.plot(Js, [circuits("amplitude", n, J) / circuits("amplitude", n, 0) for J in Js], "-", color=shade,
            label=f"amplitude, {n}q")
cx.plot(Js, [circuits("angle", 6, J) / circuits("angle", 6, 0) for J in Js], ls=(0, (1, 1)), color=S.INK2,
        label="angle, 4-10q")
cx.legend(fontsize=8.9, loc="upper left", ncol=2, handlelength=1.4, columnspacing=0.8, borderaxespad=0.3)
cx.set_ylim(0.5, 12.5)
cx.set_yticks(range(2, 13, 2))
cx.axvline(10, color=S.INK2, lw=0.6, ls=":")
CAPS.append("(c) training circuit executions"); cx.set_ylabel("circuit runs (× J = 0)")

complete = all(len(sweep[(sweep.method == WOP[m]) & (sweep.enc == e)].groupby(["setup", "seed"])) == 30
               for m, e, _, _ in GROUPS)
if complete:
    acc = {(meth, enc): curve(sweep, meth, enc)[0] for m, enc, _, _ in GROUPS for meth in (m, WOP[m])}
    short = {"HCMR": {}, "w/o pretraining": {}}
    for m, enc, name, _ in GROUPS:
        peak = max(acc[(m, enc)].max(), acc[(WOP[m], enc)].max())
        short["HCMR"][name], short["w/o pretraining"][name] = 100 * acc[(m, enc)] / peak, 100 * acc[(WOP[m], enc)] / peak
    for lab, col, a, z in (("HCMR", "#b2182b", 0.7, 2), ("w/o pretraining", S.MUTED, 0.5, 1)):
        G = pd.DataFrame(short[lab])
        dx.fill_between(G.index, G.min(1), G.max(1), color=col, alpha=a, lw=0, zorder=z, label=lab)
    dx.axvline(10, color=S.INK2, lw=0.6, ls=":")
    dx.set_ylim(60, 102)
    dx.legend(fontsize=10.2, loc="lower right")
else:
    dx.text(0.5, 0.5, "pending", transform=dx.transAxes, ha="center", color=S.MUTED)
CAPS.append("(d) reaching the peak"); dx.set_ylabel("% of peak")
for a in (cx, dx):
    a.set_xlabel("joint epochs J (of 60)"); a.set_xticks(range(0, 61, 10))
handles = [Line2D([], [], color=c, marker="o", ms=3.5, label=n) for _, _, n, c in GROUPS]
fig.legend(handles, [h.get_label() for h in handles], loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0.08, 1, 0.86))
r_ = fig.canvas.get_renderer()
for a, lab in zip(axes, CAPS):
    bb = a.get_tightbbox(r_).transformed(fig.transFigure.inverted()); p_ = a.get_position()
    fig.text((p_.x0 + p_.x1) / 2, bb.y0 - 0.02, lab, ha="center", va="top", fontsize=13)
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"joint_training.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig); print("saved", FIG / "joint_training.png")
