import os
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import blended_transform_factory

ROOT = Path(__file__).resolve().parent.parent
E = Path(__file__).resolve().parent.parent / "results"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
import json
SUF = {"amplitude": "", "angle": "_angle"}


def build_cache():
    cache = {}
    for enc in SUF:
        for ds in DS:
            for nq in (4, 6, 8, 10):
                for nc in (4, 10):
                    su = f"{nq}q{nc}c"
                    cf = E / f"ceiling{SUF[enc]}" / ds / f"{su}.json"
                    if not cf.exists():
                        continue
                    c = json.loads(cf.read_text())
                    ids = sorted(int(k) for k in c["y_test"])
                    y = np.array([c["y_test"][str(i)] for i in ids])
                    pk = {k: v for k, v in c["probes"].items() if k in ("psi", "psi2")}
                    b = max(pk, key=lambda k: pk[k]["val_acc"])
                    ok = lambda pi: np.array([int(pi[str(i)]) for i in ids]) == y

                    def runs(sub):
                        return [ok(json.loads(p.read_text())["per_idx"]) for p in sorted(sub.glob("seed_*.json"))]
                    d = dict(y=y, hem=np.array([c["hem_test"][str(i)] for i in ids]), pi=c["pi_neg_train"],
                             psi_ok=ok(pk[b]["per_idx"]), raw_ok=ok(c["probes"]["raw"]["per_idx"]),
                             zk=runs(E / "layers" / enc / ds / su / "z_k_L4"),
                             zz=runs(E / "layers" / enc / ds / su / "z_zz_L4"),
                             mlp=runs(E / f"classical{SUF[enc]}" / ds / su / "mlp_p"))
                    if d["zk"]:
                        cache[(enc, ds, nq, nc)] = d
                        print("cached", enc, ds, su, flush=True)
    return cache


DS = ("mnist", "fmnist", "cifar10")
DS_NAME = {"mnist": "MNIST", "fmnist": "FMNIST", "cifar10": "CIFAR-10"}
ENC = ("amplitude", "angle")
QS = (4, 6, 8, 10)
D_IN = {"mnist": 784, "fmnist": 784, "cifar10": 1024}

CACHE = E / "cache_L4.pkl"
if not CACHE.exists():
    CACHE.write_bytes(pickle.dumps(build_cache()))
C = pickle.loads(CACHE.read_bytes())

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as S
S.apply()
BLUE, CORAL = "#3a6fb0", "#e58ca8"
QRAMP = {4: "#3a6fb0", 6: "#6f6fc0", 8: "#b06fb8", 10: "#e58ca8"}
INK, INK2, MUTED = S.INK, S.INK2, S.MUTED
COLS = [(ds, nc) for ds in DS for nc in (4, 10)]
COL_TITLES = [f"{DS_NAME[ds]}, {nc} classes" for ds, nc in COLS]


def acc(oks, m):
    return np.mean([o[m].mean() for o in oks]) * 100 if m.any() else np.nan


GREYS = {4: "#c9c7c0", 6: "#9a988f", 8: "#5f5d57", 10: "#222222"}
CAP = 0.25
fig = plt.figure(figsize=(17, 2.3 * 2 + 2.0 + 2 * CAP + 1.0))
top, cap_a, bot, cap_b = fig.add_gridspec(4, 1, height_ratios=[2 * 2.3, CAP, 2.0, CAP])
axes = top.subgridspec(2, 6).subplots(squeeze=False)
for r, enc in enumerate(ENC):
    for c, (ds, nc) in enumerate(COLS):
        ax = axes[r, c]
        ax2 = ax.twinx(); ax2.set_zorder(ax.get_zorder() - 1); ax.patch.set_visible(False)
        allh = []
        for q in QS:
            d = C.get((enc, ds, q, nc))
            if d is None:
                continue
            h = d["hem"]; okr = np.mean(d["zk"], 0); allh.append(h)
            edges = np.quantile(h, np.linspace(0, 1, 16)); b = np.clip(np.searchsorted(edges, h, "right") - 1, 0, 14)
            pts = [(h[b == k].mean(), okr[b == k].mean() * 100) for k in range(15) if (b == k).any()]
            ax.plot(*zip(*pts), color=GREYS[q], lw=1.8, label=f"{q} qubits")
        if allh:
            hh = np.concatenate(allh); w = (hh.max() - hh.min()) / 40
            e = np.concatenate([-np.arange(0, -hh.min() + w, w)[:0:-1], np.arange(0, hh.max() + w, w)])
            cnt, e = np.histogram(hh, bins=e); cnt = cnt / cnt.max()
            for sel, col in ((e[:-1] < 0, CORAL), (e[:-1] >= 0, BLUE)):
                ax2.bar(e[:-1][sel], cnt[sel], width=np.diff(e)[sel], align="edge", color=col, alpha=0.35, lw=0)
        ax2.set_ylim(0, 2.2); ax2.set_yticks([]); ax2.grid(False)
        ax.axvline(0, color=INK, lw=1.0, ls="--")
        ax.set_ylim(0, 100)
S.label_grid(axes, COL_TITLES, ENC, "PQC accuracy [%]", "HEM")

bx = bot.subgridspec(1, 6).subplots(squeeze=False)[0]
for ax, (enc, ds) in zip(bx, ((e, d) for e in ENC for d in DS)):
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for i, q in enumerate(QS):
        for nc, off in ((4, -0.2), (10, 0.2)):
            d = C.get((enc, ds, q, nc))
            if d is None:
                continue
            v = (d["hem"] < 0).mean() * 100
            ax.bar(i + off, v, 0.36, color=CORAL, edgecolor="white", lw=0.6)
            ax.bar(i + off, 100 - v, 0.36, bottom=v, color=BLUE, edgecolor="white", lw=0.6)
            ax.text(i + off, v - 1.5, f"{v:.0f}", ha="center", va="top", fontsize=9.5, color=INK)
            ax.text(i + off, -0.04, f"{nc}c", transform=tr, ha="center", va="top", fontsize=9.5, color=INK2)
        r_ = D_IN[ds] / (2 ** q if enc == "amplitude" else q)
        ax.text(i, -0.14, f"{q}q", transform=tr, ha="center", va="top", fontsize=11.4, color=INK)
        ax.text(i, -0.25, f"×{r_:.0f}" if r_ >= 1 else f"×{r_:.2f}", transform=tr, ha="center", va="top", fontsize=9.5, color=INK2)
    ax.set_xticks([]); ax.set_xlim(-0.55, 3.55); ax.set_ylim(0, 100); ax.grid(axis="x", visible=False)
    ax.set_title(f"{enc}, {DS_NAME[ds]}", fontweight="bold")
bx[0].set_ylabel("share of test samples [%]")
h, l = axes[0, 0].get_legend_handles_labels()
h.append(Line2D([], [], color=INK, ls="--", lw=1.0)); l.append("HEM = 0 (CE boundary)")
h += [Patch(color=CORAL, alpha=0.6), Patch(color=BLUE, alpha=0.6)]; l += ["CE (HEM < 0)", "PE (HEM ≥ 0)"]
fig.legend(h, l, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.0))
caps = []
for g, lab in ((cap_a, "(a) PQC accuracy by HEM and distribution of test samples over HEM"),
               (cap_b, "(b) share of CE and PE among test samples")):
    cx_ = fig.add_subplot(g); cx_.axis("off"); caps.append(cx_)
    cx_.text(0.5, 0.5, lab, ha="center", va="center", fontsize=14, transform=cx_.transAxes)
fig.tight_layout(rect=(0, 0, 1, 0.95))
dy = axes[0, 0].get_position().y0 - axes[1, 0].get_position().y1 - 0.045
for ax in [*axes[1], *bx, *caps]:
    p_ = ax.get_position(); ax.set_position([p_.x0, p_.y0 + dy, p_.width, p_.height])
for ext in ("png", "pdf"):
    fig.savefig(FIG / f"hem_accuracy.{ext}", dpi=200, bbox_inches="tight")
plt.close(fig); print("saved", FIG / "hem_accuracy.png")

fig, axes = S.grid(2, 6, height_per_row=2.6)
for r, enc in enumerate(ENC):
    for c, (ds, nc) in enumerate(COLS):
        ax = axes[r, c]; tr = blended_transform_factory(ax.transData, ax.transAxes)
        for k, reg in enumerate(("PE", "CE")):
            x0 = k * 4.6
            for i, q in enumerate(QS):
                d = C.get((enc, ds, q, nc))
                if d is None:
                    continue
                m = d["hem"] < 0 if reg == "CE" else d["hem"] >= 0
                a, b = acc(d["zk"], m), acc(d["mlp"], m)
                ax.bar(x0 + i - 0.2, a, 0.4, color=CORAL if reg == "CE" else BLUE)
                ax.bar(x0 + i + 0.2, b, 0.4, color="#c9c7c0")
                sa, sb = [o[m].mean() * 100 for o in d["zk"]], [o[m].mean() * 100 for o in d["mlp"]]
                S.dots(ax, x0 + i - 0.2, sa, 0.4, CORAL if reg == "CE" else BLUE)
                S.dots(ax, x0 + i + 0.2, sb, 0.4, "#c9c7c0")
                ax.text(x0 + i, max(a, b, *sa, *sb) + 2.5, f"{a - b:+.0f}", ha="center", fontsize=7.5, color=INK)
                ax.text(x0 + i, -0.04, f"{q}", transform=tr, ha="center", va="top", fontsize=9.5, color=INK2)
            ax.text(x0 + 1.5, -0.16, reg, transform=tr, ha="center", va="top", fontsize=11.4, color=INK, fontweight="bold")
        ax.axvline(3.8, color=INK2, lw=0.8)
        ax.set_xticks([]); ax.set_xlim(-0.6, 8.2); ax.set_ylim(0, 112); ax.grid(axis="x", visible=False)
S.label_grid(axes, COL_TITLES, ENC, "accuracy [%]")
h = [Patch(color=BLUE), Patch(color=CORAL), Patch(color="#c9c7c0")]
l = ["PQC on PE", "PQC on CE", "classical MLP, same #params"]
S.finish(fig, FIG / "quantum_gain", h, l, ncol=3)

fig, axes = S.grid(2, 6, sharex=True)
for r, enc in enumerate(ENC):
    for c, (ds, nc) in enumerate(COLS):
        ax = axes[r, c]
        qs = [q for q in QS if (enc, ds, q, nc) in C]
        V = {k: [] for k in ("raw", "psi", "zk")}
        for q in qs:
            d = C[(enc, ds, q, nc)]; ce = d["hem"] < 0
            V["raw"].append(d["raw_ok"][ce].mean() * 100); V["psi"].append(d["psi_ok"][ce].mean() * 100)
            V["zk"].append(acc(d["zk"], ce))
        ax.fill_between(qs, V["zk"], V["psi"], color=CORAL, alpha=0.15, lw=0)
        ax.plot(qs, V["raw"], ":", color=INK, lw=1.6, label="best of LR / SVM / MLP on raw pixels")
        ax.plot(qs, V["psi"], "-o", color="#9a988f", lw=2.0, ms=3.5, label=r"best of LR / SVM / MLP on $\psi$")
        ax.plot(qs, V["zk"], "-o", color=CORAL, lw=2.0, ms=3.5, label="PQC")
        lo = min(min(v) for v in V.values()); hi = max(max(v) for v in V.values())
        ax.set_ylim(max(0, np.floor((lo - 5) / 10) * 10), min(100, np.ceil((hi + 5) / 10) * 10))
        ax.set_xticks(QS); ax.tick_params(labelbottom=True)
S.label_grid(axes, COL_TITLES, ENC, "CE accuracy [%]", "qubits")
h, l = axes[0, 0].get_legend_handles_labels()
S.finish(fig, FIG / "ceilings", h, l, ncol=3)
