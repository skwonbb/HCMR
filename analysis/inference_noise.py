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
D = Path(__file__).resolve().parent.parent / "results" / "noise" / "infer"
BACK = [("fez", "FakeFez (Heron r2)"), ("nighthawk", "FakeNighthawk (Nighthawk r1)")]
COLS = [("amplitude", 4), ("amplitude", 8), ("angle", 4), ("angle", 8)]
VARS = [("base", "ref."), ("L8", "L=8"), ("L16", "L=16"), ("zz", "Z+ZZ"), ("hcmr", "HCMR\nlin."), ("hcmr_cnn", "HCMR\nCNN")]
CONDS = [("shots_only", "prob_shots", "shot noise", "#c9c7c0"), ("noise_only", "prob_noise", "device noise", "#e8a33d"),
         ("both", "prob_both", "both", "#b2182b")]
REST = "#5f5d57"


def values(be, enc, nq, var):
    acc, tvd = {c: [] for c in ["ideal"] + [c for c, *_ in CONDS]}, {c: [] for c, *_ in CONDS}
    ntvd = {c: [] for c, *_ in CONDS}
    for s in range(5):
        a, t = {c: [] for c in ["ideal"] + [c for c, *_ in CONDS]}, {c: [] for c, *_ in CONDS}
        nt = {c: [] for c, *_ in CONDS}
        for ds in ("mnist", "fmnist"):
            f = D / be / f"{enc}_{ds}_{nq}q4c" / var / f"seed_{s}"
            r = json.loads(f.with_suffix(".json").read_text()); z = np.load(f.with_suffix(".npz"))
            pi = z["prob_ideal"].astype(float); a["ideal"].append(100 * r["ideal"]["acc"])
            to_uniform = 0.5 * np.abs(pi - 1 / pi.shape[1]).sum(1)
            for c, key, *_ in CONDS:
                a[c].append(100 * r[c]["acc"])
                tv = 0.5 * np.abs(pi - z[key].astype(float)).sum(1)
                t[c].append(tv.mean()); nt[c].append((tv / to_uniform).mean())
        for c in a:
            acc[c].append(np.mean(a[c]))
        for c, *_ in CONDS:
            tvd[c].append(np.mean(t[c])); ntvd[c].append(np.mean(nt[c]))
    st = lambda d: {c: (np.mean(v), np.std(v), v) for c, v in d.items()}
    return st(acc), st(tvd), st(ntvd)


VCOL = {"base": "#3d3b37", "L8": "#b3b0a8", "L16": "#8a877f", "zz": "#e8a33d", "hcmr": "#4b2c86", "hcmr_cnn": "#1b7837"}
x = np.arange(len(VARS))
VAL = {(be, enc, nq, v): values(be, enc, nq, v) for be, _ in BACK for enc, nq in COLS for v, _ in VARS}

CE_C, PE_C = "#e58ca8", "#3a6fb0"
plt.rcParams["hatch.linewidth"] = 0.7
LOSS = dict(facecolor="#f4c7c3", edgecolor="#b2182b", hatch="//////", lw=0)
GAIN = dict(facecolor="#d6e3f3", edgecolor="#2c5d9e", hatch="\\\\\\\\\\\\", lw=0)


def change_bar(ax, xi, ideal, noisy, width, color):
    ax.bar(xi, min(ideal, noisy), width, color=color)
    if noisy < ideal:
        ax.bar(xi, ideal - noisy, width, bottom=noisy, **LOSS)
    elif noisy > ideal:
        ax.bar(xi, noisy - ideal, width, bottom=ideal, **GAIN)
CAP = 0.25
from matplotlib.lines import Line2D


def label(axes, backs, ylab):
    S.label_grid(axes, [f"{e}, {q} qubits" for e, q in COLS], [b for _, b in backs], ylab)


def draw(backs, names):
    nr = len(backs)
    fig = plt.figure(figsize=(15, 2 * nr * 2.3 + 2 * CAP + 0.4))
    ga, ca, gb, cb = fig.add_gridspec(4, 1, height_ratios=[nr * 2.3, CAP, nr * 2.3, CAP])
    A = ga.subgridspec(nr, 4).subplots(squeeze=False); B = gb.subgridspec(nr, 4).subplots(squeeze=False)
    for r, (be, _) in enumerate(backs):
        for c_, (enc, nq) in enumerate(COLS):
            ax, bx = A[r, c_], B[r, c_]
            for i, (v, _) in enumerate(VARS):
                ideal, both = (VAL[(be, enc, nq, v)][0][k][0] for k in ("ideal", "both"))
                change_bar(ax, i, ideal, both, 0.7, VCOL[v])
                S.dots(ax, i, VAL[(be, enc, nq, v)][0]["both"][2], 0.7, VCOL[v])
                rs = [json.loads((D / be / f"{enc}_{ds}_{nq}q4c" / v / f"seed_{s_}.json").read_text())
                      for ds in ("mnist", "fmnist") for s_ in range(5)]
                for k, (reg, col) in enumerate((("acc_ce", CE_C), ("acc_pe", PE_C))):
                    ide = 100 * np.mean([q["ideal"][reg] for q in rs]); bo = 100 * np.mean([q["both"][reg] for q in rs])
                    change_bar(bx, i + (k - 0.5) * 0.38, ide, bo, 0.36, col)
                    per_seed = [100 * np.mean([rs[d_ * 5 + s_]["both"][reg] for d_ in range(2)]) for s_ in range(5)]
                    S.dots(bx, i + (k - 0.5) * 0.38, per_seed, 0.36, col)
            for q in (ax, bx):
                q.set_xticks(x); q.set_xticklabels([n for _, n in VARS], fontsize=9.5); q.grid(axis="x", visible=False)
                q.axvline(3.5, color=S.INK2, lw=0.6, ls=":"); q.set_ylim(0, 100)
    label(A, backs, "test accuracy [%]"); label(B, backs, "test accuracy [%]")
    for g, lab in ((ca, "(a) accuracy with inference noise (device + shot)"), (cb, "(b) accuracy on CE (HEM < 0) and PE")):
        cx = fig.add_subplot(g); cx.axis("off"); cx.text(0.5, 0.5, lab, ha="center", va="center", fontsize=14, transform=cx.transAxes)
    h = [Patch(facecolor="#9a988f"), Patch(color=CE_C), Patch(color=PE_C), Patch(**LOSS), Patch(**GAIN)]
    l = ["(a) accuracy", "(b) CE accuracy", "(b) PE accuracy", "lost to inference noise", "gained under inference noise"]
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.035 * 2 / nr))
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{names['acc']}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig); print("saved", FIG / f"{names['acc']}.png")

    fig, axes = S.grid(nr, 4, height_per_row=2.6, width=15)
    w = 0.26
    for r, (be, _) in enumerate(backs):
        for c, (enc, nq) in enumerate(COLS):
            ax = axes[r, c]
            for k, (cond, _, lab, col) in enumerate(CONDS):
                m = [VAL[(be, enc, nq, v)][2][cond][0] for v, _ in VARS]; e = [VAL[(be, enc, nq, v)][2][cond][1] for v, _ in VARS]
                ax.bar(x + (k - 1) * w, m, w, yerr=e, color=col, label=lab, error_kw=dict(lw=0.7, capsize=1.5, ecolor=S.INK2))
                for i, (v, _) in enumerate(VARS):
                    S.dots(ax, x[i] + (k - 1) * w, VAL[(be, enc, nq, v)][2][cond][2], w, col)
            ax.set_xticks(x); ax.set_xticklabels([n for _, n in VARS], fontsize=9.5); ax.grid(axis="x", visible=False)
            ax.axvline(3.5, color=S.INK2, lw=0.6, ls=":"); ax.set_ylim(0, 1.05)
    label(axes, backs, "TVD / TVD(ideal, uniform)")
    h, l = axes[0, 0].get_legend_handles_labels()
    S.finish(fig, FIG / names["tvd"], h, l, ncol=3, top=0.92 if nr == 2 else 0.86)

    if not names.get("gates"):
        return
    fig, axes = S.grid(nr, 4, height_per_row=2.4, width=15)
    DV = [("base", "L=4\n(ref., Z+ZZ, HCMR)"), ("L8", "L=8"), ("L16", "L=16")]
    for r, (be, _) in enumerate(backs):
        for c, (enc, nq) in enumerate(COLS):
            ax = axes[r, c]
            for i, (v, _) in enumerate(DV):
                prep = PREP[f"{be}/{enc}_{nq}q/{v}"]
                tot = np.mean([json.loads((D / be / f"{enc}_{ds}_{nq}q4c" / v / f"seed_{s_}.json").read_text())["two_qubit_gates_mean"]
                               for ds in ("mnist", "fmnist") for s_ in range(5)])
                ax.bar(i, prep, 0.6, color="#5f5d57"); ax.bar(i, tot - prep, 0.6, bottom=prep, color="#c9c7c0")
                ax.text(i, tot + 20, f"{tot:.0f}", ha="center", va="bottom", fontsize=9.5, color=S.INK)
            ax.set_xticks(range(len(DV))); ax.set_xticklabels([n for _, n in DV], fontsize=9.5); ax.grid(axis="x", visible=False)
            ax.set_ylim(0, 1500)
    label(axes, backs, "2-qubit gates (cz)")
    h = [Patch(color="#5f5d57"), Patch(color="#c9c7c0")]; l = ["state preparation (embedding)", "PQC (incl. routing SWAPs)"]
    S.finish(fig, FIG / names["gates"], h, l, ncol=2, top=0.92 if nr == 2 else 0.86)


PREP = json.loads((D / "state_prep_2q_gates.json").read_text())
draw(BACK[:1], dict(acc="inference_noise_accuracy", tvd="inference_noise_tvd", gates=None))
