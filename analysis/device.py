import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hcmr"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import figstyle as S
from matplotlib.patches import Patch
from noise_circuits import observables_from_probs
from device_fez import device_index


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


S.apply()
ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
B = ROOT / "results" / "noise" / "device" / "fez"
VARS = [("base", "reference (L=4)", "#3d3b37"), ("L8", "depth L=8", "#b3b0a8"), ("L16", "depth L=16", "#8a877f"),
        ("zz", "Z+ZZ readout", "#e8a33d"), ("hcmr", "HCMR (linear)", "#4b2c86"), ("hcmr_cnn", "HCMR (CNN)", "#1b7837")]
SETS = [("amplitude", "amplitude, 4 qubits"), ("angle", "angle, 4 qubits")]
XT = ["ref.", "L8", "L16", "ZZ", "HCMR\nlin.", "HCMR\nCNN"]


def stats(enc, v):
    P = dict(np.load(B / "prepare" / f"{enc}_mnist_4q4c" / v / "seed_0.npz"))
    F = np.load(B / "fetch" / f"{enc}_mnist_4q4c" / v / "seed_0_probs.npz")
    pd, ps = F["prob_device"].astype(float), F["prob_sim_runtime_cal"].astype(float)
    if len(P["y"]) != len(pd):
        ix = device_index(P["y"]); P = {k: (a[ix] if len(a) == len(P["y"]) else a) for k, a in P.items()}
    meta = json.loads((B / "prepare" / f"{enc}_mnist_4q4c" / v / "seed_0.json").read_text())
    lg = lambda ps_: np.array([observables_from_probs(p, 4, meta["readout"], meta["n_meas"]) for p in ps_]) @ P["head_w"].T + P["head_b"]
    pi = P["prob_ideal"].astype(float)
    Li, Ld = lg(pi), lg(pd)
    ci, cd = Li.argmax(1) == P["y"], Ld.argmax(1) == P["y"]
    u = 0.5 * np.abs(pi - 1 / pi.shape[1]).sum(1)
    tvn = lambda q: float((0.5 * np.abs(pi - q).sum(1) / u).mean())
    s = np.sort(Li, 1)
    obs = lambda ps_: np.array([observables_from_probs(p, 4, meta["readout"], meta["n_meas"]) for p in ps_])
    Oi, Od = obs(pi), obs(pd)
    o = np.argsort(Li, 1); t, r2 = o[:, -1], o[:, -2]; n = np.arange(len(Li))
    gap = Li[n, t] - Li[n, r2]
    dist = gap / np.linalg.norm(P["head_w"][t] - P["head_w"][r2], axis=1)
    return dict(acc_i=100 * ci.mean(), acc_d=100 * cd.mean(), flips=int((ci & ~cd).sum()), tvd_s=tvn(ps), tvd_d=tvn(pd),
                margin=float(np.median(s[:, -1] - s[:, -2])), head_norm=float(np.linalg.norm(P["head_w"])),
                boundary_dist_obs=float(np.median(dist)), device_shift_obs=float(np.median(np.linalg.norm(Od - Oi, axis=1))),
                top_class_share_device=float(np.bincount(Ld.argmax(1), minlength=4).max() / len(Ld)))


ST = {(enc, v): stats(enc, v) for enc, _ in SETS for v, _, _ in VARS}
AN = {r["key"]: r for r in json.loads((B / "analysis.json").read_text())}
for (enc, v), d in ST.items():
    r = AN[f"{enc}_mnist_4q4c/{v}/seed_0"]
    assert abs(r["device"]["tvd_norm"] - d["tvd_d"]) < 1e-6 and abs(r["device"]["acc"] * 100 - d["acc_d"]) < 1e-6
    d["tvd_s"] = r["sim_runtime_cal"]["tvd_norm"]; d["acc_s"] = 100 * r["sim_runtime_cal"]["acc"]
(B / "analysis_4q_margin.json").write_text(json.dumps({f"{e}/{v}": d for (e, v), d in ST.items()}, indent=1))

import matplotlib.pyplot as plt
plt.rcParams["hatch.linewidth"] = 0.7
SIM_C, DEV_C = "#f4a582", "#b2182b"
LOSS = dict(facecolor="#f4c7c3", edgecolor="#b2182b", hatch="//////", lw=0)


def change_bar(ax, k, ref, dev, col, w=0.7):
    ax.bar(k, dev, w, color=col)
    if dev < ref:
        ax.bar(k, ref - dev, w, bottom=dev, **LOSS)


fig, axes = S.grid(3, 2, height_per_row=2.6, width=7.4)
for j, (enc, title) in enumerate(SETS):
    a1, a2, a3 = axes[0, j], axes[1, j], axes[2, j]
    for k, (v, _, col) in enumerate(VARS):
        d = ST[(enc, v)]
        change_bar(a1, k, d["acc_s"], d["acc_d"], col)
        a2.bar(k - 0.19, d["tvd_s"], 0.36, color=SIM_C)
        a2.bar(k + 0.19, d["tvd_d"], 0.36, color=DEV_C)
        a3.bar(k, d["margin"], 0.7, color=col)
    a1.set_ylim(0, 100); a2.set_ylim(0, 1.0)
    for a in (a1, a2, a3):
        a.set_xticks(range(len(VARS))); a.set_xticklabels(XT, fontsize=9.5); a.grid(axis="x", visible=False)
    a1.set_title(title, fontweight="bold")
axes[0, 0].set_ylabel("accuracy [%]"); axes[1, 0].set_ylabel("normalized TVD"); axes[2, 0].set_ylabel("logit margin")
h = [Patch(color=SIM_C), Patch(color=DEV_C), Patch(**LOSS)]
l = ["noisy simulation", "ibm_fez", "lost on ibm_fez"]
fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(rect=(0, 0.03, 1, 0.95), h_pad=3.2)
r = fig.canvas.get_renderer()
for i, lab in enumerate(["(a) accuracy", "(b) normalized TVD", "(c) decision margin"]):
    y = min(a.get_tightbbox(r).transformed(fig.transFigure.inverted()).y0 for a in axes[i]) - 0.008
    x = (axes[i, 0].get_position().x0 + axes[i, -1].get_position().x1) / 2
    fig.text(x, y, lab, ha="center", va="top", fontsize=13)
for ext in ("png", "pdf"):
    fig.savefig((FIG / "device").with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
plt.close(fig); print("saved", FIG / "device.png")
