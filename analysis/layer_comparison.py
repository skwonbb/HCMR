import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as S

S.apply()
ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
EXT = Path(__file__).resolve().parent.parent / "results"
LAY = EXT / "layers"

rows = []
for f in LAY.glob("*/*/*/*/seed_*.json"):
    enc, ds, su, cond = f.parts[-5:-1]; r = json.loads(f.read_text())
    rows.append(dict(enc=enc, ds=ds, nq=int(su.split("q")[0]), nc=int(su.split("q")[1][:-1]), v=cond, seed=r["seed"],
                     acc=r["acc"] * 100, ce=r["acc_ce"] * 100))
full = pd.read_pickle(EXT / "full" / "summary_runs.pkl")
for m, v in (("pool", "base"), ("pca", "PCA"), ("hcmr", "HCMR (linear)"), ("hcmr_cnn", "HCMR (CNN)")):
    for _, r in full[full.method == m].iterrows():
        rows.append(dict(enc=r.enc, ds=r.ds, nq=r.nq, nc=r.nc, v=v, seed=r.seed, acc=r.acc, ce=r.ce))
D = pd.DataFrame(rows); D.to_pickle(LAY / "summary_runs.pkl")

BARS = [("base", "reference (avg pooling, L=4, Z_k)", "#3d3b37"),
        ("z_k_L2", "depth L=2", "#d6d4ce"), ("z_k_L8", "depth L=8", "#b3b0a8"), ("z_k_L16", "depth L=16", "#8a877f"),
        ("z_zz_L4", "Z+ZZ readout", "#e8a33d"),
        ("PCA", "PCA", "#cbbfe0"), ("HCMR (linear)", "HCMR (linear)", "#4b2c86"), ("HCMR (CNN)", "HCMR (CNN)", "#1b7837")]
COLS = [(d, c) for d in ("mnist", "fmnist", "cifar10") for c in (4, 10)]
DSN = {"mnist": "MNIST", "fmnist": "FMNIST", "cifar10": "CIFAR-10"}
fig, axes = S.grid(2, 6, height_per_row=2.6)
for i, enc in enumerate(("amplitude", "angle")):
    for j, (ds, nc) in enumerate(COLS):
        ax = axes[i, j]
        x = D[(D.enc == enc) & (D.ds == ds) & (D.nc == nc)]
        for k, (v, name, col) in enumerate(BARS):
            per_seed = x[x.v == v].groupby("seed").ce.mean()
            ax.bar(k, per_seed.mean(), 0.8, yerr=per_seed.std(), color=col, label=name,
                   error_kw=dict(lw=0.8, capsize=1.5, ecolor=S.INK2))
            S.dots(ax, k, per_seed.values, 0.8, col)
        ax.axvline(4.5, color=S.INK2, lw=0.6, ls=":")
        ax.set_xticks([]); ax.set_ylim(0, 100); ax.grid(axis="x", visible=False)
S.label_grid(axes, [f"{DSN[ds]}, {nc} classes" for ds, nc in COLS], ["amplitude", "angle"], "accuracy on CE [%]")
h, l = axes[0, 0].get_legend_handles_labels()
S.finish(fig, FIG / "layer_comparison", h, l, ncol=8, top=0.9)
