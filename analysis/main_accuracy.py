import os, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as S

S.apply()
ROOT = Path(__file__).resolve().parent.parent
EXT = Path(__file__).resolve().parent.parent / "results"
df = pd.read_pickle(EXT / "full" / "summary_runs.pkl")
FIG = ROOT / "figures"; FIG.mkdir(parents=True, exist_ok=True)
METHODS = [("pool", "avg pooling", "#e7e1ef", 1.2), ("pca", "PCA", "#cbbfe0", 1.2), ("lloyd", "Lloyd", "#a996cc", 1.4),
           ("e2e", "e2e", "#8a73b8", 1.5), ("hcmr", "HCMR (linear)", "#4b2c86", 2.8),
           ("nqe", "NQE (CNN)", "#a6dba0", 1.5), ("hcmr_cnn", "HCMR (CNN)", "#1b7837", 2.8)]
COLS = [(d, c) for d in ("mnist", "fmnist", "cifar10") for c in (4, 10)]
DSN = {"mnist": "MNIST", "fmnist": "FMNIST", "cifar10": "CIFAR-10"}
QS = (4, 6, 8, 10)

fig, axes = S.grid(2, 6, sharex=True)
for i, enc in enumerate(("amplitude", "angle")):
    for j, (ds, nc) in enumerate(COLS):
        ax = axes[i, j]
        x = df[(df.enc == enc) & (df.ds == ds) & (df.nc == nc)]
        for m, name, col, lw in METHODS:
            g = x[x.method == m].groupby("nq")["acc"]
            mu, sd = g.mean().reindex(QS).values, g.std().reindex(QS).values
            ax.plot(QS, mu, "-o", color=col, lw=lw, ms=3 if lw < 2 else 4, label=name, zorder=3 if lw > 2 else 2)
            if lw > 2:
                ax.fill_between(QS, mu - sd, mu + sd, color=col, alpha=0.18, lw=0)
        ax.set_xticks(QS); ax.tick_params(labelbottom=True)
S.label_grid(axes, [f"{DSN[ds]}, {nc} classes" for ds, nc in COLS], ["amplitude", "angle"], "test accuracy [%]", "qubits")
h, l = axes[0, 0].get_legend_handles_labels()
S.finish(fig, FIG / "main_accuracy", h, l, ncol=7)
