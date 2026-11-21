import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "hcmr"))
from config import SETUPS, RESULTS, tag
OUT = RESULTS / "full"

GROUPS = [("linear", ["pool", "pca", "lloyd", "e2e", "hcmr"]), ("nonlinear (CNN)", ["nqe", "hcmr_cnn"])]
MAIN = [m for _, ms in GROUPS for m in ms]
ABL = ["hcmr", "hcmr_frozen", "wo_lambda", "wo_mu"]
NAME = {"pool": "avg pooling", "pca": "PCA", "lloyd": "Lloyd", "e2e": "e2e", "hcmr": "**HCMR (linear)**", "nqe": "NQE",
        "hcmr_cnn": "**HCMR (CNN)**", "hcmr_frozen": "w/o joint training", "wo_lambda": "w/o λ", "wo_mu": "w/o μ"}
DSN = {"mnist": "MNIST", "fmnist": "FMNIST", "cifar10": "CIFAR-10"}
ENCN = {"amplitude": "amplitude", "angle": "angle"}

rows = []
for s in SETUPS:
    for m in MAIN + ABL[1:]:
        for f in sorted((OUT / tag(*s) / m).glob("seed_*.json")):
            r = json.loads(f.read_text()); sd = r["seed"]
            rows.append(dict(enc=s[0], ds=s[1], nq=s[2], nc=s[3], setup=tag(*s), method=m, seed=sd, acc=r["acc"] * 100,
                             ce=r["acc_ce_orig"] * 100, pe=r["acc_pe_orig"] * 100, cs=r["ce_size_own"] * 100,
                             J=r.get("joint_epochs"), mu=(r.get("margin") or {}).get("mu"),
                             lam=(r.get("margin") or {}).get("lam"),
                             **{f"J{j}": v * 100 for j, v in (r.get("joint_test_acc") or {}).items()}))
df = pd.DataFrame(rows)
sm = df.groupby(["setup", "method"]).acc.mean().unstack()
sdv = df.groupby(["setup", "method"]).acc.std().unstack()
meta = df.groupby("setup")[["enc", "ds", "nq", "nc"]].first()
out = []
P = out.append


def fmt(x, b, base):
    return f"{x:.1f}" if base else f"{x:.1f} ({x - b:+.1f})"


P("## Main table: mean per dataset (16 settings x 5 seeds; in brackets: pp vs. avg pooling)\n")
P("CE / PE accuracy: on the CE / PE samples of avg pooling (fixed). CE size: share of HEM < 0 under each method's own encoding.\n")
g = df.groupby(["ds", "method"]).mean(numeric_only=True)
win = sm[MAIN].idxmax(1)
hdr = "| encoder | method | " + " | ".join(f"{DSN[d]} {k}" for d in DSN for k in ("CE size", "CE", "PE", "overall")) + " | best (of 48) |"
P(hdr); P("|" + "---|" * (hdr.count("|") - 1))
for gname, ms in GROUPS:
    for m in ms:
        c = []
        for d in DSN:
            x, p = g.loc[(d, m)], g.loc[(d, "pool")]
            for k in ("cs", "ce", "pe", "acc"):
                c.append(fmt(x[k], p[k], m == "pool"))
        P(f"| {gname} | {NAME[m]} | " + " | ".join(c) + f" | {int((win == m).sum())} |")
P("")

P("## Comparison within the same encoder family (48 settings, seed mean)\n")
P("| comparison | settings won | mean difference (pp) | largest loss (pp) |")
P("|---|---|---|---|")
for a, b in [("hcmr", "e2e"), ("hcmr", "lloyd"), ("hcmr", "pca"), ("hcmr_cnn", "nqe")]:
    d = sm[a] - sm[b]
    P(f"| {NAME[a]} vs. {NAME[b]} | {int((d > 0).sum())} / 48 | {d.mean():+.1f} | {d[d < 0].min() if (d < 0).any() else 0:+.1f} |")
lin_win = sm[["lloyd", "e2e", "hcmr"]].idxmax(1).value_counts().to_dict()
P(f"\nbest in the linear family: {lin_win}\n")
d = sm["hcmr"] - sm["nqe"]
P(f"across families: linear HCMR >= CNN NQE in {int((d >= 0).sum())} / 48 settings\n")

P("## Mean overall accuracy per embedding x dataset (%)\n")
t = df.groupby(["enc", "ds", "method"]).acc.mean().unstack()[MAIN]
P("| embedding | dataset | " + " | ".join(NAME[m] for m in MAIN) + " |"); P("|" + "---|" * (len(MAIN) + 2))
for (e, d), r in t.iterrows():
    P(f"| {ENCN[e]} | {DSN[d]} | " + " | ".join(f"{r[m]:.1f}" for m in MAIN) + " |")
P("")

P("## Ablation (linear, 48 settings x 5 seeds)\n")
sel = df[df.method == "hcmr"].groupby("setup")[["mu", "lam"]].first()
P("| method | mean | settings where HCMR chose lambda > 0 | settings where HCMR chose mu > 0 |")
P("|---|---|---|---|")
lam_set, mu_set = sel.index[sel.lam > 0], sel.index[sel.mu > 0]
for m in ABL:
    P(f"| {NAME[m]} | {sm[m].mean():.1f} | {sm.loc[lam_set, m].mean():.1f} | {sm.loc[mu_set, m].mean():.1f} |")
P(f"\nlambda > 0 chosen in {len(lam_set)} settings, mu > 0 in {len(mu_set)}.\n")

P("## Selected hyperparameters\n")
for m in ("hcmr", "hcmr_cnn"):
    s2 = df[df.method == m].groupby("setup")[["enc", "mu", "lam"]].first()
    P(f"**{NAME[m]}**: (mu, lambda) chosen")
    for e in ENCN:
        vc = s2[s2.enc == e].groupby(["mu", "lam"]).size().sort_values(ascending=False)
        P(f"- {ENCN[e]}: " + ", ".join(f"({a:g}, {b:g}) x{n}" for (a, b), n in vc.items()))
    jv = df[df.method == m].groupby(["enc", "J"]).size().unstack(fill_value=0)
    P(f"- J chosen (settings x seeds): " + "; ".join(f"{ENCN[e]} " + ", ".join(f"J={j}: {n}" for j, n in r.items()) for e, r in jv.iterrows()))
    P("")

P("## Test accuracy per joint-training length J (mean over settings and seeds, %)\n")
P("| method | embedding | J=0 | J=10 | J=30 | J=60 | selected on validation |"); P("|---|---|---|---|---|---|---|")
for m in ("hcmr", "hcmr_cnn"):
    for e in ENCN:
        x = df[(df.method == m) & (df.enc == e)]
        P(f"| {NAME[m]} | {ENCN[e]} | " + " | ".join(f"{x[f'J{j}'].mean():.1f}" for j in (0, 10, 30, 60)) + f" | {x.acc.mean():.1f} |")
P("")

P("## All 48 settings (overall accuracy %, mean +- std over 5 seeds)\n")
cols = MAIN + ABL[1:]
P("| embedding | dataset | setting | " + " | ".join(NAME[m] for m in cols) + " | selected (mu, lambda) | most selected J |"); P("|" + "---|" * (len(cols) + 5))


def mode_j(v):
    c = v.astype(int).value_counts(); top = sorted(c[c == c.max()].index)
    return "/".join(map(str, top)) + f" ({c.max()}/{len(v)})"
jsel = df[df.method == "hcmr"].groupby("setup").J.apply(mode_j)
order = {"mnist": 0, "fmnist": 1, "cifar10": 2}
for st in sorted(sm.index, key=lambda k: (meta.loc[k, "enc"], order[meta.loc[k, "ds"]], meta.loc[k, "nc"], meta.loc[k, "nq"])):
    e, d, q, c = meta.loc[st]
    best = sm.loc[st, MAIN].max()
    cell = lambda m: f"{sm.loc[st, m]:.1f} ± {sdv.loc[st, m]:.1f}"
    cells = [(f"**{cell(m)}**" if m in MAIN and sm.loc[st, m] == best else cell(m)) for m in cols]
    P(f"| {ENCN[e]} | {DSN[d]} | {q}q{c}c | " + " | ".join(cells) + f" | ({sel.loc[st, 'mu']:g}, {sel.loc[st, 'lam']:g}) | {jsel[st]} |")

(OUT / "summary_tables.md").write_text("\n".join(out), encoding="utf-8")
df.to_pickle(OUT / "summary_runs.pkl")
print("\n".join(out[:80]))
