import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "hcmr"))
from config import CLASS_ORDER, N_TRAIN, N_VAL, N_TEST, DATA_SEED, BATCH
from data import load_data
from hem import hem_split
from qnn import pool_reducer

R_GRID = np.round(np.arange(0.05, 0.751, 0.05), 2)
E = Path(__file__).resolve().parent.parent / "results"


def runs(ds, su, cond):
    return [json.loads(p.read_text()) for p in sorted((E / "layers" / "amplitude" / ds / su / cond).glob("seed_*.json"))]


print(f"{'setup':16s} {'pi_neg':>6s} | {'raw CE/PE':>11s} {'psi CE/PE':>11s} | {'z_k CE/PE':>11s} {'z_zz CE/PE':>11s} |"
      f" {'CE inacc/lost':>13s} {'PE inacc/lost':>13s} | {'r*':>5s} {'F1@r*':>6s} {'F1@20':>6s} {'F1@pi':>6s}")
summary = []
for ds in ("mnist", "fmnist", "cifar10"):
    for nq in (4, 6, 8, 10):
        for nc in (2, 4, 6, 10):
            su = f"{nq}q{nc}c"; cf = E / "ceiling" / ds / f"{su}.json"
            zk, zz = runs(ds, su, "z_k_L4"), runs(ds, su, "z_zz_L4")
            if not cf.exists() or not zk:
                continue
            c = json.loads(cf.read_text())
            y = {int(k): v for k, v in c["y_test"].items()}; ids = sorted(y)
            hem = np.array([c["hem_test"][str(i)] for i in ids]); yy = np.array([y[i] for i in ids])
            ce = hem < 0
            pk = {k: v for k, v in c["probes"].items() if k in ("psi", "psi2")}
            best = max(pk, key=lambda k: pk[k]["val_acc"])
            pok = np.array([int(pk[best]["per_idx"][str(i)]) == y[i] for i in ids])
            tl, *_ = load_data(ds, N_TRAIN, N_VAL, N_TEST, BATCH, DATA_SEED, CLASS_ORDER[ds][:nc])
            tr = np.array(list(hem_split(pool_reducer("amplitude", nq, nc), "amplitude", nq, nc, tl, tl)[0].values()))
            taus = np.quantile(tr, R_GRID)

            def region(rs):
                o = np.array([[r["per_idx"][str(i)] == y[i] for i in ids] for r in rs])
                return o[:, ce].mean() * 100, o[:, ~ce].mean() * 100, o

            zk_ce, zk_pe, ok = region(zk)
            zz_ce, zz_pe, _ = region(zz) if zz else (np.nan, np.nan, None)
            inacc = lambda m: np.mean([(pok & ~o)[m].mean() for o in ok]) * 100
            lost = lambda m: (~pok)[m].mean() * 100

            def f1(mask, err):
                tp = (mask & err).sum(); return 2 * tp / max(mask.sum() + err.sum(), 1)
            F = np.array([[f1(hem < t, ~o) for t in taus] for o in ok]).mean(0)
            pi = c["pi_neg_train"]
            f_pi = np.mean([f1(hem < 0, ~o) for o in ok])
            r_star = R_GRID[F.argmax()]
            raw = c["probes"]["raw"]
            print(f"{ds + ' ' + su:16s} {pi*100:5.1f}% | {raw['acc_ce']*100:5.1f}/{raw['acc_pe']*100:5.1f} "
                  f"{pk[best]['acc_ce']*100:5.1f}/{pk[best]['acc_pe']*100:5.1f} | {zk_ce:5.1f}/{zk_pe:5.1f} {zz_ce:5.1f}/{zz_pe:5.1f} |"
                  f" {inacc(ce):5.1f}/{lost(ce):5.1f}   {inacc(~ce):5.1f}/{lost(~ce):5.1f}   | {r_star:5.2f} {F.max():6.3f}"
                  f" {F[list(R_GRID).index(0.2)]:6.3f} {f_pi:6.3f}", flush=True)
            summary.append(dict(ds=ds, nq=nq, nc=nc, pi_neg=pi, r_star=float(r_star),
                                err_rate=float(np.mean([(~o).mean() for o in ok])),
                                f1_hem0=float(f_pi), f1_rstar=float(F.max()), f1_20=float(F[list(R_GRID).index(0.2)])))

s = summary
pi = np.array([x["pi_neg"] for x in s]); rs = np.array([x["r_star"] for x in s]); er = np.array([x["err_rate"] for x in s])
print(f"\ncorr(r*, pi_neg) = {np.corrcoef(rs, pi)[0, 1]:.3f}   corr(r*, PQC error rate) = {np.corrcoef(rs, er)[0, 1]:.3f}")
print(f"mean |r* - 0.20| = {np.abs(rs - 0.2).mean():.3f}   mean |r* - pi_neg| = {np.abs(rs - pi).mean():.3f}"
      f"   mean |r* - err_rate| = {np.abs(rs - er).mean():.3f}")
g = [x for x in s if x["ds"] in ("mnist", "fmnist")]
if g:
    gp, gr = np.array([x["pi_neg"] for x in g]), np.array([x["r_star"] for x in g])
    print()
    print(f"MNIST + FMNIST ({len(g)} setups): F1 HEM<0 {np.mean([x['f1_hem0'] for x in g]):.3f} | post-hoc r* "
          f"{np.mean([x['f1_rstar'] for x in g]):.3f} | fixed 20% {np.mean([x['f1_20'] for x in g]):.3f} | corr(r*, pi_neg) "
          f"{np.corrcoef(gr, gp)[0, 1]:.3f} | mean |r* - pi_neg| {np.abs(gr - gp).mean():.3f} | mean |r* - 0.20| {np.abs(gr - 0.2).mean():.3f}")
(E / "phase2_summary.json").write_text(json.dumps(s, indent=1))
