import json
from pathlib import Path
import numpy as np

E = Path(__file__).resolve().parent.parent / "results"
SUF = {"amplitude": "", "angle": "_angle"}


def load_runs(enc, ds, su, cond):
    return [json.loads(p.read_text()) for p in sorted((E / "layers" / enc / ds / su / cond).glob("seed_*.json"))]


def f1(mask, err):
    return 2 * (mask & err).sum() / max(mask.sum() + err.sum(), 1)


agg = []
for enc in ("amplitude", "angle"):
    print(f"\n######## {enc} ########")
    print(f"{'setup':15s} {'pi':>5s} | {'psi CE/PE':>11s} | {'PQC zk':>11s} {'PQC zz':>11s} {'cls mlp_p':>11s} |"
          f" {'CE inac/lost':>12s} | {'conc lost/inac/err':>18s} | {'F1 0/20':>11s}")
    for ds in ("mnist", "fmnist", "cifar10"):
        for nq in (4, 6, 8, 10):
            for nc in (4, 10):
                su = f"{nq}q{nc}c"
                cf = E / f"ceiling{SUF[enc]}" / ds / f"{su}.json"
                zk, zz = load_runs(enc, ds, su, "z_k_L4"), load_runs(enc, ds, su, "z_zz_L4")
                if not cf.exists() or not zk:
                    continue
                c = json.loads(cf.read_text())
                ids = sorted(int(k) for k in c["y_test"])
                y = np.array([c["y_test"][str(i)] for i in ids]); hem = np.array([c["hem_test"][str(i)] for i in ids])
                ce = hem < 0
                pk = {k: v for k, v in c["probes"].items() if k in ("psi", "psi2")}
                b = max(pk, key=lambda k: pk[k]["val_acc"])
                pok = np.array([pk[b]["per_idx"][str(i)] for i in ids]) == y
                ok = lambda rs: [np.array([r["per_idx"][str(i)] for i in ids]) == y for r in rs]
                reg = lambda oks: (np.mean([o[ce].mean() for o in oks]) * 100, np.mean([o[~ce].mean() for o in oks]) * 100)
                ok_k, ok_z = ok(zk), ok(zz)
                cls = [json.loads(p.read_text()) for p in
                       sorted((E / f"classical{SUF[enc]}" / ds / su / "mlp_p").glob("seed_*.json"))]
                ok_c = ok(cls) if cls else None
                inacc_ce = np.mean([(pok & ~o)[ce].mean() for o in ok_k]) * 100
                lost_ce = (~pok)[ce].mean() * 100
                conc_lost = ((~pok) & ce).sum() / max((~pok).sum(), 1) * 100
                conc_inac = np.mean([((pok & ~o) & ce).sum() / max((pok & ~o).sum(), 1) for o in ok_k]) * 100
                conc_err = np.mean([((~o) & ce).sum() / max((~o).sum(), 1) for o in ok_k]) * 100
                tau = zk[0]["tau20"]
                f0 = np.mean([f1(ce, ~o) for o in ok_k]); f20 = np.mean([f1(hem < tau, ~o) for o in ok_k])
                pc, pp = reg(ok_k); zc, zp = reg(ok_z); cc, cp = reg(ok_c) if ok_c else (np.nan, np.nan)
                print(f"{ds + ' ' + su:15s} {c['pi_neg_train'] * 100:4.1f}% | "
                      f"{pk[b]['per_idx'] and (pok[ce].mean() * 100):5.1f}/{pok[~ce].mean() * 100:5.1f} | "
                      f"{pc:5.1f}/{pp:5.1f} {zc:5.1f}/{zp:5.1f} {cc:5.1f}/{cp:5.1f} | {inacc_ce:5.1f}/{lost_ce:5.1f} | "
                      f"{conc_lost:5.1f}/{conc_inac:5.1f}/{conc_err:5.1f} | {f0:.3f}/{f20:.3f}", flush=True)
                agg.append(dict(enc=enc, ds=ds, nq=nq, nc=nc, pi=c["pi_neg_train"], ce_frac=ce.mean(),
                                pqc_ce=pc, pqc_pe=pp, zz_ce=zc, zz_pe=zp, cls_ce=cc, cls_pe=cp,
                                psi_ce=pok[ce].mean() * 100, psi_pe=pok[~ce].mean() * 100,
                                conc_lost=conc_lost, conc_inac=conc_inac, conc_err=conc_err, f0=f0, f20=f20))

print("\n######## averages ########")
print(f"{'group':22s} {'n':>2s} {'pi':>5s} {'CE%':>5s} | {'PQC CE/PE':>11s} {'cls CE/PE':>11s} {'adv CE/PE':>11s} |"
      f" {'conc lost/inac/err':>18s} | {'F1 0/20':>11s}")
for enc in ("amplitude", "angle"):
    for ds in ("mnist", "fmnist", "cifar10", "mnist+fmnist"):
        g = [a for a in agg if a["enc"] == enc and (a["ds"] == ds or (ds == "mnist+fmnist" and a["ds"] != "cifar10"))]
        if not g:
            continue
        m = lambda k: np.nanmean([a[k] for a in g])
        print(f"{enc + ' ' + ds:22s} {len(g):2d} {m('pi') * 100:4.1f}% {m('ce_frac') * 100:4.1f}% | "
              f"{m('pqc_ce'):5.1f}/{m('pqc_pe'):5.1f} {m('cls_ce'):5.1f}/{m('cls_pe'):5.1f} "
              f"{m('pqc_ce') - m('cls_ce'):+5.1f}/{m('pqc_pe') - m('cls_pe'):+5.1f} | "
              f"{m('conc_lost'):5.1f}/{m('conc_inac'):5.1f}/{m('conc_err'):5.1f} | {m('f0'):.3f}/{m('f20'):.3f}")
(E / "phase3_summary.json").write_text(json.dumps(agg, indent=1, default=float))
