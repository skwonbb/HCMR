import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hcmr"))
from config import RESULTS, SETUPS, CLASS_ORDER, N_TRAIN, N_VAL, N_TEST, DATA_SEED, BATCH

OUT = {"amplitude": RESULTS / "ceiling", "angle": RESULTS / "ceiling_angle"}


def run(setup):
    try:
        return _run(setup)
    except Exception as e:
        import traceback
        return f"FAILED {setup}: {type(e).__name__}: {e}\n{traceback.format_exc()}"


def _run(setup):
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np, torch
    torch.set_num_threads(1)
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from data import load_data
    from hem import states, hem_of
    from qnn import pool_reducer
    enc, ds, nq, nc = setup
    p = OUT[enc] / ds / f"{nq}q{nc}c.json"
    if p.exists():
        return f"{setup}: cached"
    t0 = time.perf_counter()
    loaders = dict(zip(("tr", "va", "te"), load_data(ds, N_TRAIN, N_VAL, N_TEST, BATCH, DATA_SEED, CLASS_ORDER[ds][:nc])[:3]))
    red = pool_reducer(enc, nq, nc)

    def arrays(ld):
        X, P, Y, I = [], [], [], []
        with torch.no_grad():
            for x, y, idx in ld:
                X.append(x.flatten(1)); P.append(red(x).double()); Y.append(y); I.append(idx)
        I = torch.cat(I); o = torch.argsort(I)
        return torch.cat(X)[o], torch.cat(P)[o], torch.cat(Y)[o], I[o]

    A = {k: arrays(ld) for k, ld in loaders.items()}
    psi = {k: states(A[k][1], enc, nq) for k in A}
    htr, means = hem_of(psi["tr"], A["tr"][2], nc)
    hte, _ = hem_of(psi["te"], A["te"][2], nc, means)
    D = {}
    for k in A:
        ps = psi[k].numpy()
        fid = torch.einsum("ni,cij,nj->nc", psi[k], means, psi[k]).numpy()
        f = dict(raw=A[k][0].numpy(), pool=A[k][1].numpy(), psi=ps, fid=fid)
        if nq == 4:
            iu = np.triu_indices(ps.shape[1]); f["psi2"] = np.einsum("ni,nj->nij", ps, ps)[:, iu[0], iu[1]]
        D[k] = (f, A[k][2].numpy(), A[k][3].numpy())

    def probe(key):
        cands = [make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)) for C in (0.1, 1, 10, 100)]
        cands += [make_pipeline(StandardScaler(), SVC(C=C, gamma="scale")) for C in (1, 10, 100)]
        cands += [make_pipeline(StandardScaler(), MLPClassifier((256,), alpha=a, max_iter=600, random_state=0))
                  for a in (1e-4, 1e-2)]
        best, bs = None, -1.0
        for m in cands:
            s = m.fit(D["tr"][0][key], D["tr"][1]).score(D["va"][0][key], D["va"][1])
            if s > bs:
                best, bs = m, s
        return best, bs

    yte, ite = D["te"][1], D["te"][2]
    hem = dict(zip(ite.tolist(), hte.tolist())); ce = np.array([hem[int(i)] < 0 for i in ite])
    res = dict(encoding=enc, ds=ds, nq=nq, nc=nc, ce_rule="HEM<0", pi_neg_train=float((htr < 0).float().mean()),
               hem_test={int(i): float(hem[int(i)]) for i in ite}, y_test={int(i): int(y) for i, y in zip(ite, yte)}, probes={})
    for key in D["tr"][0]:
        m, val = probe(key)
        pred = m.predict(D["te"][0][key]); ok = pred == yte
        res["probes"][key] = dict(model=type(m[-1]).__name__, val_acc=val, acc=float(ok.mean()), acc_ce=float(ok[ce].mean()),
                                  acc_pe=float(ok[~ce].mean()), per_idx={int(i): int(q) for i, q in zip(ite, pred)})
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res))
    return f"{setup} psi CE {res['probes']['psi']['acc_ce'] * 100:.1f} ({time.perf_counter() - t0:.0f}s)"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--setups", default="")
    a = ap.parse_args()
    only = set(a.setups.split(",")) if a.setups else None
    todo = sorted([s for s in SETUPS if not only or "_".join(map(str, s[:2])) + f"_{s[2]}q{s[3]}c" in only],
                  key=lambda s: -(2 ** s[2]) * s[3])
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(run, todo):
            print(msg, flush=True)
