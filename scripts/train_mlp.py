import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hcmr"))
from config import (RESULTS, SETUPS, CLASS_ORDER, N_TRAIN, N_VAL, N_TEST, DATA_SEED, BATCH, LR, T_EPOCHS, FIXED_EPOCHS,
                    parse_seeds)

SUF = {"amplitude": "", "angle": "_angle"}


def run(task):
    try:
        return _run(*task)
    except Exception as e:
        import traceback
        return f"FAILED {task}: {type(e).__name__}: {e}\n{traceback.format_exc()}"


def _run(enc, ds, nq, nc, seed):
    import torch, torch.nn as nn, torch.nn.functional as F
    torch.set_num_threads(1)
    from data import load_data
    from hem import states, arrays
    from qnn import pool_reducer, make_model
    su = f"{nq}q{nc}c"
    p = RESULTS / f"classical{SUF[enc]}" / ds / su / "mlp_p" / f"seed_{seed}.json"
    if p.exists():
        return f"{enc} {ds} {su} s{seed}: cached"
    t0 = time.perf_counter()
    tl, vl, te, *_ = load_data(ds, N_TRAIN, N_VAL, N_TEST, BATCH, DATA_SEED, CLASS_ORDER[ds][:nc])
    red = pool_reducer(enc, nq, nc)

    def feats(ld):
        X, Y, I = arrays(ld, flat=False)
        with torch.no_grad():
            return states(red(X).double(), enc, nq).float(), Y, I
    Xtr, Ytr, _ = feats(tl); Xte, Yte, Ite = feats(te)
    ceil = json.loads((RESULTS / f"ceiling{SUF[enc]}" / ds / f"{su}.json").read_text())
    ce = torch.tensor([ceil["hem_test"][str(int(i))] < 0 for i in Ite])
    D = Xtr.shape[1]
    pqc_p = sum(q.numel() for q in make_model(enc, nq, nc, 0).parameters())
    h = max(1, round((pqc_p - nc) / (D + 1 + nc)))
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(D, h), nn.ReLU(), nn.Linear(h, nc))
    g = torch.Generator().manual_seed(seed)
    for ep in range(T_EPOCHS):
        if ep in (0, FIXED_EPOCHS):
            opt = torch.optim.Adam(model.parameters(), lr=LR)
        perm = torch.randperm(len(Xtr), generator=g)
        for s in range(0, len(Xtr), BATCH):
            b = perm[s:s + BATCH]
            opt.zero_grad(); F.cross_entropy(model(Xtr[b]), Ytr[b]).backward(); opt.step()
    with torch.no_grad():
        pred = model(Xte).argmax(1)
    ok = pred == Yte
    res = dict(encoding=enc, ds=ds, nq=nq, nc=nc, model="mlp_p", seed=seed, schedule=f"{FIXED_EPOCHS}+{T_EPOCHS - FIXED_EPOCHS}",
               n_params=sum(q.numel() for q in model.parameters()), pqc_params=pqc_p, hidden=h,
               acc=float(ok.float().mean()), acc_ce=float(ok[ce].float().mean()) if ce.any() else float("nan"),
               acc_pe=float(ok[~ce].float().mean()), ce_frac=float(ce.float().mean()),
               per_idx={int(i): int(q) for i, q in zip(Ite, pred)})
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(res))
    return f"{enc} {ds} {su} s{seed}: CE {res['acc_ce'] * 100:.1f} PE {res['acc_pe'] * 100:.1f} ({time.perf_counter() - t0:.0f}s)"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seeds", default="", help="comma-separated; empty = random seeds")
    ap.add_argument("--setups", default="")
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds); only = set(a.setups.split(",")) if a.setups else None
    tasks = [(e, d, q, c, s) for e, d, q, c in SETUPS if not only or f"{e}_{d}_{q}q{c}c" in only for s in seeds]
    print(f"{len(tasks)} tasks, {a.workers} workers, seeds {seeds}", flush=True)
    t0 = time.perf_counter()
    with Pool(a.workers) as pool:
        for i, msg in enumerate(pool.imap_unordered(run, tasks), 1):
            print(f"[{i}/{len(tasks)} {(time.perf_counter() - t0) / 60:.0f}min] {msg}", flush=True)
