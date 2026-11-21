import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hcmr"))
from config import (RESULTS, SETUPS, CLASS_ORDER, N_TRAIN, N_VAL, N_TEST, DATA_SEED, BATCH, T_EPOCHS, FIXED_EPOCHS,
                    tag, parse_seeds)

CONDS = [("z_k", 2), ("z_k", 4), ("z_k", 8), ("z_k", 16), ("z_zz", 4)]
OUT = RESULTS / "layers"


def out_path(enc, ds, nq, nc, ro, L, seed):
    return OUT / enc / ds / f"{nq}q{nc}c" / f"{ro}_L{L}" / f"seed_{seed}.json"


def run(task):
    try:
        return _run(*task)
    except Exception as e:
        import traceback
        return f"FAILED {task[:4]}: {type(e).__name__}: {e}\n{traceback.format_exc()}"


def _run(setup, ro, L, seed, device):
    import numpy as np, torch
    torch.set_num_threads(1)
    from data import load_data
    from hem import hem_split
    from qnn import make_model, pool_reducer, set_trainable, train, predictions
    enc, ds, nq, nc = setup
    outp = out_path(enc, ds, nq, nc, ro, L, seed)
    if outp.exists():
        return f"{outp}: cached"
    t0 = time.perf_counter()
    tl, vl, te, *_ = load_data(ds, N_TRAIN, N_VAL, N_TEST, BATCH, DATA_SEED, CLASS_ORDER[ds][:nc])
    model = make_model(enc, nq, nc, seed, L=L, readout=ro).to(device)
    set_trainable(model.reducer, False)
    c1, _ = train(model, tl, FIXED_EPOCHS, device); c2, _ = train(model, tl, T_EPOCHS - FIXED_EPOCHS, device)
    P, Y = predictions(model, te, device)
    ids = sorted(P); ok = np.array([P[i] == Y[i] for i in ids])
    htr, hem = hem_split(pool_reducer(enc, nq, nc), enc, nq, nc, tl, te)
    ce = np.array([hem[i] < 0 for i in ids])
    fr = lambda m: float(ok[m].mean()) if m.any() else float("nan")
    res = dict(encoding=enc, ds=ds, nq=nq, nc=nc, readout=ro, L=L, seed=seed, classes=list(CLASS_ORDER[ds][:nc]),
               schedule=f"{FIXED_EPOCHS}+{T_EPOCHS - FIXED_EPOCHS}", n_obs=model.n_meas,
               n_params=sum(p.numel() for p in model.qlayer.parameters()) + sum(p.numel() for p in model.head.parameters()),
               pi_neg_train=float(np.mean([v < 0 for v in htr.values()])), acc=float(ok.mean()), acc_ce=fr(ce),
               acc_pe=fr(~ce), test_ce_frac=float(ce.mean()), train_loss_curve=c1 + c2, elapsed=time.perf_counter() - t0,
               per_idx={int(i): int(P[i]) for i in ids}, y_test={int(i): int(Y[i]) for i in ids},
               hem_test={int(i): float(v) for i, v in hem.items()})
    outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, outp.with_suffix(".pt"))
    outp.write_text(json.dumps(res))
    return f"{tag(*setup):24s} {ro}_L{L:<2d} s{seed}: acc {res['acc'] * 100:5.1f} | CE {res['acc_ce'] * 100:5.1f} ({res['elapsed']:.0f}s)"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--qubits", default="4,6,8,10"); ap.add_argument("--setups", default="")
    ap.add_argument("--conds", default=",".join(f"{r}_L{L}" for r, L in CONDS), help="e.g. z_k_L4,z_zz_L4")
    ap.add_argument("--seeds", default="", help="comma-separated; empty = random seeds")
    a = ap.parse_args()
    only = set(a.setups.split(",")) if a.setups else None
    conds = [(c.rsplit("_L", 1)[0], int(c.rsplit("_L", 1)[1])) for c in a.conds.split(",")]
    qs = {int(q) for q in a.qubits.split(",")}; seeds = parse_seeds(a.seeds)
    tasks = [(s, ro, L, sd, a.device) for s in SETUPS if s[2] in qs and (not only or tag(*s) in only)
             for ro, L in conds for sd in seeds]
    tasks.sort(key=lambda t: -(2 ** t[0][2]) * t[0][3] * t[2])
    print(f"layer runs on {a.device}: {len(tasks)}, {a.workers} workers, seeds {seeds}", flush=True)
    t0 = time.perf_counter()
    with Pool(a.workers) as pool:
        for i, msg in enumerate(pool.imap_unordered(run, tasks), 1):
            print(f"[{i}/{len(tasks)} {(time.perf_counter() - t0) / 60:.0f}min] {msg}", flush=True)
