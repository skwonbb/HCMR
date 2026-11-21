import argparse, copy, json, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hcmr"))
from config import (RESULTS, SETUPS, CLASS_ORDER, N_TRAIN, N_VAL, N_TEST, DATA_SEED, BATCH, T_EPOCHS, FIXED_EPOCHS,
                    J_GRID, MUS, LAMS, MARGIN_GRID, tag, parse_seeds)

FIXED = ("pool", "pca", "lloyd", "nqe")
SELECT_J = ("hcmr", "hcmr_cnn", "wo_mu", "wo_lambda", "wo_pretrain", "cnn_wo_pretrain")
METHODS = FIXED + ("e2e",) + SELECT_J
FITTED = ("pca", "lloyd", "nqe", "hcmr", "hcmr_cnn", "wo_mu", "wo_lambda")
GRID = {"hcmr": MARGIN_GRID, "hcmr_cnn": MARGIN_GRID,
        "wo_mu": [(0.0, l) for l in LAMS if l > 0], "wo_lambda": [(m, 0.0) for m in MUS if m > 0]}
ENCODER_SRC = {"wo_pretrain": "pca"}
CNN = ("nqe", "hcmr_cnn")


def stage_fit(out, device, only):
    import torch
    from data import load_data
    from hem import arrays
    from encoders import fit_linear, fit_cnn
    for s in SETUPS:
        if only and tag(*s) not in only:
            continue
        enc, ds, nq, nc = s
        todo = [k for k in FITTED if not (out / tag(*s) / k / "encoder.pt").exists()]
        if not todo:
            continue
        tl, vl, *_ = load_data(ds, N_TRAIN, N_VAL, N_TEST, BATCH, DATA_SEED, CLASS_ORDER[ds][:nc])
        Xtr, ytr, _ = arrays(tl); Xva, yva, _ = arrays(vl)
        for k in todo:
            t0 = time.perf_counter(); cands = []
            for mu, lam in GRID.get(k, [(0.0, 0.0)]):
                if k in CNN:
                    obj, vce = fit_cnn(k, Xtr, ytr, Xva, yva, enc, nq, nc, ds, device, mu, lam)
                else:
                    W, b, vce = fit_linear(k if k in ("pca", "lloyd") else "hcmr", Xtr, ytr, Xva, yva, enc, nq, nc,
                                           device, mu, lam)
                    obj = (W, b)
                cands.append((vce, mu, lam, obj))
            vce, mu, lam, obj = min(cands, key=lambda c: (c[0], c[1], c[2]))
            p = out / tag(*s) / k / "encoder.pt"; p.parent.mkdir(parents=True, exist_ok=True)
            torch.save(obj, p)
            if k in GRID:
                (p.parent / "margin.json").write_text(json.dumps(dict(
                    mu=mu, lam=lam, val_ce=vce, grid=[dict(mu=c[1], lam=c[2], val_ce=c[0]) for c in cands])))
            print(f"[fit] {tag(*s):24s} {k:9s} {time.perf_counter() - t0:6.1f}s"
                  + (f" mu={mu} lam={lam} val CE {vce * 100:.1f}%" if k in GRID else ""), flush=True)


def run(task):
    try:
        return _run(*task)
    except Exception as e:
        import traceback
        return f"FAILED {task[:4]}: {type(e).__name__}: {e}\n{traceback.format_exc()}"


def _run(setup, method, seed, device, opt):
    import torch
    torch.set_num_threads(1)
    from data import load_data
    from hem import arrays, hem_split
    from encoders import Linear, make_cnn, pca_encoder
    from qnn import make_model, pool_reducer, set_trainable, train, accuracy, evaluate
    enc, ds, nq, nc = setup
    out, enc_dir = Path(opt["out"]), Path(opt["enc_dir"])
    od = out / tag(*setup) / method; od.mkdir(parents=True, exist_ok=True)
    if (od / f"seed_{seed}.json").exists():
        return f"{tag(*setup)} {method} s{seed}: cached"
    t0 = time.perf_counter()
    tl, vl, te, *_ = load_data(ds, N_TRAIN, N_VAL, N_TEST, BATCH, DATA_SEED, CLASS_ORDER[ds][:nc])
    if method == "pool":
        red = pool_reducer(enc, nq, nc)
    elif method == "e2e":
        Xtr, ytr, _ = arrays(tl); W, b = pca_encoder(Xtr.double(), enc, nq); red = Linear(W, b, enc)
    elif method in CNN:
        red = make_cnn(enc, nq, ds); red.load_state_dict(torch.load(enc_dir / tag(*setup) / method / "encoder.pt"))
    elif method == "cnn_wo_pretrain":
        torch.manual_seed(0); red = make_cnn(enc, nq, ds)
    else:
        W, b = torch.load(enc_dir / tag(*setup) / ENCODER_SRC.get(method, method) / "encoder.pt"); red = Linear(W, b, enc)
    model = make_model(enc, nq, nc, seed, reducer=red).to(device)
    mfile = enc_dir / tag(*setup) / method / "margin.json"
    margin = json.loads(mfile.read_text()) if mfile.exists() else None
    T = T_EPOCHS
    extra, results = {}, []

    if method in SELECT_J and opt["window_starts"]:
        starts, wl = opt["window_starts"], opt["window_len"]
        s0 = copy.deepcopy(model.state_dict()); set_trainable(model.reducer, False)
        fz_curve, fz = train(model, tl, max(starts), device, snaps={s for s in starts if s > 0})
        win = dict(window_len=wl, window_val_acc={}, window_test_acc={})
        for s in starts:
            model.load_state_dict(s0 if s == 0 else fz[s])
            set_trainable(model.reducer, True); c2, _ = train(model, tl, wl, device)
            set_trainable(model.reducer, False); c3, _ = train(model, tl, T - s - wl, device)
            curve = list(fz_curve[:s]) + c2 + c3
            win["window_val_acc"][str(s)] = accuracy(model, vl, device); win["window_test_acc"][str(s)] = accuracy(model, te, device)
        results.append((method, copy.deepcopy(model.state_dict()), curve, win))
    elif method in SELECT_J:
        jg = opt["j_grid"]
        s0 = copy.deepcopy(model.state_dict()); set_trainable(model.reducer, False)
        fz_curve, fz = train(model, tl, T, device, snaps={T - j for j in jg if 0 < j < T} | {T})
        cand = {}
        for j in jg:
            if j == 0:
                model.load_state_dict(fz[T]); cj = list(fz_curve)
            else:
                model.load_state_dict(s0 if j == T else fz[T - j])
                set_trainable(model.reducer, True); c2, _ = train(model, tl, j, device); set_trainable(model.reducer, False)
                cj = list(fz_curve[:T - j]) + c2
            cand[j] = (accuracy(model, vl, device), copy.deepcopy(model.state_dict()), cj, accuracy(model, te, device))
        J = max(jg, key=lambda j: (cand[j][0], -j))
        info = dict(joint_epochs=J, joint_val_acc={str(j): cand[j][0] for j in jg}, joint_test_acc={str(j): cand[j][3] for j in jg})
        if method == "hcmr" and 0 in cand and not opt["is_sweep"]:
            results.append(("hcmr_frozen", cand[0][1], cand[0][2], dict(joint_epochs=0)))
        results.append((method, cand[J][1], cand[J][2], info))
    elif method == "e2e":
        curve, _ = train(model, tl, T, device)
        results.append((method, copy.deepcopy(model.state_dict()), curve, {}))
    else:
        set_trainable(model.reducer, False)
        c1, _ = train(model, tl, FIXED_EPOCHS, device); c2, _ = train(model, tl, T - FIXED_EPOCHS, device)
        results.append((method, copy.deepcopy(model.state_dict()), c1 + c2, dict(schedule=f"{FIXED_EPOCHS}+{T - FIXED_EPOCHS}")))

    _, hem_ref = hem_split(pool_reducer(enc, nq, nc), enc, nq, nc, tl, te)
    elapsed = time.perf_counter() - t0
    for name, state, curve, info in results:
        model.load_state_dict(state); model.to(device)
        res = dict(encoding=enc, ds=ds, nq=nq, nc=nc, method=name, seed=seed, device=device, elapsed=elapsed,
                   n_params_pqc=sum(p.numel() for p in model.qlayer.parameters()) + sum(p.numel() for p in model.head.parameters()),
                   margin={k: margin[k] for k in ("mu", "lam")} if margin else None, train_loss_curve=curve, **info)
        res.update(evaluate(model, enc, nq, nc, tl, te, device, hem_ref))
        d = out / tag(*setup) / name; d.mkdir(parents=True, exist_ok=True)
        (d / f"seed_{seed}.json").write_text(json.dumps(res))
        if opt["save_models"]:
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, d / f"seed_{seed}.pt")
    r = json.loads((od / f"seed_{seed}.json").read_text())
    return (f"{tag(*setup):24s} {method:15s} s{seed}: acc {r['acc'] * 100:5.1f} | CE {r['acc_ce_orig'] * 100:5.1f} "
            f"| PE {r['acc_pe_orig'] * 100:5.1f} | CE size {r['ce_size_own'] * 100:5.1f}%"
            + (f" J={r['joint_epochs']}" if "joint_epochs" in r else "") + f" ({elapsed:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("fit", "train"), required=True)
    ap.add_argument("--device", default="cpu"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--qubits", default="4,6,8,10"); ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--setups", default="", help="comma-separated, e.g. amplitude_mnist_4q4c (default: all 48)")
    ap.add_argument("--seeds", default="", help="comma-separated; empty = random seeds")
    ap.add_argument("--out", default=str(RESULTS / "full")); ap.add_argument("--enc-dir", default="")
    ap.add_argument("--j-grid", default=""); ap.add_argument("--window-starts", default="")
    ap.add_argument("--window-len", type=int, default=10); ap.add_argument("--save-models", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); only = set(a.setups.split(",")) if a.setups else None
    if a.stage == "fit":
        stage_fit(out, a.device, only); sys.exit()
    opt = dict(out=str(out), enc_dir=a.enc_dir or str(out), save_models=a.save_models, window_len=a.window_len,
               j_grid=tuple(sorted(int(j) for j in a.j_grid.split(","))) if a.j_grid else J_GRID,
               window_starts=tuple(sorted(int(s) for s in a.window_starts.split(","))) if a.window_starts else (),
               is_sweep=bool(a.j_grid or a.window_starts))
    qs = {int(q) for q in a.qubits.split(",")}; seeds = parse_seeds(a.seeds)
    tasks = [(s, m, sd, a.device, opt) for s in SETUPS if s[2] in qs and (not only or tag(*s) in only)
             for m in a.methods.split(",") for sd in seeds]
    tasks.sort(key=lambda t: -(2 ** t[0][2]) * t[0][3])
    print(f"train on {a.device}: {len(tasks)} runs, {a.workers} workers, seeds {seeds}", flush=True)
    t0 = time.perf_counter()
    with Pool(a.workers) as pool:
        for i, msg in enumerate(pool.imap_unordered(run, tasks), 1):
            print(f"[{i}/{len(tasks)} {(time.perf_counter() - t0) / 60:.0f}min] {msg}", flush=True)
