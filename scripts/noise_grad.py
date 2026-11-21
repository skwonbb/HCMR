import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, copy, json, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hcmr"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS, CLASS_ORDER, N_TRAIN, N_VAL, N_TEST, DATA_SEED, BATCH, LR, T_EPOCHS, tag
from noise_train import SETUPS, METHODS, CAL_FILE, OUT as TRAIN, main_seeds, load_model

OUT = RESULTS / "noise" / "grad"
K_BATCH = 4
CONDS = ("shots", "device", "both")


def run(task):
    try:
        return _run(*task)
    except Exception as e:
        import traceback
        return f"FAILED {task}: {type(e).__name__}: {e}\n{traceback.format_exc()}"


def _run(setup, method, seed):
    import numpy as np, torch, torch.nn.functional as F
    torch.set_num_threads(1)
    from data import load_data
    from densitymatrix import DMQNN, SHOTS, load_calibration
    from hem import arrays
    enc, ds, nq, nc = setup
    tagd = tag(*setup)
    outp = OUT / tagd / method / f"seed_{seed}.json"
    if outp.exists():
        return f"{tagd} {method} s{seed}: cached"
    if not (TRAIN / "none" / tagd / method / f"seed_{seed}.pt").exists():
        return f"{tagd} {method} s{seed}: skipped (run scripts/noise_train.py --conditions none first)"
    t0 = time.perf_counter(); device = "cpu"
    cal = load_calibration(CAL_FILE)
    tl, vl, te, *_ = load_data(ds, N_TRAIN, N_VAL, N_TEST, BATCH, DATA_SEED, CLASS_ORDER[ds][:nc])
    model, n_meas = load_model(enc, ds, nq, nc, method, seed, device)
    clean = DMQNN(nq, 4, n_meas, enc, False, 0, device, cal)
    noisy = {"shots": DMQNN(nq, 4, n_meas, enc, False, SHOTS, device, cal),
             "device": DMQNN(nq, 4, n_meas, enc, True, 0, device, cal),
             "both": DMQNN(nq, 4, n_meas, enc, True, SHOTS, device, cal)}

    def enc_trainable(flag):
        for p_ in model.reducer.parameters():
            p_.requires_grad_(flag)

    def logits(x, s, train_enc):
        e = s.forward(model.reducer(x), model.weights, train_enc)
        return model.head(e), e

    with torch.random.fork_rng(devices=[]):
        Xtr, ytr, _ = arrays(tl, flat=False)
    batches = [(Xtr[k * BATCH:(k + 1) * BATCH], ytr[k * BATCH:(k + 1) * BATCH]) for k in range(K_BATCH)]

    def grad_of(s, x, y, train_enc):
        model.zero_grad(set_to_none=True)
        out, e = logits(x, s, train_enc); loss = F.cross_entropy(out, y); loss.backward()
        if s.shots:
            s.pqc_grad_noise(model.weights, e)
        gp = model.weights.grad.detach().flatten().clone()
        ge = (torch.cat([p_.grad.detach().flatten() for p_ in model.reducer.parameters()]).clone() if train_enc else None)
        s.g_obs = None; model.zero_grad(set_to_none=True)
        return gp, ge

    def measure(epoch, train_enc):
        parts = ("pqc", "enc") if train_enc else ("pqc",)
        rec = {c: {f"{k}_{p_}": [] for k in ("cos", "ratio") for p_ in parts} for c in CONDS}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(10_000 + 100 * seed + epoch)
            for x, y in batches:
                g0 = dict(zip(("pqc", "enc"), grad_of(clean, x, y, train_enc)))
                for c in CONDS:
                    g = dict(zip(("pqc", "enc"), grad_of(noisy[c], x, y, train_enc)))
                    for p_ in parts:
                        rec[c][f"cos_{p_}"].append(float(F.cosine_similarity(g[p_], g0[p_], dim=0)))
                        rec[c][f"ratio_{p_}"].append(float(g[p_].norm() / g0[p_].norm().clamp_min(1e-30)))
        return dict(epoch=epoch, encoder_trained=train_enc, **rec)

    def train(epochs, first_epoch, measure_until, snaps=()):
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
        train_enc = any(p_.requires_grad for p_ in model.reducer.parameters())
        pts, saved = [], {}
        for k in range(epochs):
            ep = first_epoch + k
            if ep <= measure_until:
                pts.append(measure(ep, train_enc))
            for x, y, _ in tl:
                opt.zero_grad(); out, e = logits(x, clean, train_enc); loss = F.cross_entropy(out, y); loss.backward()
                opt.step(); del out, e, loss
            if k + 1 in snaps:
                saved[k + 1] = copy.deepcopy(model.state_dict())
        return pts, saved

    def loader_passes():
        for _ in vl:
            pass
        for _ in te:
            pass

    T_ = T_EPOCHS
    s0 = copy.deepcopy(model.state_dict()); enc_trainable(False)
    p_frozen, fz = train(T_, 1, T_ - 10, snaps={T_ - 10, T_})
    st0 = copy.deepcopy(fz[T_]); loader_passes()
    model.load_state_dict(fz[T_ - 10]); enc_trainable(True)
    p_joint10, _ = train(10, T_ - 9, T_); enc_trainable(False)
    st10 = copy.deepcopy(model.state_dict()); loader_passes()
    model.load_state_dict(s0); enc_trainable(True)
    p_joint60, _ = train(T_, 1, T_); enc_trainable(False)
    ref = torch.load(TRAIN / "none" / tagd / method / f"seed_{seed}.pt")
    dev = {j: max(float((st[k].cpu() - ref[j][k]).abs().max()) for k in ref[j])
           for j, st in (("0", st0), ("10", st10), ("60", model.state_dict()))}
    res = dict(encoding=enc, ds=ds, nq=nq, nc=nc, method=method, seed=seed, shots=SHOTS, k_batches=K_BATCH,
               batch=BATCH, J10=p_frozen + p_joint10, J60=p_joint60, max_abs_dev_vs_saved_none=dev,
               elapsed=time.perf_counter() - t0)
    outp.parent.mkdir(parents=True, exist_ok=True); outp.write_text(json.dumps(res))
    c = lambda p, k: np.mean(p["both"][k])
    return (f"{tagd:22s} {method:8s} s{seed}: cos(both) pqc {c(p_frozen[0], 'cos_pqc'):.3f} | enc J10 "
            f"{c(p_joint10[-1], 'cos_enc'):.3f} J60 {c(p_joint60[-1], 'cos_enc'):.3f} | dev vs saved "
            f"{dev['0']:.1e}/{dev['10']:.1e}/{dev['60']:.1e} ({res['elapsed']:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--qubits", default="8"); ap.add_argument("--setups", default="")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    only = set(a.setups.split(",")) if a.setups else None
    qs = {int(q) for q in a.qubits.split(",")}
    tasks = [(s, m, sd) for s in SETUPS if s[2] in qs and (not only or tag(*s) in only)
             for m in METHODS for sd in main_seeds(s, m)]
    tasks.sort(key=lambda t: -(t[0][2] * (2 if t[1] == "hcmr_cnn" else 1)))
    print(f"gradient noise: {len(tasks)} runs, {a.workers} workers", flush=True)
    t0 = time.perf_counter()
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for i, msg in enumerate(pool.imap_unordered(run, tasks), 1):
            print(f"[{i}/{len(tasks)} {(time.perf_counter() - t0) / 60:.0f}min] {msg}", flush=True)
