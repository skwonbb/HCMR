import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse, copy, json, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hcmr"))
from config import RESULTS, CLASS_ORDER, N_TRAIN, N_VAL, N_TEST, DATA_SEED, BATCH, LR, T_EPOCHS, tag

OUT = RESULTS / "noise" / "train"
CAL_FILE = OUT / "calibration_fake_fez.json"
J_LIST = (0, 10, 60)
SETUPS = [(e, d, q, 4) for e in ("amplitude", "angle") for d in ("mnist", "fmnist") for q in (4, 8)]
METHODS = ("hcmr", "hcmr_cnn")


def main_seeds(setup, method):
    return sorted(int(p.stem.split("_")[1]) for p in (RESULTS / "full" / tag(*setup) / method).glob("seed_*.json"))


def load_model(enc, ds, nq, nc, method, seed, device):
    import torch, torch.nn as nn
    from encoders import Linear, make_cnn
    from student import QuantumStudent
    from qnn import student_kwargs
    d = RESULTS / "full" / tag(enc, ds, nq, nc) / method / "encoder.pt"
    if method == "hcmr_cnn":
        red = make_cnn(enc, nq, ds); red.load_state_dict(torch.load(d))
    else:
        W, b = torch.load(d); red = Linear(W, b, enc)
    torch.manual_seed(seed)
    pl = QuantumStudent(nq, 4, nc, **student_kwargs(enc))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.reducer = red
            self.weights = nn.Parameter(pl.qlayer.weights.detach().clone().float())
            self.head = nn.Linear(pl.head.in_features, nc); self.head.load_state_dict(pl.head.state_dict())
    return Model().to(device), pl.head.in_features


def run(task):
    try:
        return _run(*task)
    except Exception as e:
        import traceback
        return f"FAILED {task[:4]}: {type(e).__name__}: {e}\n{traceback.format_exc()}"


def _run(setup, method, seed, cond, device):
    import numpy as np, torch, torch.nn.functional as F
    torch.set_num_threads(1)
    from data import load_data
    from densitymatrix import DMQNN, SHOTS, load_calibration
    from hem import states, hem_of, arrays
    from qnn import pool_reducer
    enc, ds, nq, nc = setup
    od = OUT / cond / tag(*setup) / method; outp = od / f"seed_{seed}.json"
    if outp.exists():
        return f"{cond} {tag(*setup)} {method} s{seed}: cached"
    t0 = time.perf_counter()
    cal = load_calibration(CAL_FILE)
    dn, sh = cond in ("device", "both"), (SHOTS if cond in ("shots", "both") else 0)
    tl, vl, te, *_ = load_data(ds, N_TRAIN, N_VAL, N_TEST, BATCH, DATA_SEED, CLASS_ORDER[ds][:nc])
    model, n_meas = load_model(enc, ds, nq, nc, method, seed, device)
    sim = DMQNN(nq, 4, n_meas, enc, dn, sh, device, cal)
    clean = DMQNN(nq, 4, n_meas, enc, False, 0, device, cal)
    enc0 = {k: v.detach().clone() for k, v in model.reducer.state_dict().items()}

    def enc_trainable(flag):
        for p_ in model.reducer.parameters():
            p_.requires_grad_(flag)

    def logits(x, s, train_enc):
        e = s.forward(model.reducer(x), model.weights, train_enc)
        return model.head(e), e

    def train(epochs, snaps=()):
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
        train_enc = any(p_.requires_grad for p_ in model.reducer.parameters())
        losses, saved = [], {}
        for ep in range(1, epochs + 1):
            tot = k = 0
            for x, y, _ in tl:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(); out, e = logits(x, sim, train_enc); loss = F.cross_entropy(out, y); loss.backward()
                if sh:
                    sim.pqc_grad_noise(model.weights, e)
                opt.step(); tot += loss.item(); k += 1
                sim.g_obs = None; del out, e, loss
            losses.append(tot / max(k, 1))
            if ep in snaps:
                saved[ep] = copy.deepcopy(model.state_dict())
        return losses, saved

    def accuracy(loader, s):
        ok = n_ = 0
        with torch.no_grad():
            for x, y, _ in loader:
                pr = logits(x.to(device), s, False)[0].argmax(1).cpu(); ok += (pr == y).sum().item(); n_ += len(y)
        return ok / n_

    with torch.random.fork_rng(devices=[]):
        Xtr, ytr, _ = arrays(tl, flat=False); Xte, yte, ite = arrays(te, flat=False)
        pool = pool_reducer(enc, nq, nc)
        with torch.no_grad():
            _, mp = hem_of(states(pool(Xtr).double(), enc, nq), ytr, nc)
            hp, _ = hem_of(states(pool(Xte).double(), enc, nq), yte, nc, mp)
    ce = (hp < 0).numpy()

    def evaluate():
        out = {}
        with torch.no_grad():
            for name, s in (("ideal", clean), ("noisy", DMQNN(nq, 4, n_meas, enc, True, 0, device, cal))):
                pr = torch.cat([logits(Xte[i:i + 200].to(device), s, False)[0].argmax(1).cpu() for i in range(0, len(Xte), 200)])
                ok = (pr == yte).numpy()
                out[name] = dict(acc=float(ok.mean()), acc_ce=float(ok[ce].mean()), acc_pe=float(ok[~ce].mean()))
            ptr = states(model.reducer(Xtr.to(device)).double(), enc, nq); hmtr, means = hem_of(ptr, ytr.to(device), nc)
            pte = states(model.reducer(Xte.to(device)).double(), enc, nq); hmte, _ = hem_of(pte, yte.to(device), nc, means)
            num = sum(((v - enc0[k]) ** 2).sum() for k, v in model.reducer.state_dict().items())
            den = sum((v ** 2).sum() for v in enc0.values())
        val = accuracy(vl, clean); accuracy(te, clean)
        out.update(val_acc_ideal=val, ce_size_train=float((hmtr < 0).double().mean()),
                   ce_size_test=float((hmte < 0).double().mean()), hem_test_mean=float(hmte.mean()),
                   encoder_drift=float((num / den).sqrt()))
        return out

    s0 = copy.deepcopy(model.state_dict()); enc_trainable(False)
    T = T_EPOCHS
    fz_curve, fz = train(T, snaps={T - j for j in J_LIST if 0 < j < T} | {T})
    res = dict(encoding=enc, ds=ds, nq=nq, nc=nc, method=method, seed=seed, condition=cond, shots=sh, calibration=cal, J={})
    saved = {}
    for j in J_LIST:
        if j == 0:
            model.load_state_dict(fz[T]); curve = list(fz_curve)
        else:
            model.load_state_dict(s0 if j == T else fz[T - j]); enc_trainable(True)
            c2, _ = train(j); enc_trainable(False); curve = list(fz_curve[:T - j]) + c2
        res["J"][str(j)] = dict(train_loss_curve=curve, **evaluate()); saved[str(j)] = copy.deepcopy(model.state_dict())
    res["elapsed"] = time.perf_counter() - t0
    od.mkdir(parents=True, exist_ok=True)
    torch.save({k: {n_: v.cpu() for n_, v in sd.items()} for k, sd in saved.items()}, od / f"seed_{seed}.pt")
    outp.write_text(json.dumps(res))
    s = " | ".join(f"J={j}: {res['J'][str(j)]['ideal']['acc'] * 100:5.1f}/{res['J'][str(j)]['noisy']['acc'] * 100:5.1f}" for j in J_LIST)
    return f"{cond:6s} {tag(*setup):22s} {method:8s} s{seed}: (ideal/noisy inference) {s} ({res['elapsed']:.0f}s)"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="none,shots,device,both"); ap.add_argument("--qubits", default="4,8")
    ap.add_argument("--methods", default=",".join(METHODS)); ap.add_argument("--setups", default="")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    only = set(a.setups.split(",")) if a.setups else None
    qs = {int(q) for q in a.qubits.split(",")}
    tasks = [(s, m, sd, c, a.device) for c in a.conditions.split(",") for s in SETUPS
             if s[2] in qs and (not only or tag(*s) in only) for m in a.methods.split(",") for sd in main_seeds(s, m)]
    tasks.sort(key=lambda t: -(t[0][2] * (2 if t[1] == "hcmr_cnn" else 1)))
    print(f"noisy training: {len(tasks)} runs, {a.workers} workers on {a.device}", flush=True)
    t0 = time.perf_counter()
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for i, msg in enumerate(pool.imap_unordered(run, tasks), 1):
            print(f"[{i}/{len(tasks)} {(time.perf_counter() - t0) / 60:.0f}min] {msg}", flush=True)
