import copy
import numpy as np
import torch
import torch.nn.functional as F

from config import LR
from student import QuantumStudent
from torchsim import from_pennylane
from hem import hem_split, arrays


def student_kwargs(enc):
    return {"reducer_type": "angle", "encoding": "angle"} if enc == "angle" else {}


def pool_reducer(enc, nq, nc):
    return QuantumStudent(nq, 1, nc, **student_kwargs(enc)).reducer


def make_model(enc, nq, nc, seed, L=4, readout="z_k", reducer=None):
    torch.manual_seed(seed)
    pl = QuantumStudent(nq, L, nc, readout=readout, **student_kwargs(enc))
    model = from_pennylane(pl, enc, readout)
    if reducer is not None:
        model.reducer = reducer
    return model


def set_trainable(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


def train(model, loader, epochs, device, snaps=()):
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    losses, saved = [], {}
    for ep in range(1, epochs + 1):
        model.train(); tot = n = 0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); loss = F.cross_entropy(model(x), y); loss.backward(); opt.step()
            tot += loss.item(); n += 1
        losses.append(tot / max(n, 1))
        if ep in snaps:
            saved[ep] = copy.deepcopy(model.state_dict())
    return losses, saved


def accuracy(model, loader, device):
    model.eval(); ok = n = 0
    with torch.no_grad():
        for x, y, _ in loader:
            ok += (model(x.to(device)).argmax(1).cpu() == y).sum().item(); n += len(y)
    return ok / n


def predictions(model, loader, device):
    model.eval(); P, Y = {}, {}
    with torch.no_grad():
        for x, y, idx in loader:
            pr = model(x.to(device)).argmax(1).cpu()
            for i in range(len(idx)):
                P[int(idx[i])] = int(pr[i]); Y[int(idx[i])] = int(y[i])
    return P, Y


def evaluate(model, enc, nq, nc, tl, te, device, hem_ref):
    P, Y = predictions(model, te, device)
    ids = sorted(P)
    pred = np.array([P[i] for i in ids]); ok = pred == np.array([Y[i] for i in ids])
    red = copy.deepcopy(model.reducer).cpu().eval()
    _, hem_own = hem_split(red, enc, nq, nc, tl, te)
    ce_ref = np.array([hem_ref[i] < 0 for i in ids]); ce_own = np.array([hem_own[i] < 0 for i in ids])
    fr = lambda m: float(ok[m].mean()) if m.any() else float("nan")
    return dict(acc=float(ok.mean()), acc_ce_orig=fr(ce_ref), acc_pe_orig=fr(~ce_ref), ce_size_orig=float(ce_ref.mean()),
                ce_size_own=float(ce_own.mean()), acc_ce_own=fr(ce_own), acc_pe_own=fr(~ce_own),
                per_idx={int(i): int(p) for i, p in zip(ids, pred)}, hem_test_own={int(i): float(hem_own[i]) for i in ids})
