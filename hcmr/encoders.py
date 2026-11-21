import torch
import torch.nn as nn

from config import SIDE, FIT_STEPS, FIT_LR
from hem import states, hem_of, competitor_fidelity


def pca_encoder(Xtr, enc, nq):
    d = 2 ** nq if enc == "amplitude" else nq
    mu = Xtr.mean(0)
    _, _, Vt = torch.linalg.svd(Xtr - mu, full_matrices=False)
    W0 = Vt[:d]
    if len(W0) < d:
        W0 = torch.cat([W0, W0.new_zeros(d - len(W0), W0.shape[1])])
    if enc == "angle":
        sc = ((Xtr - mu) @ W0.T).std(0)
        W0 = W0 / sc[:, None]
    return W0.clone(), -(W0 @ mu)


class Linear(nn.Module):

    def __init__(self, W, b, enc, trainable=True):
        super().__init__()
        W, b = W.float().clone(), b.float().clone()
        if trainable:
            self.W, self.b = nn.Parameter(W), nn.Parameter(b)
        else:
            self.register_buffer("W", W); self.register_buffer("b", b)
        self.enc = enc

    def forward(self, x):
        z = x.flatten(1) @ self.W.T + self.b
        return torch.pi * torch.sigmoid(z) if self.enc == "angle" else z


def make_cnn(enc, nq, ds):
    side, d = SIDE[ds], (2 ** nq if enc == "amplitude" else nq)

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                   nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
                                   nn.Linear(16 * (side // 4) ** 2, d))

        def forward(self, x):
            z = self.f(x.reshape(-1, 1, side, side))
            return torch.pi * torch.sigmoid(z) if enc == "angle" else z
    return CNN()


def _margin_loss(psi, y, nc, mu, lam):
    h, means = hem_of(psi, y, nc)
    return torch.relu(mu + lam * competitor_fidelity(psi, y, means, h) - h).mean()


def _val_ce(reduce, Xt, yt, Xv, yv, enc, nq, nc):
    with torch.no_grad():
        _, mtr = hem_of(states(reduce(Xt), enc, nq), yt, nc)
        hv, _ = hem_of(states(reduce(Xv), enc, nq), yv, nc, mtr)
        return (hv <= 0).float().mean().item()


def fit_linear(kind, Xtr, ytr, Xva, yva, enc, nq, nc, device, mu=0.0, lam=0.0):
    W0, b0 = pca_encoder(Xtr.double(), enc, nq)
    if kind == "pca":
        return W0, b0, float("nan")
    Xt, Xv, yt, yv = Xtr.float().to(device), Xva.float().to(device), ytr.to(device), yva.to(device)
    W = W0.float().to(device).clone().requires_grad_(True); b = b0.float().to(device).clone().requires_grad_(True)
    opt = torch.optim.Adam([W, b], lr=FIT_LR)
    red = lambda X: (torch.pi * torch.sigmoid(X @ W.T + b)) if enc == "angle" else X @ W.T + b
    best, best_ce = (W0.clone(), b0.clone()), _val_ce(red, Xt, yt, Xv, yv, enc, nq, nc)
    for step in range(FIT_STEPS):
        psi = states(red(Xt), enc, nq)
        if kind == "lloyd":
            means = torch.stack([(psi[yt == c].T @ psi[yt == c]) / (yt == c).sum() for c in range(nc)])
            loss = -((means[:, None] - means[None]) ** 2).sum((-1, -2)).mean()
        else:
            loss = _margin_loss(psi, yt, nc, mu, lam)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 20 == 19:
            ce = _val_ce(red, Xt, yt, Xv, yv, enc, nq, nc)
            if ce < best_ce:
                best_ce, best = ce, (W.detach().double().cpu().clone(), b.detach().double().cpu().clone())
    return best[0], best[1], best_ce


def fit_cnn(kind, Xtr, ytr, Xva, yva, enc, nq, nc, ds, device, mu=0.0, lam=0.0, init_seed=0):
    torch.manual_seed(init_seed)
    net = make_cnn(enc, nq, ds).to(device)
    Xt, Xv, yt, yv = Xtr.float().to(device), Xva.float().to(device), ytr.to(device), yva.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    same = (yt[:, None] == yt[None]).float()
    snap = lambda: {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    best, best_ce = snap(), _val_ce(net, Xt, yt, Xv, yv, enc, nq, nc)
    for step in range(FIT_STEPS):
        psi = states(net(Xt), enc, nq)
        loss = (((psi @ psi.T) ** 2 - same) ** 2).mean() if kind == "nqe" else _margin_loss(psi, yt, nc, mu, lam)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 20 == 19:
            ce = _val_ce(net, Xt, yt, Xv, yv, enc, nq, nc)
            if ce < best_ce:
                best_ce, best = ce, snap()
    return best, best_ce
