import torch


def states(z, enc, nq):
    if enc == "amplitude":
        return z / z.norm(dim=1, keepdim=True).clamp_min(1e-10)
    s = torch.ones(len(z), 1, dtype=z.dtype, device=z.device)
    for i in range(nq):
        one = torch.stack([torch.cos(z[:, i] / 2), torch.sin(z[:, i] / 2)], 1)
        s = torch.einsum("nb,nk->nbk", s, one).reshape(len(z), -1)
    return s


def class_means(psi, y, nc):
    return torch.stack([(psi[y == c].T @ psi[y == c]) / (y == c).sum() for c in range(nc)])


def hem_of(psi, y, nc, means=None):
    if means is None:
        means = class_means(psi, y, nc)
    F = torch.einsum("ni,cij,nj->nc", psi, means, psi)
    own = F.gather(1, y[:, None]).squeeze(1)
    other = F.masked_fill(torch.nn.functional.one_hot(y, nc).bool(), -1e9).max(1).values
    return own - other, means


def competitor_fidelity(psi, y, means, h):
    F = torch.einsum("ni,cij,nj->nc", psi, means, psi)
    return F.gather(1, y[:, None]).squeeze(1) - h


def arrays(loader, flat=True):
    X, Y, I = [], [], []
    for x, y, i in loader:
        X.append(x.flatten(1) if flat else x); Y.append(y); I.append(i)
    X, Y, I = torch.cat(X), torch.cat(Y), torch.cat(I)
    o = torch.argsort(I)
    return X[o], Y[o], I[o]


def hem_split(reducer, enc, nq, nc, train_loader, other_loader):
    Xtr, ytr, itr = arrays(train_loader, flat=False); Xo, yo, io = arrays(other_loader, flat=False)
    with torch.no_grad():
        ptr = states(reducer(Xtr).double(), enc, nq); po = states(reducer(Xo).double(), enc, nq)
        htr, means = hem_of(ptr, ytr, nc); ho, _ = hem_of(po, yo, nc, means)
    return dict(zip(itr.tolist(), htr.tolist())), dict(zip(io.tolist(), ho.tolist()))
