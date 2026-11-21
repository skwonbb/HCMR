from __future__ import annotations
import math
import torch
import torch.nn as nn


def _sign_matrix(nq: int, readout: str, n_meas: int) -> torch.Tensor:
    b = torch.arange(2 ** nq)
    z = torch.stack([1 - 2 * ((b >> (nq - 1 - i)) & 1) for i in range(nq)]).double()
    if readout == "z_zz":
        zz = [z[i] * z[j] for i in range(nq) for j in range(i + 1, nq)]
        return torch.cat([z, torch.stack(zz)]) if zz else z
    return z[:n_meas]


def _cnot_perm(nq: int, c: int, t: int) -> torch.Tensor:
    b = torch.arange(2 ** nq)
    cbit = (b >> (nq - 1 - c)) & 1
    return b ^ (cbit << (nq - 1 - t))


class TorchQLayer(nn.Module):
    def __init__(self, nq: int, n_layers: int, encoding: str, readout: str, n_meas: int, mode: str = "auto"):
        super().__init__()
        assert mode in ("auto", "fused", "layerwise", "pergate")
        self.nq, self.L, self.encoding, self.mode = nq, n_layers, encoding, mode
        self.weights = nn.Parameter(torch.empty(n_layers, nq, 3).uniform_(0, 2 * math.pi))
        ranges = [(l % (nq - 1)) + 1 for l in range(n_layers)] if nq > 1 else [0] * n_layers
        perms = []
        for r in ranges:
            p = torch.arange(2 ** nq)
            if nq > 1:
                for i in range(nq):
                    p = p[_cnot_perm(nq, i, (i + r) % nq)]
            perms.append(p)
        self.register_buffer("perms", torch.stack(perms))
        self.register_buffer("signs", _sign_matrix(nq, readout, n_meas))

    @staticmethod
    def _rot(w):
        phi, th, om = w[:, 0], w[:, 1], w[:, 2]
        ct, st = torch.cos(th / 2), torch.sin(th / 2)
        ep = torch.exp(-0.5j * (phi + om)); em = torch.exp(-0.5j * (phi - om))
        return torch.stack([torch.stack([ep * ct, -em.conj() * st], -1),
                            torch.stack([em * st, ep.conj() * ct], -1)], -2)

    def unitary(self, cdt):
        rdt = torch.float64 if cdt == torch.complex128 else torch.float32
        U = torch.eye(2 ** self.nq, dtype=cdt, device=self.weights.device)
        for l in range(self.L):
            R = self._rot(self.weights[l].to(rdt)).to(cdt)
            K = R[0]
            for i in range(1, self.nq):
                K = torch.kron(K, R[i])
            U = (K @ U)[self.perms[l]]
        return U

    def resolved_mode(self, device):
        if self.mode != "auto":
            return self.mode
        return "pergate" if torch.device(device).type == "cpu" else "layerwise"

    def forward(self, x):
        cdt = torch.complex128 if x.dtype == torch.float64 else torch.complex64
        if self.encoding == "amplitude":
            psi = (x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)).to(cdt)
        else:
            c, s = torch.cos(x / 2), torch.sin(x / 2)
            psi = torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device)
            for i in range(self.nq):
                psi = torch.einsum("nb,nk->nbk", psi, torch.stack([c[:, i], s[:, i]], 1)).reshape(x.shape[0], -1)
            psi = psi.to(cdt)
        mode = self.resolved_mode(x.device)
        if mode == "fused":
            psi = psi @ self.unitary(cdt).T
        else:
            rdt = torch.float64 if cdt == torch.complex128 else torch.float32
            B, n = x.shape[0], self.nq
            for l in range(self.L):
                R = self._rot(self.weights[l].to(rdt)).to(cdt)
                if mode == "layerwise":
                    K = R[0]
                    for i in range(1, n):
                        K = torch.kron(K, R[i])
                    psi = psi @ K.T
                else:
                    for i in range(n):
                        psi = torch.einsum("ab,xbz->xaz", R[i], psi.reshape(B * 2 ** i, 2, 2 ** (n - i - 1)))
                    psi = psi.reshape(B, -1)
                psi = psi[:, self.perms[l]]
        probs = psi.real ** 2 + psi.imag ** 2
        return probs @ self.signs.to(probs.dtype).T


class TorchQuantumStudent(nn.Module):

    def __init__(self, n_qubits=4, n_layers=2, num_classes=4, reducer=None, encoding="amplitude", readout="z_k",
                 mode="auto"):
        super().__init__()
        from student import readout_size
        self.n_qubits, self.encoding = n_qubits, encoding
        self.n_meas = readout_size(readout, n_qubits, num_classes)
        self.reducer = reducer
        self.qlayer = TorchQLayer(n_qubits, n_layers, encoding, readout, min(n_qubits, num_classes), mode)
        self.head = nn.Linear(self.n_meas, num_classes)

    def forward(self, x):
        z = self.reducer(x)
        return self.head(self.qlayer(z).to(self.head.weight.dtype))


def from_pennylane(pl_model, encoding="amplitude", readout="z_k", mode="auto"):
    nq = pl_model.n_qubits
    L = pl_model.qlayer.weights.shape[0]
    nc = pl_model.head.out_features
    m = TorchQuantumStudent(nq, L, nc, reducer=pl_model.reducer, encoding=encoding, readout=readout, mode=mode)
    with torch.no_grad():
        m.qlayer.weights.copy_(pl_model.qlayer.weights)
        m.head.weight.copy_(pl_model.head.weight); m.head.bias.copy_(pl_model.head.bias)
    return m
