import json

SHOTS = 4096


def load_calibration(path):
    return json.loads(open(path, encoding="utf-8").read())


class DMQNN:

    def __init__(self, n, L, n_meas, enc, device_noise, shots, dev, cal):
        import torch
        self.n, self.L, self.n_meas, self.enc, self.dn, self.shots, self.dev = n, L, n_meas, enc, device_noise, shots, dev
        d = 2 ** n
        idx = torch.arange(d)
        bit = lambda x, w: (x >> (n - 1 - w)) & 1
        self.zsign = torch.stack([1 - 2 * bit(idx, w) for w in range(n)], 1).float().to(dev)
        self.perm = {}
        for l in range(L):
            r = l % (n - 1) + 1
            for i in range(n):
                c, t = i, (i + r) % n
                self.perm[(c, t)] = (idx ^ (bit(idx, c) << (n - 1 - t))).to(dev)
        r1, r2 = cal["sx_error"], cal["cz_error"]
        self.lam1 = 1 - (1 - 2 * r1) ** 2
        self.lam2 = 4 * r2 / 3
        g = cal["amplitude_prep_gates"].get(str(n), {"two_qubit": 0, "one_qubit": 0})
        self.prep_survival = (1 - 4 * r2 / 3) ** g["two_qubit"] * (1 - 2 * r1) ** g["one_qubit"]
        fit = cal.get("prep_survival_calibrated", {})
        if enc == "amplitude" and device_noise:
            assert str(n) in fit, "amplitude state-preparation noise not calibrated yet"
            self.prep_survival = fit[str(n)]
        self.ro = 1 - 2 * cal["readout_error"]
        self.g_obs = None

    def _v(self, rho, q):
        n = self.n
        return rho.reshape(rho.shape[0], 2 ** q, 2, 2 ** (n - q - 1), 2 ** q, 2, 2 ** (n - q - 1))

    def u1(self, rho, U, q):
        import torch
        r = torch.einsum("ij,bxjyzkw,lk->bxiyzlw", U, self._v(rho, q), U.conj())
        return r.reshape(rho.shape)

    def full_dep(self, rho, q):
        import torch
        r = self._v(rho, q)
        tr = r[:, :, 0, :, :, 0, :] + r[:, :, 1, :, :, 1, :]
        eye = torch.eye(2, dtype=rho.dtype, device=rho.device)
        return (0.5 * tr[:, :, None, :, :, None, :] * eye[None, None, :, None, None, :, None]).reshape(rho.shape)

    def dep1(self, rho, q, lam):
        return (1 - lam) * rho + lam * self.full_dep(rho, q)

    def dep2(self, rho, q1, q2, lam):
        return (1 - lam) * rho + lam * self.full_dep(self.full_dep(rho, q1), q2)

    def cnot(self, rho, c, t):
        p = self.perm[(c, t)]
        return rho[:, p][:, :, p]

    def _super1(self, U, lam):
        import torch
        SU = torch.einsum("ai,bj->abij", U, U.conj())
        eye = torch.eye(2, dtype=U.dtype, device=U.device)
        D = (1 - lam) * torch.einsum("ai,bj->abij", eye, eye) + lam / 2 * torch.einsum("ab,ij->abij", eye, eye)
        return torch.einsum("abkl,klij->abij", D, SU)

    def _super2(self, lam):
        import torch
        dev = self.dev
        C = torch.zeros(4, 4, dtype=torch.complex64, device=dev)
        for x in range(4):
            c_, t_ = x >> 1, x & 1
            C[(c_ << 1) | (t_ ^ c_), x] = 1
        SU = torch.einsum("ai,bj->abij", C, C.conj())
        e4 = torch.eye(4, dtype=torch.complex64, device=dev)
        D = (1 - lam) * torch.einsum("ai,bj->abij", e4, e4) + lam / 4 * torch.einsum("ab,ij->abij", e4, e4)
        S = torch.einsum("abkl,klij->abij", D, SU)
        return S.reshape(2, 2, 2, 2, 2, 2, 2, 2)

    def apply_super(self, T, S, wires):
        import torch
        n, k = self.n, len(wires)
        ax = [1 + w for w in wires] + [1 + n + w for w in wires]
        out = torch.tensordot(S, T, dims=(list(range(2 * k, 4 * k)), ax))
        return torch.movedim(out, list(range(2 * k)), ax)

    def _angle_hook(self, angles, e_detached):
        import torch
        S, sim = self.shots, self

        def hook(grad):
            g = sim.g_obs
            var = (g ** 2 * (1 - e_detached ** 2).clamp_min(0)).sum(1, keepdim=True) / (2 * S)
            return grad + torch.randn_like(grad) * var.sqrt()
        angles.register_hook(hook)

    @staticmethod
    def rot(w):
        import torch
        phi, th, om = w[..., 0], w[..., 1], w[..., 2]
        c, s = torch.cos(th / 2), torch.sin(th / 2)
        e = lambda a: torch.exp(1j * a.to(torch.complex64))
        return torch.stack([torch.stack([e(-(phi + om) / 2) * c, -e((phi - om) / 2) * s], -1),
                            torch.stack([e(-(phi - om) / 2) * s, e((phi + om) / 2) * c], -1)], -2)

    def forward(self, z, weights, encoder_trainable):
        import torch
        n, d, B = self.n, 2 ** self.n, z.shape[0]
        hook_angles = self.shots and encoder_trainable
        if self.enc == "amplitude":
            v = z / z.norm(dim=1, keepdim=True).clamp_min(1e-12)
            if hook_angles:
                ang = tree_angles(v, n); self._pending = ang; v = tree_state(ang, n)
            psi = v.to(torch.complex64)
        else:
            if hook_angles:
                self._pending = z
            one = torch.stack([torch.cos(z / 2), torch.sin(z / 2)], -1)
            psi = one[:, 0]
            for i in range(1, n):
                psi = torch.einsum("ba,bc->bac", psi, one[:, i]).reshape(B, -1)
            psi = psi.to(torch.complex64)
        R = self.rot(weights)
        if not self.dn:
            st = psi
            for l in range(self.L):
                for i in range(n):
                    st = torch.einsum("ij,bxjy->bxiy", R[l, i], st.reshape(B, 2 ** i, 2, -1)).reshape(B, d)
                r = l % (n - 1) + 1
                for i in range(n):
                    st = st[:, self.perm[(i, (i + r) % n)]]
            p = st.real ** 2 + st.imag ** 2
        else:
            rho = torch.einsum("bi,bj->bij", psi, psi.conj())
            if self.enc == "amplitude":
                eye = torch.eye(d, dtype=rho.dtype, device=rho.device) / d
                rho = self.prep_survival * rho + (1 - self.prep_survival) * eye
            else:
                for q in range(n):
                    rho = self.dep1(rho, q, self.lam1)
            if not getattr(self, "fused", True):
                for l in range(self.L):
                    for i in range(n):
                        rho = self.dep1(self.u1(rho, R[l, i], i), i, self.lam1)
                    r = l % (n - 1) + 1
                    for i in range(n):
                        rho = self.dep2(self.cnot(rho, i, (i + r) % n), i, (i + r) % n, self.lam2)
                p = torch.diagonal(rho, dim1=1, dim2=2).real
            else:
                T = rho.reshape((B,) + (2,) * (2 * n))
                S2 = self._super2(self.lam2)
                for l in range(self.L):
                    for i in range(n):
                        T = self.apply_super(T, self._super1(R[l, i], self.lam1), [i])
                    r = l % (n - 1) + 1
                    for i in range(n):
                        T = self.apply_super(T, S2, [i, (i + r) % n])
                p = torch.diagonal(T.reshape(B, d, d), dim1=1, dim2=2).real
        e = p @ self.zsign[:, : self.n_meas]
        if self.dn:
            e = e * self.ro
        if self.shots:
            e = e + (torch.randn_like(e) * ((1 - e.detach() ** 2).clamp_min(0) / self.shots).sqrt())
            e.register_hook(self._capture)
            if hook_angles:
                self._angle_hook(self._pending, e.detach())
        self._pending = None
        return e

    def _capture(self, grad):
        self.g_obs = grad.detach()
        return grad

    def pqc_grad_noise(self, weights, e_noisy):
        import torch
        g = self.g_obs
        var = (g ** 2 * (1 - e_noisy.detach() ** 2).clamp_min(0)).sum() / (2 * self.shots)
        weights.grad = weights.grad + torch.randn_like(weights.grad) * var.sqrt()


def tree_angles(v, n):
    import torch
    out = []
    for k in range(n):
        blk = v.reshape(v.shape[0], 2 ** k, 2, -1)
        if k < n - 1:
            a, b = blk[:, :, 0].norm(dim=-1), blk[:, :, 1].norm(dim=-1)
        else:
            a, b = blk[:, :, 0, 0], blk[:, :, 1, 0]
        out.append(2 * torch.atan2(b, a + 0.0))
    return torch.cat(out, 1)


def tree_state(ang, n):
    import torch
    psi, o = torch.ones(ang.shape[0], 1, dtype=ang.dtype, device=ang.device), 0
    for k in range(n):
        th = ang[:, o:o + 2 ** k]; o += 2 ** k
        psi = torch.stack([psi * torch.cos(th / 2), psi * torch.sin(th / 2)], -1).reshape(ang.shape[0], -1)
    return psi
