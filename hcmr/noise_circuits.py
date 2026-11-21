import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import UCRYGate


def real_state_prep(qc, v):
    n = qc.num_qubits
    v = np.asarray(v, dtype=float); v = v / np.linalg.norm(v)
    for k in range(n):
        blocks = v.reshape(2 ** k, 2, -1)
        if k < n - 1:
            a, b = np.linalg.norm(blocks[:, 0], axis=1), np.linalg.norm(blocks[:, 1], axis=1)
        else:
            a, b = blocks[:, 0, 0], blocks[:, 1, 0]
        theta = 2 * np.arctan2(b, a)
        if k == 0:
            qc.ry(float(theta[0]), 0)
        else:
            order = [int(format(i, f"0{k}b")[::-1], 2) for i in range(2 ** k)]
            qc.append(UCRYGate([float(theta[j]) for j in order]), [k] + list(range(k)))


def qnn_circuit(z, enc, weights, measure=True):
    L, n, _ = weights.shape
    qc = QuantumCircuit(n)
    if enc == "amplitude":
        real_state_prep(qc, z)
    else:
        for i in range(n):
            qc.ry(float(z[i]), i)
    for l in range(L):
        for i in range(n):
            phi, theta, omega = (float(a) for a in weights[l, i])
            qc.rz(phi, i); qc.ry(theta, i); qc.rz(omega, i)
        if n > 1:
            r = l % (n - 1) + 1
            for i in range(n):
                qc.cx(i, (i + r) % n)
    if measure:
        qc.measure_all()
    return qc


def observables_from_probs(p, n, readout, n_meas):
    idx = np.arange(2 ** n)
    z = 1 - 2 * ((idx[:, None] >> np.arange(n)[None, :]) & 1)
    ez = p @ z
    if readout == "z_zz":
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        return np.concatenate([ez, np.array([p @ (z[:, i] * z[:, j]) for i, j in pairs])])
    return ez[:n_meas]


def counts_to_probs(counts, n):
    p = np.zeros(2 ** n)
    tot = sum(counts.values())
    for k, c in counts.items():
        p[int(k.replace(" ", ""), 2)] += c / tot
    return p
