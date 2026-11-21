from __future__ import annotations

from typing import Literal

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F


class TutorialReducer(nn.Module):

    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.n_qubits = n_qubits
        self.target_dim = 2 ** n_qubits
        side = int(round(self.target_dim ** 0.5))
        if side * side != self.target_dim:
            raise ValueError(
                f"n_qubits={n_qubits} -> target_dim={self.target_dim} is not a perfect square; "
                "TutorialReducer only supports even-qubit counts."
            )
        self.target_side = side

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = F.adaptive_avg_pool2d(x, (self.target_side, self.target_side))
        return x.flatten(1)


class AngleReducer(nn.Module):

    def __init__(self, n_qubits: int = 4):
        super().__init__()
        self.n_qubits = n_qubits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = F.adaptive_avg_pool2d(x, (1, self.n_qubits))
        x = x.flatten(1)
        return x * 3.14159265


def readout_size(readout: str, n_qubits: int, num_classes: int) -> int:
    if readout == "z_k":
        return min(n_qubits, num_classes)
    if readout == "z_all":
        return n_qubits
    if readout == "z_zz":
        return n_qubits + n_qubits * (n_qubits - 1) // 2
    raise ValueError(f"Unknown readout: {readout}")


def _observables(readout: str, n_qubits: int, n_meas: int):
    if readout == "z_zz":
        obs = [qml.PauliZ(i) for i in range(n_qubits)]
        obs += [qml.PauliZ(i) @ qml.PauliZ(j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
        return obs
    return [qml.PauliZ(i) for i in range(n_meas)]


def _make_qnode(n_qubits: int, n_meas: int, device_kind: str, encoding: str = "amplitude",
                readout: str = "z_k"):
    if device_kind == "default":
        dev = qml.device("default.qubit", wires=n_qubits)
        diff = "backprop"
    elif device_kind == "lightning":
        dev = qml.device("lightning.qubit", wires=n_qubits)
        diff = "adjoint"
    else:
        raise ValueError(f"Unknown device_kind: {device_kind}")

    @qml.qnode(dev, interface="torch", diff_method=diff)
    def circuit(inputs, weights):
        if encoding == "amplitude":
            qml.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
        elif encoding == "angle":
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        else:
            raise ValueError(f"Unknown encoding: {encoding}")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(o) for o in _observables(readout, n_qubits, n_meas)]

    return circuit


class QuantumStudent(nn.Module):

    def __init__(
        self,
        n_qubits: int = 4,
        n_layers: int = 2,
        num_classes: int = 4,
        reducer_type: Literal["tutorial", "angle", "pca"] = "tutorial",
        device_kind: Literal["default", "lightning"] = "default",
        pca_reducer: nn.Module | None = None,
        encoding: Literal["amplitude", "angle"] = "amplitude",
        readout: Literal["z_k", "z_all", "z_zz"] = "z_k",
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_meas = readout_size(readout, n_qubits, num_classes)
        self.encoding = encoding

        if reducer_type == "tutorial":
            self.reducer = TutorialReducer(n_qubits)
        elif reducer_type == "angle":
            self.reducer = AngleReducer(n_qubits)
        elif reducer_type == "pca":
            if pca_reducer is None:
                raise ValueError("reducer_type='pca' requires a fitted pca_reducer instance.")
            self.reducer = pca_reducer
        else:
            raise ValueError(f"Unknown reducer_type: {reducer_type}")

        circuit = _make_qnode(n_qubits, self.n_meas, device_kind, encoding=encoding, readout=readout)
        weight_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=n_qubits)
        self.qlayer = qml.qnn.TorchLayer(circuit, {"weights": weight_shape})
        self.head = nn.Linear(self.n_meas, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.reducer(x)
        q = self.qlayer(z)
        return self.head(q)


if __name__ == "__main__":
    import time
    from data import load_data

    torch.manual_seed(42)
    tl, _, _, _, _, _ = load_data("mnist", batch_size=8, seed=42)
    xb, yb, ib = next(iter(tl))
    print(f"input batch: x={tuple(xb.shape)}")

    red = TutorialReducer(n_qubits=4)
    z = red(xb)
    print(f"reducer 4q: x{tuple(xb.shape)} -> z{tuple(z.shape)}  (expect (8,16))")
    assert z.shape == (8, 16)

    red8 = TutorialReducer(n_qubits=8)
    z8 = red8(xb)
    print(f"reducer 8q: x{tuple(xb.shape)} -> z{tuple(z8.shape)}  (expect (8,256))")
    assert z8.shape == (8, 256)

    print("\n-- default.qubit (backprop, broadcast) --")
    model_d = QuantumStudent(n_qubits=4, n_layers=2, num_classes=4, device_kind="default")
    t0 = time.perf_counter()
    out_d = model_d(xb)
    t_d = time.perf_counter() - t0
    print(f"forward batch=8: out={tuple(out_d.shape)}  time={t_d * 1000:.1f}ms")
    assert out_d.shape == (8, 4)

    loss = F.cross_entropy(out_d, yb)
    loss.backward()
    head_grad = model_d.head.weight.grad
    qw_grad = next(p.grad for n, p in model_d.qlayer.named_parameters() if "weights" in n)
    print(f"backward OK: head_grad_norm={head_grad.norm():.4f}  qweights_grad_norm={qw_grad.norm():.4f}")
    assert head_grad is not None and qw_grad is not None

    print("\n-- lightning.qubit (adjoint, internal loop) --")
    torch.manual_seed(42)
    model_l = QuantumStudent(n_qubits=4, n_layers=2, num_classes=4, device_kind="lightning")
    t0 = time.perf_counter()
    out_l = model_l(xb)
    t_l = time.perf_counter() - t0
    print(f"forward batch=8: out={tuple(out_l.shape)}  time={t_l * 1000:.1f}ms")
    assert out_l.shape == (8, 4)

    print(f"\nspeed ratio (default / lightning): {t_d / t_l:.2f}x")
    print(f"  default = {t_d*1000:.1f}ms  lightning = {t_l*1000:.1f}ms  for batch=8, 4q, 2 layers")
