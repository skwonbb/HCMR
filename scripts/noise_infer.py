import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("QISKIT_PARALLEL", "FALSE")
import argparse, json, sys, time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hcmr"))
from config import RESULTS, CLASS_ORDER, N_TRAIN, N_VAL, N_TEST, DATA_SEED, BATCH, tag

OUT = RESULTS / "noise" / "infer"
SHOTS = 4096
SETUPS = [(e, d, q, 4) for e in ("amplitude", "angle") for d in ("mnist", "fmnist") for q in (4, 8)]
VARIANTS = {"base": ("z_k", 4), "L8": ("z_k", 8), "L16": ("z_k", 16), "zz": ("z_zz", 4), "hcmr": ("z_k", 4),
            "hcmr_cnn": ("z_k", 4)}


def backend(name):
    from qiskit_ibm_runtime.fake_provider import FakeFez, FakeNighthawk
    return {"fez": FakeFez, "nighthawk": FakeNighthawk}[name]()


def weights_dir(enc, ds, nq, nc, var):
    if var in ("hcmr", "hcmr_cnn"):
        return RESULTS / "noise" / "models" / tag(enc, ds, nq, nc) / var
    ro, L = VARIANTS[var]
    return RESULTS / "layers" / enc / ds / f"{nq}q{nc}c" / f"{ro}_L{L}"


def weights_path(enc, ds, nq, nc, var, seed):
    d = weights_dir(enc, ds, nq, nc, var)
    return d / f"seed_{seed}.pt", d / f"seed_{seed}.json"


def model_seeds(setup, var):
    return sorted(int(p.stem.split("_")[1]) for p in weights_dir(*setup, var).glob("seed_*.pt"))


def build_model(enc, ds, nq, nc, var, seed):
    import torch, torch.nn as nn
    from encoders import make_cnn
    from student import QuantumStudent
    from torchsim import from_pennylane
    ro, L = VARIANTS[var]
    kw = {"reducer_type": "angle", "encoding": "angle"} if enc == "angle" else {}
    torch.manual_seed(seed)
    pl = QuantumStudent(nq, L, nc, readout=ro, **kw)
    model = from_pennylane(pl, enc, ro)
    sd = torch.load(weights_path(enc, ds, nq, nc, var, seed)[0], map_location="cpu")
    if var == "hcmr_cnn":
        model.reducer = make_cnn(enc, nq, ds)
    elif var == "hcmr":
        class Lin(nn.Module):
            def __init__(self):
                super().__init__()
                self.W = nn.Parameter(sd["reducer.W"].clone()); self.b = nn.Parameter(sd["reducer.b"].clone())

            def forward(self, x):
                z = x.flatten(1) @ self.W.T + self.b
                return torch.pi * torch.sigmoid(z) if enc == "angle" else z
        model.reducer = Lin()
    model.load_state_dict(sd, strict=False)
    return model.eval(), ro


def qubit_subset(be, first, n):
    import networkx as nx
    g = nx.Graph(list(be.coupling_map.get_edges()))
    start = [first.layout.initial_index_layout()[i] for i in range(n)]
    chosen = [start[0]]
    pref = set(start)
    while len(chosen) < n:
        nb = sorted({v for u in chosen for v in g.neighbors(u)} - set(chosen), key=lambda v: (v not in pref, v))
        chosen.append(nb[0])
    return chosen


def readout_matrices(be, phys):
    import numpy as np
    props = be.properties()
    out = []
    for q in phys:
        p10 = props.qubit_property(q, "prob_meas1_prep0")[0]; p01 = props.qubit_property(q, "prob_meas0_prep1")[0]
        out.append(np.array([[1 - p10, p01], [p10, 1 - p01]]))
    return out


def apply_readout(p, mats):
    import numpy as np
    n = len(mats); t = p.reshape([2] * n)
    for i, A in enumerate(mats):
        ax = n - 1 - i
        t = np.moveaxis(np.tensordot(A, t, axes=([1], [ax])), 0, ax)
    return t.reshape(-1)


def run(task):
    try:
        return _run(task)
    except Exception as e:
        import traceback
        return f"FAILED {task}: {type(e).__name__}: {e}\n{traceback.format_exc()}"


def _run(task):
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np, torch
    torch.set_num_threads(1)
    from hem import arrays, hem_split
    from qnn import pool_reducer
    from data import load_data
    from noise_circuits import qnn_circuit, observables_from_probs
    from qiskit.quantum_info import Statevector
    (enc, ds, nq, nc), var, seed, bname, check = task
    tagd = f"{enc}_{ds}_{nq}q{nc}c"
    outp = OUT / bname / tagd / var / f"seed_{seed}.json"
    if outp.exists() and not check:
        return f"{bname} {tagd} {var} s{seed}: cached"
    t0 = time.perf_counter()
    model, ro = build_model(enc, ds, nq, nc, var, seed)
    tl, _, te, *_ = load_data(ds, N_TRAIN, N_VAL, N_TEST, BATCH, DATA_SEED, CLASS_ORDER[ds][:nc])
    X, y, ids = arrays(te, flat=False)
    with torch.no_grad():
        z = model.reducer(X).double().numpy(); logit_ref = model(X).numpy()
    W = model.qlayer.weights.detach().double().numpy()
    hw, hb = model.head.weight.detach().double().numpy(), model.head.bias.detach().double().numpy()
    n_meas = model.n_meas
    _, hem = hem_split(pool_reducer(enc, nq, nc), enc, nq, nc, tl, te)
    ce = np.array([hem[int(i)] < 0 for i in ids]); y = y.numpy()

    def score(probs_list):
        logits = np.array([hw @ observables_from_probs(p, nq, ro, n_meas) + hb for p in probs_list])
        ok = logits.argmax(1) == y
        return dict(acc=float(ok.mean()), acc_ce=float(ok[ce].mean()), acc_pe=float(ok[~ce].mean())), logits

    p_ideal = [Statevector(qnn_circuit(zi, enc, W, measure=False)).probabilities() for zi in z]
    res_ideal, logits = score(p_ideal)
    stored = json.loads(weights_path(enc, ds, nq, nc, var, seed)[1].read_text())["acc"]
    dev = float(np.abs(logits - logit_ref).max())
    if check:
        return (f"{tagd:22s} {var:8s} s{seed}: ideal acc {res_ideal['acc']*100:5.2f} | stored {stored*100:5.2f} "
                f"| max |logit diff| {dev:.1e}")
    rng = np.random.default_rng([seed, nq, len(var), SETUPS.index((enc, ds, nq, nc))])
    sample = lambda p: rng.multinomial(SHOTS, np.clip(p, 0, None) / np.clip(p, 0, None).sum()) / SHOTS
    p_shots = [sample(p) for p in p_ideal]
    res_shots, logits_shots = score(p_shots)
    res = dict(encoding=enc, ds=ds, nq=nq, nc=nc, variant=var, seed=seed, backend=bname, shots=SHOTS,
               stored_acc=stored, max_logit_dev_ideal=dev, ideal=res_ideal, shots_only=res_shots)
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    be = backend(bname)
    sim = AerSimulator(method="density_matrix", noise_model=NoiseModel.from_backend(be), max_parallel_threads=1,
                       max_parallel_experiments=1)
    from qiskit import QuantumCircuit
    layout = qubit_subset(be, transpile(qnn_circuit(z[0], enc, W, measure=False), be, optimization_level=1,
                                        seed_transpiler=0), nq)
    cmap = be.coupling_map.reduce(layout)
    basis = [g for g in ("rz", "sx", "x", "cz", "ecr") if g in be.operation_names]
    p_noise, n2q = [], []
    for k in range(0, len(z), 50):
        cs = transpile([qnn_circuit(zi, enc, W, measure=False) for zi in z[k:k + 50]], coupling_map=cmap,
                       basis_gates=basis, optimization_level=1, seed_transpiler=0, initial_layout=list(range(nq)))
        wide, physs = [], []
        for c in cs:
            phys = [layout[c.layout.final_index_layout()[i]] for i in range(nq)]
            w = QuantumCircuit(be.num_qubits); w.compose(c, qubits=layout, inplace=True); w.save_probabilities(qubits=phys)
            wide.append(w); physs.append(phys)
            n2q.append(sum(v for g, v in c.count_ops().items() if g in ("cz", "ecr", "cx")))
        out = sim.run(wide, shots=1).result()
        for j, phys in enumerate(physs):
            p_noise.append(apply_readout(np.asarray(out.data(j)["probabilities"]), readout_matrices(be, phys)))
    p_both = [sample(p) for p in p_noise]
    res["noise_only"], logits_noise = score(p_noise); res["both"], logits_both = score(p_both)
    res.update(layout=layout, two_qubit_gates_mean=float(np.mean(n2q)), elapsed=time.perf_counter() - t0)
    outp.parent.mkdir(parents=True, exist_ok=True)
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    np.savez_compressed(outp.with_suffix(".npz"), ids=np.asarray(ids), y=y, ce=ce, encoder_out=f32(z),
                        prob_ideal=f32(p_ideal), prob_shots=f32(p_shots), prob_noise=f32(p_noise), prob_both=f32(p_both),
                        logit_ideal=f32(logits), logit_shots=f32(logits_shots), logit_noise=f32(logits_noise),
                        logit_both=f32(logits_both), two_qubit_gates=np.asarray(n2q))
    outp.write_text(json.dumps(res))
    return (f"{bname:9s} {tagd:22s} {var:8s} s{seed}: ideal {res_ideal['acc']*100:5.1f} | shots "
            f"{res['shots_only']['acc']*100:5.1f} | noise {res['noise_only']['acc']*100:5.1f} | both "
            f"{res['both']['acc']*100:5.1f} | 2q {np.mean(n2q):.0f} ({res['elapsed']:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="fez,nighthawk"); ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--variants", default=",".join(VARIANTS)); ap.add_argument("--setups", default="")
    a = ap.parse_args()
    only = set(a.setups.split(",")) if a.setups else None
    tasks = [(s, v, sd, b, False) for b in a.backends.split(",") for s in SETUPS if not only or tag(*s) in only
             for v in a.variants.split(",") for sd in model_seeds(s, v)]
    tasks.sort(key=lambda t: -(t[0][2] * (4 if t[0][0] == "amplitude" else 1) * VARIANTS[t[1]][1]))
    print(f"noise inference: {len(tasks)} tasks, {a.workers} workers", flush=True)
    t0 = time.perf_counter()
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for i, msg in enumerate(pool.imap_unordered(run, tasks), 1):
            print(f"[{i}/{len(tasks)} {(time.perf_counter() - t0) / 60:.0f}min] {msg}", flush=True)
