import json, sys, warnings
from pathlib import Path
import numpy as np, torch
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hcmr"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS
from densitymatrix import DMQNN
from noise_circuits import observables_from_probs, real_state_prep
import noise_infer as NI

INFER = RESULTS / "noise" / "infer" / "fez"
OUT = RESULTS / "noise" / "train" / "calibration_fake_fez.json"
be = NI.backend("fez")
props = be.properties()
q_ok = range(be.num_qubits)
sx = [props.gate_error("sx", [q]) for q in q_ok]
cz = [props.gate_error("cz", list(e)) for e in be.coupling_map.get_edges()]
ro = [props.readout_error(q) for q in q_ok]
cal = dict(backend="fake_fez", sx_error=float(np.median(sx)), cz_error=float(np.median(cz)), readout_error=float(np.median(ro)))

from qiskit import QuantumCircuit, transpile
gates = {}
for nq in (4, 8):
    c2, c1 = [], []
    for ds in ("mnist", "fmnist"):
        fs = sorted((INFER / f"amplitude_{ds}_{nq}q4c" / "base").glob("seed_*.json"))
        if not fs:
            continue
        f = fs[0]
        layout = json.loads(f.read_text())["layout"]
        z = np.load(f.with_suffix(".npz"))["encoder_out"][:20].astype(float)
        cmap = be.coupling_map.reduce(layout)
        for zi in z:
            qc = QuantumCircuit(nq); real_state_prep(qc, zi)
            t = transpile(qc, coupling_map=cmap, basis_gates=["rz", "sx", "x", "cz"], optimization_level=1,
                          seed_transpiler=0, initial_layout=list(range(nq)))
            o = t.count_ops(); c2.append(o.get("cz", 0)); c1.append(o.get("sx", 0) + o.get("x", 0))
    if c2:
        gates[str(nq)] = dict(two_qubit=float(np.mean(c2)), one_qubit=float(np.mean(c1)))
cal["amplitude_prep_gates"] = gates

check = {}
for enc in ("amplitude", "angle"):
    for nq in (4, 8):
        xi, xa, xm = [], [], []
        for ds in ("mnist", "fmnist"):
            for f in sorted((INFER / f"{enc}_{ds}_{nq}q4c" / "base").glob("seed_*.npz")):
                seed = int(f.stem.split("_")[1]); d = np.load(f)
                ei = np.array([observables_from_probs(p.astype(float), nq, "z_k", 4) for p in d["prob_ideal"]])
                ea = np.array([observables_from_probs(p.astype(float), nq, "z_k", 4) for p in d["prob_noise"]])
                model, _ = NI.build_model(enc, ds, nq, 4, "base", seed)
                sim = DMQNN(nq, 4, 4, enc, False, 0, "cpu", cal); sim.dn = True; sim.prep_survival = 1.0
                W = model.qlayer.weights.detach().float(); z = torch.tensor(d["encoder_out"], dtype=torch.float32)
                with torch.no_grad():
                    em = torch.cat([sim.forward(z[i:i + 100], W, False) for i in range(0, len(z), 100)]).numpy()
                xi.append(ei); xa.append(ea); xm.append(em)
        if not xi:
            continue
        ei, ea, em = (np.concatenate(v).ravel() for v in (xi, xa, xm))
        slope = lambda y: float((y * ei).sum() / (ei * ei).sum())
        check[f"{enc}_{nq}q"] = dict(a_aer=slope(ea), a_model_without_prep=slope(em), ratio=slope(ea) / slope(em), runs=len(xi))
        print(f"{enc:9s} {nq}q: attenuation Aer {slope(ea):.4f} | model without the prep channel {slope(em):.4f} "
              f"| ratio {slope(ea) / slope(em):.4f}")
cal["attenuation_check"] = check
cal["prep_survival_calibrated"] = {str(q): check[f"amplitude_{q}q"]["ratio"] for q in (4, 8) if f"amplitude_{q}q" in check}
OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(cal, indent=1))
print({k: v for k, v in cal.items() if k != "attenuation_check"})
