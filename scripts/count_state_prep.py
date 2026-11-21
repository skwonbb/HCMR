import json, sys, warnings
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hcmr")); sys.path.insert(0, str(Path(__file__).resolve().parent))
import noise_infer as I
from noise_circuits import real_state_prep
from qiskit import QuantumCircuit, transpile

D = I.OUT
out = {}
for bname in ("fez", "nighthawk"):
    be = I.backend(bname)
    for enc in ("amplitude", "angle"):
        for nq in (4, 8):
            for v in I.VARIANTS:
                cnt = []
                for ds in ("mnist", "fmnist"):
                    runs = sorted((D / bname / f"{enc}_{ds}_{nq}q4c" / v).glob("seed_*.json"))
                    if not runs:
                        continue
                    r = json.loads(runs[0].read_text())
                    z = np.load(runs[0].with_suffix(".npz"))["encoder_out"][:20].astype(float)
                    cmap = be.coupling_map.reduce(r["layout"])
                    for zi in z:
                        qc = QuantumCircuit(nq)
                        if enc == "amplitude":
                            real_state_prep(qc, zi)
                        else:
                            for i in range(nq):
                                qc.ry(float(zi[i]), i)
                        t = transpile(qc, coupling_map=cmap, basis_gates=["rz", "sx", "x", "cz"], optimization_level=1,
                                      seed_transpiler=0, initial_layout=list(range(nq)))
                        cnt.append(t.count_ops().get("cz", 0))
                if cnt:
                    out[f"{bname}/{enc}_{nq}q/{v}"] = float(np.mean(cnt))
            print(bname, enc, nq, {v: out.get(f"{bname}/{enc}_{nq}q/{v}") for v in I.VARIANTS}, flush=True)
(D / "state_prep_2q_gates.json").write_text(json.dumps(out, indent=1))
print("saved", D / "state_prep_2q_gates.json")
