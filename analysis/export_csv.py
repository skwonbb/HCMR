import csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "hcmr"))
from config import SETUPS, tag

E = ROOT / "results"
OUT = ROOT / "paper_data"
SUF = {"amplitude": "", "angle": "_angle"}


def write(name, rows):
    OUT.mkdir(exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / name, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"{name}: {len(rows)} rows")


def seed_runs(d, pattern="seed_*.json"):
    fs = sorted(d.glob(pattern), key=lambda p: int(p.stem.split("_")[1]))
    return [(i + 1, json.loads(f.read_text())) for i, f in enumerate(fs)]


def setting(enc, ds, nq, nc):
    return dict(embedding=enc, dataset=ds, qubits=nq, classes=nc)


rows = []
for s in SETUPS:
    for m in sorted(p.name for p in (E / "full" / tag(*s)).iterdir() if p.is_dir()):
        for run, r in seed_runs(E / "full" / tag(*s) / m):
            rows.append(dict(**setting(*s), method=m, run=run, acc=r["acc"], acc_ce=r["acc_ce_orig"], acc_pe=r["acc_pe_orig"],
                             ce_size=r["ce_size_own"], mu=(r.get("margin") or {}).get("mu"), lam=(r.get("margin") or {}).get("lam"),
                             joint_epochs=r.get("joint_epochs"),
                             **{f"test_acc_J{j}": v for j, v in (r.get("joint_test_acc") or {}).items()},
                             **{f"val_acc_J{j}": v for j, v in (r.get("joint_val_acc") or {}).items()}))
write("main_runs.csv", rows)

rows = []
for s in SETUPS:
    for f in sorted((E / "full" / tag(*s)).glob("*/margin.json")):
        m = json.loads(f.read_text())
        for c in m["grid"]:
            rows.append(dict(**setting(*s), method=f.parent.name, mu=c["mu"], lam=c["lam"], val_ce_size=c["val_ce"],
                             selected=int(c["mu"] == m["mu"] and c["lam"] == m["lam"])))
write("margin_grid.csv", rows)

for sub, field in (("j_sweep", "joint"), ("joint_window", "window")):
    rows = []
    for d in sorted((E / sub).glob("*/*")):
        if not d.is_dir():
            continue
        enc, ds, su = d.parent.name.split("_"); nq, nc = int(su.split("q")[0]), int(su.split("q")[1][:-1])
        for run, r in seed_runs(d):
            if f"{field}_test_acc" not in r:
                continue
            for k, v in r[f"{field}_test_acc"].items():
                rows.append(dict(**setting(enc, ds, nq, nc), method=d.name, run=run,
                                 **({"joint_epochs": int(k)} if field == "joint" else {"window_start": int(k), "window_len": r["window_len"]}),
                                 test_acc=v, val_acc=r[f"{field}_val_acc"][k]))
    write(f"{sub}.csv", rows)

rows, per = [], []
for enc, ds, nq, nc in SETUPS:
    su = f"{nq}q{nc}c"
    for cond in sorted(p.name for p in (E / "layers" / enc / ds / su).iterdir()):
        ro, L = cond.rsplit("_L", 1)
        for run, r in seed_runs(E / "layers" / enc / ds / su / cond):
            rows.append(dict(**setting(enc, ds, nq, nc), readout=ro, depth=int(L), run=run, acc=r["acc"], acc_ce=r["acc_ce"],
                             acc_pe=r["acc_pe"], n_params=r.get("n_params"), n_observables=r.get("n_obs")))
write("layers_runs.csv", rows)

rows, probes = [], []
for enc, ds, nq, nc in SETUPS:
    su = f"{nq}q{nc}c"
    c = json.loads((E / f"ceiling{SUF[enc]}" / ds / f"{su}.json").read_text())
    for k, v in c["probes"].items():
        probes.append(dict(**setting(enc, ds, nq, nc), stage=k, classifier=v["model"], val_acc=v["val_acc"], acc=v["acc"],
                           acc_ce=v["acc_ce"], acc_pe=v["acc_pe"]))
    ids = sorted(int(i) for i in c["y_test"])
    ref = seed_runs(E / "layers" / enc / ds / su / "z_k_L4"); zz = seed_runs(E / "layers" / enc / ds / su / "z_zz_L4")
    mlp = seed_runs(E / f"classical{SUF[enc]}" / ds / su / "mlp_p")
    psi = max(("psi", "psi2"), key=lambda k: c["probes"][k]["val_acc"] if k in c["probes"] else -1)
    for i in ids:
        q = str(i)
        rows.append(dict(**setting(enc, ds, nq, nc), sample=i, label=c["y_test"][q], hem=c["hem_test"][q],
                         ce=int(c["hem_test"][q] < 0), pred_raw_probe=c["probes"]["raw"]["per_idx"][q],
                         pred_psi_probe=c["probes"][psi]["per_idx"][q],
                         **{f"pred_qnn_zk_run{n}": r["per_idx"][q] for n, r in ref},
                         **{f"pred_qnn_zzk_run{n}": r["per_idx"][q] for n, r in zz},
                         **{f"pred_mlp_run{n}": r["per_idx"][q] for n, r in mlp}))
write("diagnosis_per_sample.csv", rows)
write("diagnosis_probes.csv", probes)

rows = []
for enc in SUF:
    for p in sorted((E / f"classical{SUF[enc]}").glob("*/*/mlp_p")):
        ds, su = p.parent.parent.name, p.parent.name; nq, nc = int(su.split("q")[0]), int(su.split("q")[1][:-1])
        for run, r in seed_runs(p):
            rows.append(dict(**setting(enc, ds, nq, nc), run=run, acc=r["acc"], acc_ce=r["acc_ce"], acc_pe=r["acc_pe"],
                             n_params=r["n_params"]))
write("diagnosis_mlp_runs.csv", rows)

rows = []
for cond in ("none", "shots", "device", "both"):
    for d in sorted((E / "noise" / "train" / cond).glob("*/*")):
        enc, ds, su = d.parent.name.split("_"); nq, nc = int(su.split("q")[0]), int(su.split("q")[1][:-1])
        for run, r in seed_runs(d):
            for j, v in r["J"].items():
                rows.append(dict(**setting(enc, ds, nq, nc), method=d.name, run=run, training_noise=cond, joint_epochs=int(j),
                                 acc=v["ideal"]["acc"], acc_ce=v["ideal"]["acc_ce"], acc_pe=v["ideal"]["acc_pe"],
                                 acc_noisy_inference=v["noisy"]["acc"], ce_size=v["ce_size_test"], encoder_drift=v["encoder_drift"]))
write("noise_training_runs.csv", rows)

rows = []
for d in sorted((E / "noise" / "grad").glob("*/*")):
    enc, ds, su = d.parent.name.split("_"); nq, nc = int(su.split("q")[0]), int(su.split("q")[1][:-1])
    for run, r in seed_runs(d):
        for path in ("J10", "J60"):
            for e in r[path]:
                for cond in ("shots", "device", "both"):
                    for part in ("pqc", "enc"):
                        if f"cos_{part}" in e[cond]:
                            rows.append(dict(**setting(enc, ds, nq, nc), method=d.name, run=run, path=path, epoch=e["epoch"],
                                             encoder_trained=e["encoder_trained"], noise=cond, part=part,
                                             cos=float(np.mean(e[cond][f"cos_{part}"])), norm_ratio=float(np.mean(e[cond][f"ratio_{part}"]))))
write("noise_gradient.csv", rows)

prep = json.loads((E / "noise" / "infer" / "state_prep_2q_gates.json").read_text())
rows = []
for be in ("fez", "nighthawk"):
    for d in sorted((E / "noise" / "infer" / be).glob("*/*")):
        enc, ds, su = d.parent.name.split("_"); nq, nc = int(su.split("q")[0]), int(su.split("q")[1][:-1])
        for run, r in seed_runs(d):
            z = np.load(d / f"seed_{r['seed']}.npz"); pi = z["prob_ideal"].astype(float)
            to_u = 0.5 * np.abs(pi - 1 / pi.shape[1]).sum(1)
            tvd = {k: 0.5 * np.abs(pi - z[f"prob_{k}"].astype(float)).sum(1) for k in ("shots", "noise", "both")}
            for cond, k in (("ideal", None), ("shots_only", "shots"), ("noise_only", "noise"), ("both", "both")):
                rows.append(dict(backend=be, **setting(enc, ds, nq, nc), variant=d.name, run=run, condition=cond,
                                 acc=r[cond]["acc"], acc_ce=r[cond]["acc_ce"], acc_pe=r[cond]["acc_pe"],
                                 tvd=float(tvd[k].mean()) if k else 0.0, tvd_normalized=float((tvd[k] / to_u).mean()) if k else 0.0,
                                 cz_total=r["two_qubit_gates_mean"], cz_state_preparation=prep.get(f"{be}/{enc}_{nq}q/{d.name}")))
write("noise_inference_runs.csv", rows)

dev = E / "noise" / "device" / "fez"
m = json.loads((dev / "analysis_4q_margin.json").read_text())
an = {r["key"]: r for r in json.loads((dev / "analysis.json").read_text())}
rows = []
for k, v in m.items():
    enc, var = k.split("/"); r = an[f"{enc}_mnist_4q4c/{var}/seed_{next(iter(an[x]['seed'] for x in an if x.startswith(f'{enc}_mnist_4q4c/{var}/')))}"]
    rows.append(dict(**setting(enc, "mnist", 4, 4), variant=var, samples=100, acc_ideal=v["acc_i"] / 100, acc_noisy_simulation=v["acc_s"] / 100,
                     acc_device=v["acc_d"] / 100, acc_ce_device=r["device"]["acc_ce"], acc_pe_device=r["device"]["acc_pe"],
                     flipped_samples=v["flips"], tvd_normalized_simulation=v["tvd_s"], tvd_normalized_device=v["tvd_d"],
                     logit_margin_median=v["margin"], boundary_distance_median=v["boundary_dist_obs"],
                     device_shift_median=v["device_shift_obs"], top_class_share_device=v["top_class_share_device"],
                     cz_total=r["two_qubit_gates_mean"], physical_qubits=" ".join(map(str, r["layout"]))))
write("device_ibm_fez.csv", rows)

def fake_medians(name):
    from qiskit_ibm_runtime import fake_provider
    b = getattr(fake_provider, name)(); t = b.target
    err = lambda g: float(np.median([p.error for p in t[g].values() if p is not None and p.error is not None]))
    qp = [b.qubit_properties(i) for i in range(b.num_qubits)]
    return dict(sx_error=err("sx"), cz_error=err("cz"), readout_error=err("measure"),
                t1_us=float(np.median([q.t1 for q in qp if q.t1])) * 1e6, t2_us=float(np.median([q.t2 for q in qp if q.t2])) * 1e6)


def device_medians(files):
    per = []
    for f in files:
        d = json.loads(f.read_text()); q = {k: [] for k in ("sx_error", "cz_error", "readout_error", "t1_us", "t2_us")}
        for qb in d["qubits"]:
            m = {p["name"]: p["value"] for p in qb}
            q["readout_error"].append(m.get("readout_error")); q["t1_us"].append(m.get("T1")); q["t2_us"].append(m.get("T2"))
        for g in d["gates"]:
            e = [p["value"] for p in g["parameters"] if p["name"] == "gate_error"]
            if e and g["gate"] in ("sx", "cz"):
                q[f"{g['gate']}_error"].append(e[0])
        row = {}
        for k, a in q.items():
            a = np.array([x for x in a if x is not None], float)
            row[k] = float(np.median(a[a < 1] if k.endswith("error") else a))
        per.append(row)
    return {k: float(np.median([r[k] for r in per])) for k in per[0]}


cal = json.loads((E / "noise" / "train" / "calibration_fake_fez.json").read_text())
rows = [dict(backend="FakeFez", quantity=k, value=v) for k, v in fake_medians("FakeFez").items()]
rows += [dict(backend="FakeNighthawk", quantity=k, value=v) for k, v in fake_medians("FakeNighthawk").items()]
dev = sorted((E / "noise" / "device" / "fez" / "fetch" / "calibration").glob("*_mnist_4q4c__*.json"))
rows += [dict(backend="ibm_fez (median over the runs)", quantity=k, value=v) for k, v in device_medians(dev).items()]
T = "training-noise model (FakeFez)"
rows += [dict(backend=T, quantity="depolarizing_one_qubit", value=1 - (1 - 2 * cal["sx_error"]) ** 2),
         dict(backend=T, quantity="depolarizing_two_qubit", value=4 * cal["cz_error"] / 3),
         dict(backend=T, quantity="readout_scale", value=1 - 2 * cal["readout_error"])]
rows += [dict(backend=T, quantity=f"state_preparation_survival_{q}q", value=v) for q, v in cal["prep_survival_calibrated"].items()]
rows += [dict(backend=T, quantity=f"state_preparation_{k}_{q}q", value=v) for q, g in cal["amplitude_prep_gates"].items() for k, v in g.items()]
rows += [dict(backend=T, quantity=f"attenuation_ratio_{k}", value=v["ratio"]) for k, v in cal["attenuation_check"].items()]
write("noise_model_calibration.csv", rows)
