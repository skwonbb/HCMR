import os
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("QISKIT_PARALLEL", "FALSE")
import argparse, datetime as dt, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hcmr")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import RESULTS, CLASS_ORDER, N_TRAIN, N_VAL, N_TEST, DATA_SEED, BATCH
import noise_infer as NI

BACKEND = "ibm_fez"
SHOTS = 4096
PER_CLASS = 50
SETUPS = [("amplitude", "mnist", 4, 4), ("angle", "mnist", 4, 4)]
VARIANTS = list(NI.VARIANTS)


def out_dir(fake):
    return RESULTS / "noise" / "device" / ("fake_test" if fake else "fez")


def tag(s):
    return f"{s[0]}_{s[1]}_{s[2]}q{s[3]}c"


DEVICE_PER_CLASS = 25


def items(seed):
    return [(s, v, seed) for s in SETUPS for v in VARIANTS]


def first_seed():
    common = set.intersection(*(set(NI.model_seeds(s, v)) for s in SETUPS for v in VARIANTS))
    return min(common)


def device_index(y):
    import numpy as np
    return np.sort(np.concatenate([np.flatnonzero(y == c)[:DEVICE_PER_CLASS] for c in np.unique(y)]))


def key(s, v, sd):
    return f"{tag(s)}/{v}/seed_{sd}"


def sample_ids(ds, ids, y):
    import numpy as np
    rng = np.random.default_rng([0, {"mnist": 0, "fmnist": 1}[ds]])
    pick = [rng.choice(np.sort(ids[y == c]), PER_CLASS, replace=False) for c in np.unique(y)]
    return np.sort(np.concatenate(pick))


def prepare(base, seed):
    import numpy as np, torch
    from hem import arrays, hem_split
    from qnn import pool_reducer
    from data import load_data
    from noise_circuits import qnn_circuit
    from qiskit import qpy
    from qiskit.quantum_info import Statevector
    for s in SETUPS:
        enc, ds, nq, nc = s
        tl, _, te, *_ = load_data(ds, N_TRAIN, N_VAL, N_TEST, BATCH, DATA_SEED, CLASS_ORDER[ds][:nc])
        X, y, ids = arrays(te, flat=False); y, ids = y.numpy(), np.asarray(ids)
        chosen = np.sort(sample_ids(ds, ids, y)); yc = y[np.isin(ids, chosen)]
        keep = np.isin(ids, chosen[device_index(yc)])
        _, hem = hem_split(pool_reducer(enc, nq, nc), enc, nq, nc, tl, te)
        for v in VARIANTS:
          for sd in (seed,):
            f0 = base / "prepare" / key(s, v, sd); f0.parent.mkdir(parents=True, exist_ok=True)
            if f0.with_suffix(".npz").exists():
                continue
            model, ro = NI.build_model(enc, ds, nq, nc, v, sd)
            with torch.no_grad():
                z = model.reducer(X[keep]).double().numpy()
            W = model.qlayer.weights.detach().double().numpy()
            circs = [qnn_circuit(zi, enc, W, measure=False) for zi in z]
            p_ideal = np.array([Statevector(c).probabilities() for c in circs])
            with open(f0.with_suffix(".qpy"), "wb") as f:
                qpy.dump(circs, f)
            np.savez_compressed(f0.with_suffix(".npz"), ids=ids[keep], y=y[keep],
                                ce=np.array([hem[int(i)] < 0 for i in ids[keep]]), encoder_out=z,
                                head_w=model.head.weight.detach().double().numpy(),
                                head_b=model.head.bias.detach().double().numpy(), prob_ideal=p_ideal)
            f0.with_suffix(".json").write_text(json.dumps(dict(encoding=enc, ds=ds, nq=nq, nc=nc, variant=v, seed=sd,
                                                         readout=ro, n_meas=int(model.n_meas), n_samples=int(keep.sum()))))
            print("prepared", key(s, v, sd), int(keep.sum()), "samples", flush=True)


def get_backend(fake, instance):
    if fake:
        from qiskit_ibm_runtime.fake_provider import FakeFez
        return None, FakeFez()
    from qiskit_ibm_runtime import QiskitRuntimeService
    svc = QiskitRuntimeService(instance=instance) if instance else QiskitRuntimeService()
    return svc, svc.backend(BACKEND)


def save_calibration(be, path, when=None):
    props = be.properties(datetime=when) if when is not None and hasattr(be, "service") else be.properties()
    path.write_text(json.dumps(props.to_dict(), default=str))


def transpile_like_5_4_2(be, circs, nq):
    from qiskit import ClassicalRegister, QuantumCircuit, transpile
    first = transpile(circs[0], be, optimization_level=1, seed_transpiler=0)
    layout = NI.qubit_subset(be, first, nq)
    cmap = be.coupling_map.reduce(layout)
    basis = [g for g in ("rz", "sx", "x", "cz", "ecr") if g in be.operation_names]
    cs = transpile(circs, coupling_map=cmap, basis_gates=basis, optimization_level=1, seed_transpiler=0,
                   initial_layout=list(range(nq)))
    wide, physs, n2q = [], [], []
    for c in cs:
        phys = [layout[c.layout.final_index_layout()[i]] for i in range(nq)]
        w = QuantumCircuit(be.num_qubits); w.add_register(ClassicalRegister(nq, "meas"))
        w.compose(c, qubits=layout, inplace=True)
        w.barrier(phys); w.measure(phys, list(range(nq)))
        wide.append(w); physs.append(phys)
        n2q.append(int(sum(v for g, v in c.count_ops().items() if g in ("cz", "ecr", "cx"))))
    return wide, layout, physs, n2q


def submit(base, fake, instance, limit_jobs, resubmit_failed=False, seed=0):
    from qiskit import qpy
    from qiskit_ibm_runtime import Batch, SamplerV2
    svc, be = get_backend(fake, instance)
    sd = base / "submit"; sd.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    save_calibration(be, sd / f"calibration_at_submit_{stamp}.json")
    jobs_f = sd / "jobs.json"
    jobs = json.loads(jobs_f.read_text()) if jobs_f.exists() else {}
    ff = base / "fetch" / "failed.json"
    if resubmit_failed and ff.exists():
        hist_f = sd / "failed_jobs_history.json"
        hist = json.loads(hist_f.read_text()) if hist_f.exists() else []
        for k, f in json.loads(ff.read_text()).items():
            if k in jobs:
                hist.append(dict(key=k, **jobs.pop(k), final_status=f["status"]))
        hist_f.write_text(json.dumps(hist, indent=1)); jobs_f.write_text(json.dumps(jobs, indent=1)); ff.unlink()
    todo = [(s, v, sd_) for s, v, sd_ in items(seed) if key(s, v, sd_) not in jobs]
    if limit_jobs:
        todo = todo[:limit_jobs]
    print(f"{BACKEND if not fake else 'FakeFez'}: {len(todo)} jobs to submit ({len(jobs)} already submitted)", flush=True)
    if not todo:
        return
    batch = None if fake else Batch(backend=be)
    sampler = SamplerV2(mode=be if fake else batch)
    try:
        for s, v, seed in todo:
            k = key(s, v, seed)
            with open((base / "prepare" / k).with_suffix(".qpy"), "rb") as f:
                circs = qpy.load(f)
            import numpy as np
            idx = device_index(np.load((base / "prepare" / k).with_suffix(".npz"))["y"])
            circs = [circs[i] for i in idx]
            wide, layout, physs, n2q = transpile_like_5_4_2(be, circs, s[2])
            fq = (sd / k).with_suffix(".qpy"); fq.parent.mkdir(parents=True, exist_ok=True)
            with open(fq, "wb") as f:
                qpy.dump(wide, f)
            job = sampler.run(wide, shots=SHOTS)
            jobs[k] = dict(job_id=job.job_id(), submitted=dt.datetime.now(dt.timezone.utc).isoformat(),
                             layout=layout, phys=physs, two_qubit_gates=n2q, shots=SHOTS, n=len(wide),
                             sample_index=[int(i) for i in idx])
            jobs_f.write_text(json.dumps(jobs, indent=1))
            if fake:
                save_counts(base, k, job.result(), len(wide))
            print(f"submitted {k}: job {job.job_id()}, layout {layout}, 2q gates {sum(n2q) / len(n2q):.0f}", flush=True)
    finally:
        if batch is not None:
            batch.close()


def counts_file(base, k):
    return (base / "fetch" / k).with_suffix(".json")


def save_counts(base, k, result, n):
    f = counts_file(base, k); f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps([result[i].data.meas.get_counts() for i in range(n)]))


def fetch(base, fake, instance):
    svc, be = get_backend(fake, instance)
    jobs = json.loads((base / "submit" / "jobs.json").read_text())
    cal_dir = base / "fetch" / "calibration"; cal_dir.mkdir(parents=True, exist_ok=True)
    meta_f = base / "fetch" / "job_info.json"
    meta = json.loads(meta_f.read_text()) if meta_f.exists() else {}
    failed = {}
    for key, j in jobs.items():
        if counts_file(base, key).exists():
            continue
        job = svc.job(j["job_id"])
        st = str(job.status())
        if any(x in st.upper() for x in ("ERROR", "CANCEL")):
            failed[key] = dict(job_id=j["job_id"], status=st)
            print(f"{key}: {st} (resubmit with: python scripts/device_fez.py submit --resubmit-failed)", flush=True); continue
        if "DONE" not in st.upper():
            print(f"{key}: {st}", flush=True); continue
        save_counts(base, key, job.result(), j["n"])
        info = dict(status=st)
        try:
            m = job.metrics(); info.update(metrics=m)
            ts = (m.get("timestamps") or {}).get("running") or (m.get("timestamps") or {}).get("finished")
        except Exception as e:
            info.update(metrics_error=str(e)); ts = None
        when = dt.datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else job.creation_date
        info.update(run_time=str(when))
        cf = cal_dir / f"{key.replace('/', '__')}.json"
        save_calibration(be, cf, when)
        info.update(calibration=cf.name); meta[key] = info
        meta_f.write_text(json.dumps(meta, indent=1, default=str))
        print(f"fetched {key} (ran {when})", flush=True)
    (base / "fetch" / "failed.json").write_text(json.dumps(failed, indent=1))
    left = [k for k in jobs if not counts_file(base, k).exists()]
    print(f"{len(jobs) - len(left)} / {len(jobs)} jobs fetched" + (f"; not fetched yet: {len(left)}" if left else "")
          + (f" ({len(failed)} failed on the device)" if failed else ""))


def _analyze_one(args):
    import warnings; warnings.filterwarnings("ignore")
    import numpy as np
    from noise_circuits import counts_to_probs, observables_from_probs
    from qiskit import qpy
    base, key, cal_file, fake = args
    st, v, sdir = key.split("/"); seed = int(sdir.split("_")[1])
    meta = json.loads((base / "prepare" / key).with_suffix(".json").read_text()); P = np.load((base / "prepare" / key).with_suffix(".npz"))
    j = json.loads((base / "submit" / "jobs.json").read_text())[key]
    nq = meta["nq"]
    counts = json.loads(counts_file(base, key).read_text())
    idx = device_index(P["y"])
    run_idx = j.get("sample_index", list(range(len(counts))))
    pos = [run_idx.index(int(i)) for i in idx]
    counts = [counts[q] for q in pos]
    j = dict(j, phys=[j["phys"][q] for q in pos], two_qubit_gates=[j["two_qubit_gates"][q] for q in pos])
    P = {**{k_: P[k_] for k_ in P.files}, **{k_: P[k_][idx] for k_ in ("ids", "y", "ce", "encoder_out", "prob_ideal")}}
    p_dev = np.array([counts_to_probs(c, nq) for c in counts])
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
    from qiskit_ibm_runtime.models import BackendProperties
    cal = json.loads(cal_file.read_text())
    med = {}
    for name in ("T1", "T2", "frequency"):
        vals = [it for qp in cal["qubits"] for it in qp if it["name"] == name]
        med[name] = (float(np.median([it["value"] for it in vals])), vals[0]["unit"]) if vals else (5.0, "GHz")
    filled = {}
    for q, qp in enumerate(cal["qubits"]):
        for name, (val, unit) in med.items():
            if not any(it["name"] == name for it in qp):
                qp.append(dict(name=name, value=val, unit=unit, date=qp[0]["date"] if qp else cal["last_update_date"]))
                filled.setdefault(q, []).append(name)
    used = sorted({q for ph in j["phys"] for q in ph})
    filled_used = {q: v for q, v in filled.items() if q in used and v != ["frequency"]}
    props = BackendProperties.from_dict(cal)
    sim = AerSimulator(method="density_matrix", noise_model=NoiseModel.from_backend_properties(props),
                       max_parallel_threads=1, max_parallel_experiments=1)
    with open((base / "submit" / key).with_suffix(".qpy"), "rb") as f:
        wide = qpy.load(f)
    cs = []
    wide = [wide[q] for q in pos]
    for c, phys in zip(wide, j["phys"]):
        c = c.remove_final_measurements(inplace=False); c.save_probabilities(qubits=phys); cs.append(c)
    p_sim, out = [], []
    for k in range(0, len(cs), 50):
        r_ = sim.run(cs[k:k + 50], shots=1).result(); out += [np.asarray(r_.data(i)["probabilities"]) for i in range(len(cs[k:k + 50]))]
    for p, phys in zip(out, j["phys"]):
        mats = []
        for q in phys:
            p10 = props.qubit_property(q, "prob_meas1_prep0")[0]; p01 = props.qubit_property(q, "prob_meas0_prep1")[0]
            mats.append(np.array([[1 - p10, p01], [p10, 1 - p01]]))
        p_sim.append(NI.apply_readout(p, mats))
    p_sim = np.array(p_sim)
    rng = np.random.default_rng([seed, nq, len(v)])
    p_sim_shots = np.array([rng.multinomial(SHOTS, np.clip(p, 0, None) / np.clip(p, 0, None).sum()) / SHOTS for p in p_sim])
    y, ce = P["y"], P["ce"]

    def score(ps):
        lg = np.array([P["head_w"] @ observables_from_probs(p, nq, meta["readout"], meta["n_meas"]) + P["head_b"] for p in ps])
        ok = lg.argmax(1) == y
        return dict(acc=float(ok.mean()), acc_ce=float(ok[ce].mean()), acc_pe=float(ok[~ce].mean()))

    pi = P["prob_ideal"]; to_u = 0.5 * np.abs(pi - 1 / pi.shape[1]).sum(1)
    tv = lambda q: dict(tvd=float((0.5 * np.abs(pi - q).sum(1)).mean()),
                        tvd_norm=float((0.5 * np.abs(pi - q).sum(1) / to_u).mean()))
    res = dict(meta, key=key, device=dict(**score(p_dev), **tv(p_dev)),
               sim_runtime_cal=dict(**score(p_sim_shots), **tv(p_sim_shots)),
               sim_runtime_cal_exact=dict(**score(p_sim), **tv(p_sim)), ideal=score(pi),
               two_qubit_gates_mean=float(np.mean(j["two_qubit_gates"])), layout=j["layout"], calibration=cal_file.name,
               calibration_filled_qubits=len(filled), calibration_filled_on_used_qubits=filled_used)
    fz = NI.OUT / "fez" / st / v / f"seed_{seed}.npz"
    if fz.exists() and not fake:
        Z = np.load(fz); sel = np.isin(Z["ids"], P["ids"])
        ok = Z["logit_both"][sel].argmax(1) == Z["y"][sel]
        res["fakefez_5_4_2"] = dict(acc=float(ok.mean()), acc_ce=float(ok[Z["ce"][sel]].mean()),
                                    acc_pe=float(ok[~Z["ce"][sel]].mean()))
    np.savez_compressed((base / "fetch" / key).with_name(f"{sdir}_probs.npz"), prob_device=p_dev.astype(np.float32),
                        prob_sim_runtime_cal=p_sim.astype(np.float32))
    return res


def analyze(base, fake, workers):
    from multiprocessing import Pool
    jobs = json.loads((base / "submit" / "jobs.json").read_text())
    info = json.loads((base / "fetch" / "job_info.json").read_text()) if (base / "fetch" / "job_info.json").exists() else {}
    tasks = []
    for key in jobs:
        if not counts_file(base, key).exists():
            continue
        cal = base / "fetch" / "calibration" / info[key]["calibration"] if key in info else \
            sorted((base / "submit").glob("calibration_at_submit_*.json"))[-1]
        tasks.append((base, key, cal, fake))
    with Pool(workers, maxtasksperchild=1) as pool:
        res = pool.map(_analyze_one, tasks)
    (base / "analysis.json").write_text(json.dumps(res, indent=1))
    import numpy as np
    lines = ["| setup | variant | seeds | ideal | device | sim (run-time cal.) | FakeFez (5.4.2) | TVD norm device "
             "| TVD norm sim | 2q gates |", "|---|---|---|---|---|---|---|---|---|---|"]
    ms = lambda v: f"{100 * np.mean(v):.1f} ± {100 * np.std(v):.1f}" if len(v) else ""
    for st in sorted({r["key"].split("/")[0] for r in res}):
        for v in VARIANTS:
            rs = [r for r in res if r["key"].split("/")[0] == st and r["variant"] == v]
            if not rs:
                continue
            g = lambda f: [f(r) for r in rs]
            lines.append(f"| {st} | {v} | {len(rs)} | {ms(g(lambda r: r['ideal']['acc']))} | {ms(g(lambda r: r['device']['acc']))} "
                         f"| {ms(g(lambda r: r['sim_runtime_cal']['acc']))} | {ms([r['fakefez_5_4_2']['acc'] for r in rs if 'fakefez_5_4_2' in r])} "
                         f"| {np.mean(g(lambda r: r['device']['tvd_norm'])):.3f} | {np.mean(g(lambda r: r['sim_runtime_cal']['tvd_norm'])):.3f} "
                         f"| {np.mean(g(lambda r: r['two_qubit_gates_mean'])):.0f} |")
    (base / "analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("prepare", "submit", "fetch", "analyze"))
    ap.add_argument("--fake", action="store_true", help="FakeFez with the local sampler (test without the device)")
    ap.add_argument("--instance", default=None); ap.add_argument("--limit-jobs", type=int, default=0)
    ap.add_argument("--resubmit-failed", action="store_true", help="submit again the jobs that fetch found failed")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--setups", default="", help="e.g. amplitude_mnist_4q4c,angle_mnist_4q4c (default: the paper's)")
    ap.add_argument("--seed", type=int, default=None, help="model seed (default: the first trained seed)")
    a = ap.parse_args()
    if a.setups:
        SETUPS[:] = [s for s in NI.SETUPS if NI.tag(*s) in set(a.setups.split(","))]
    base = out_dir(a.fake)
    if a.step == "prepare":
        prepare(out_dir(False), a.seed if a.seed is not None else first_seed())
        if a.fake:
            import shutil
            shutil.copytree(out_dir(False) / "prepare", base / "prepare", dirs_exist_ok=True)
    elif a.step == "submit":
        submit(base, a.fake, a.instance, a.limit_jobs, a.resubmit_failed, a.seed if a.seed is not None else first_seed())
    elif a.step == "fetch":
        fetch(base, a.fake, a.instance)
    else:
        analyze(base, a.fake, a.workers)
