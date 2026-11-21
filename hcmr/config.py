from pathlib import Path
import secrets

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

N_TRAIN, N_VAL, N_TEST = 400, 100, 200
DATA_SEED = 42
CLASS_ORDER = {
    "mnist":   (0, 3, 6, 8, 1, 2, 4, 5, 7, 9),
    "fmnist":  (0, 1, 5, 9, 2, 3, 4, 6, 7, 8),
    "cifar10": (0, 1, 5, 9, 2, 3, 4, 6, 7, 8),
}
SIDE = {"mnist": 28, "fmnist": 28, "cifar10": 32}

SETUPS = [(e, d, q, c) for q in (4, 6, 8, 10) for e in ("amplitude", "angle") for d in ("mnist", "fmnist", "cifar10")
          for c in (4, 10)]

BATCH, LR = 32, 1e-3
T_EPOCHS = 60
FIXED_EPOCHS = 50
J_GRID = (0, 10, 30, 60)

FIT_STEPS, FIT_LR = 400, 5e-3
MUS, LAMS = (0.0, 0.1, 0.3), (0.0, 0.1, 0.2, 0.5)
MARGIN_GRID = [(m, l) for m in MUS for l in LAMS if (m, l) != (0.0, 0.0)]

N_SEEDS = 5


def tag(enc, ds, nq, nc):
    return f"{enc}_{ds}_{nq}q{nc}c"


def parse_seeds(text, n=N_SEEDS):
    if text:
        return tuple(int(s) for s in text.split(","))
    return tuple(secrets.randbelow(2 ** 31) for _ in range(n))
