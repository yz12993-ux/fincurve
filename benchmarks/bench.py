"""Wall-clock benchmark of representative fincurve workloads.

    python benchmarks/bench.py            # all workloads
    python benchmarks/bench.py smile path # substring filter
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import scenarios as S  # noqa: E402
from fincurve import analyze  # noqa: E402


def large_curve(rng, n=3000):
    x = rng.uniform(0.5, 30, n)
    return pd.Series(x, name="size"), pd.Series((3 + 2 * np.log(x)) * np.exp(rng.normal(0, 0.2, n)), name="cost")


def large_additive(rng, n=10000):
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.uniform(0, 5, n), "c": rng.exponential(size=n) + 0.1,
                      "d": rng.normal(size=n), "cat": rng.choice(list("ABCDE"), n)})
    y = 1 + np.sin(X.a) + np.log(X.c) + 0.5 * X.b + (X.cat == "C") * 0.7 + rng.normal(0, 0.5, n)
    return X, pd.Series(y, name="y")


WORKLOADS = [
    ("curve_loo_n13", lambda r: S.yield_curve(r), {}),
    ("curve_smile_n25", lambda r: S.vol_smile(r), {}),
    ("curve_two_families_n90", lambda r: S.market_impact(r), {}),
    ("curve_binomial_n16", lambda r: S.default_rate(r), {}),
    ("curve_poisson_n400", lambda r: S.trade_counts(r), {}),
    ("curve_path_n121", lambda r: S.price_path(r), {}),
    ("curve_large_n3000", large_curve, {}),
    ("distribution_n2520", lambda r: (S.daily_returns(r),), {}),
    ("additive_lognormal_n1500", lambda r: S.credit_spreads(r), {}),
    ("additive_binary_n4000", lambda r: S.loan_defaults(r), {}),
    ("additive_gaussian_n10000", large_additive, {}),
]


def run(name, make, kw):
    out = make(np.random.default_rng(7))
    args, extra = (out[:2], {"trials": out[2]}) if len(out) == 3 else (out, {})
    start = time.perf_counter()
    rep = analyze(*args, **extra, **kw)
    elapsed = time.perf_counter() - start
    rec = rep.recommended_id
    return elapsed, rec


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    only = sys.argv[1:]
    results = {}
    for name, make, kw in WORKLOADS:
        if only and not any(o in name for o in only):
            continue
        elapsed, rec = run(name, make, kw)
        results[name] = {"seconds": round(elapsed, 3), "recommended": rec}
        print(f"{name:28s} {elapsed:8.2f}s   recommended={rec}", flush=True)
    print(json.dumps(results))
