"""Automatic description of the data before any curve is fitted.

The profile drives the "smart" choices: which error model to use, whether the
data should be cross-validated as a time series, and which warnings to raise.
"""
import re

import numpy as np
from scipy import stats

from .kernels import local_linear

TIME_NAME = re.compile(r"\b(date|datetime|time|timestamp|year|month|week|day|period)s?\b", re.I)
TERM_NAME = re.compile(r"^(?!.*\b(yield|ytm)\b).*\b(tenor|maturity|term|expiry|ttm|horizon)", re.I)
CONTEXTS = [
    # (regex on the x name, label, library groups that carry the matching finance meaning)
    (TERM_NAME, "term structure", ("term_structure", "credit")),
    (re.compile(r"(strike|moneyness|delta|K/F)", re.I), "volatility smile", ("volatility",)),
    (re.compile(r"\b(yield|ytm|interest rate|discount rate|coupon rate|swap rate)\b", re.I), "price–yield relationship", ("fixed_income",)),
    (re.compile(r"(score|rating|fico)", re.I), "credit-score relationship", ("sigmoid", "credit")),
    (re.compile(r"(spot|underlying)", re.I), "option–underlying relationship", ("options",)),
]

TARGET_KINDS = {
    "binary": "binary 0/1 outcome",
    "proportion": "proportion / probability (0–1)",
    "count": "non-negative integer count",
    "positive": "positive continuous value",
    "real": "continuous value of either sign",
}
SHAPES = {
    "flat": "flat",
    "increasing": "increasing",
    "decreasing": "decreasing",
    "peak": "rises then falls (peak)",
    "valley": "falls then rises (valley)",
    "wavy": "oscillating",
}
PATTERNS = {
    "flattening": "flattening (saturation / diminishing returns)",
    "steepening": "steepening (accelerating)",
    "steady": "roughly constant slope",
    "s_shaped": "flat ends and a steep middle (S-shape)",
}


def acf1(r):
    r = np.asarray(r, float) - np.mean(r)
    den = np.sum(r * r)
    return float(np.sum(r[1:] * r[:-1]) / den) if den > 0 else 0.0


def choose_frac(x, y, fracs=(0.1, 0.2, 0.3, 0.5, 0.8)):
    """Bandwidth with the smallest leave-one-out error (subsampled for large n)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size > 6000:
        idx = np.random.default_rng(0).choice(x.size, 6000, replace=False)
        x, y = x[idx], y[idx]
    scores = {}
    for f in fracs:
        pred = local_linear(x, y, frac=f, loo=True)
        ok = np.isfinite(pred)
        scores[f] = np.mean((y[ok] - pred[ok]) ** 2) if ok.sum() > 3 else np.inf
    return min(scores, key=scores.get)


def smooth_curve(x, y, frac=0.3):
    """Local-linear smooth evaluated at x; subsamples and interpolates for large n."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size <= 20000:
        return local_linear(x, y, frac=frac)
    idx = np.random.default_rng(0).choice(x.size, 20000, replace=False)
    grid = np.unique(np.quantile(x, np.linspace(0, 1, 200)))
    return np.interp(x, grid, local_linear(x[idx], y[idx], grid, frac))


def describe_target(y, trials=None):
    y = np.asarray(y, float)
    uniq = np.unique(y)
    integer = bool(np.all(np.isclose(y, np.round(y))))
    if uniq.size <= 2 and set(uniq.tolist()) <= {0.0, 1.0}:
        kind = "binary"
    elif trials is not None:
        kind = "proportion"
    elif integer and y.min() >= 0 and uniq.size > 2:
        kind = "count"
    elif y.min() >= 0 and y.max() <= 1:
        kind = "proportion"
    elif y.min() > 0:
        kind = "positive"
    else:
        kind = "real"
    return {
        "kind": kind, "n": int(y.size), "n_unique": int(uniq.size),
        "min": float(y.min()), "max": float(y.max()), "mean": float(y.mean()), "std": float(y.std()),
        "skew": float(stats.skew(y)) if y.size > 2 else 0.0,
        "excess_kurtosis": float(stats.kurtosis(y)) if y.size > 3 else 0.0,
        "max_min_ratio": float(y.max() / y.min()) if y.min() > 0 else np.inf,
    }


def describe_relation(x, y, name=None):
    """Shape, noise and ordering diagnostics of y against a single x."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = x.size
    o = np.argsort(x, kind="stable")
    xs, ys = x[o], y[o]
    frac = choose_frac(xs, ys) if n >= 8 else 0.8
    sm = smooth_curve(xs, ys, frac)
    sm_shape = sm if frac >= 0.25 else smooth_curve(xs, ys, 0.25)
    resid = ys - sm
    out = {"n": n, "frac": frac}

    rho, p = stats.spearmanr(xs, ys) if np.ptp(ys) > 0 and np.ptp(xs) > 0 else (0.0, 1.0)
    out["spearman"], out["spearman_p"] = float(rho), float(p)

    nb = int(min(12, max(3, n // 4)))
    bins = np.array_split(np.arange(n), nb)
    yb = np.array([sm_shape[b].mean() for b in bins])
    xb = np.array([xs[b].mean() for b in bins])
    span = np.ptp(yb) or 1.0
    dy = np.diff(yb)
    sig = np.sign(np.where(np.abs(dy) < 0.03 * span, 0, dy))
    nz = sig[sig != 0]
    turns = int(np.sum(nz[1:] != nz[:-1])) if nz.size > 1 else 0
    if nz.size == 0:
        shape = "flat"
    elif turns == 0:
        shape = "increasing" if nz[0] > 0 else "decreasing"
    elif turns == 1:
        shape = "peak" if nz[0] > 0 else "valley"
    else:
        shape = "wavy"
    out["shape"], out["n_turns"] = shape, turns

    pattern = None
    dxb = np.diff(xb)
    if shape in ("increasing", "decreasing") and nb >= 6 and np.all(dxb > 0):
        slope = np.abs(dy / dxb)
        third = max(1, len(slope) // 3)
        first, last = slope[:third].mean(), slope[-third:].mean()
        middle = slope[third:-third].mean() if len(slope) > 2 * third else np.nan
        if np.isfinite(middle) and middle > 0 and first < 0.35 * middle and last < 0.35 * middle:
            pattern = "s_shaped"
        elif first > 0:
            ratio = last / first
            pattern = "flattening" if ratio < 0.3 else "steepening" if ratio > 3 else "steady"
    out["pattern"] = pattern

    out["smooth_r2"] = float(1 - np.var(resid) / np.var(ys)) if np.var(ys) > 0 else 0.0
    if n >= 15 and np.ptp(sm) > 0:
        h_rho, h_p = stats.spearmanr(sm, np.abs(resid))
    else:
        h_rho, h_p = 0.0, 1.0
    out["hetero_rho"], out["hetero_p"] = float(h_rho), float(h_p)
    out["resid_lag1"] = acf1(resid) if n >= 10 else 0.0

    dx = np.diff(xs)
    out["equally_spaced"] = bool(n >= 10 and np.all(dx > 0) and np.std(dx) < 0.05 * np.mean(dx))
    out["max_gap_share"] = float(dx.max() / np.ptp(xs)) if n > 1 and np.ptp(xs) > 0 else 1.0
    mad = stats.median_abs_deviation(resid, scale="normal")
    out["n_outliers"] = int(np.sum(np.abs(resid - np.median(resid)) > 5 * mad)) if mad > 0 else 0
    out["resid_excess_kurtosis"] = float(stats.kurtosis(resid)) if n >= 20 else 0.0
    out["time_like_name"] = bool(name is not None and TIME_NAME.search(str(name)))
    out["term_like_name"] = bool(name is not None and TERM_NAME.search(str(name)))
    out["context"] = next(((label, groups) for rx, label, groups in CONTEXTS
                           if name is not None and rx.search(str(name))), None)
    out["_smooth"] = (xs, sm)
    return out


def random_walk_like(y_in_time_order):
    """Does an ordered series look like a price path (a random walk) rather than a curve plus noise?"""
    y = np.asarray(y_in_time_order, float)
    if y.size < 30:
        return False, {}
    z = np.log(y) if np.all(y > 0) else y
    dz = np.diff(z)
    if np.std(dz) == 0:
        return False, {}
    phi = float(np.corrcoef(z[:-1], z[1:])[0, 1])
    r1 = acf1(dz)
    vr = {q: float(np.var(z[q:] - z[:-q]) / (q * np.var(dz))) for q in (2, 4, 8) if z.size > 3 * q}
    # A smooth curve plus noise leaves nearly independent residuals around a moderate smoother;
    # a random walk wanders away from any smoother, so its residuals stay strongly autocorrelated.
    t = np.arange(z.size, dtype=float)
    resid_acf = acf1(z - local_linear(t, z, frac=0.25))
    flag = (phi > 0.9 and abs(r1) < 0.3 and all(0.5 < v < 2.0 for v in vr.values()) and resid_acf > 0.5)
    return flag, {"ar1": phi, "diff_acf1": r1, "variance_ratios": vr, "resid_acf1": resid_acf,
                  "log_scale": bool(np.all(y > 0))}


def ljung_box(x, lags=10):
    x = np.asarray(x, float) - np.mean(x)
    n = x.size
    den = np.sum(x * x)
    if den == 0 or n <= lags + 1:
        return 0.0, 1.0
    acf = np.array([np.sum(x[k:] * x[:-k]) / den for k in range(1, lags + 1)])
    q = n * (n + 2) * np.sum(acf**2 / (n - np.arange(1, lags + 1)))
    return float(q), float(stats.chi2.sf(q, lags))
