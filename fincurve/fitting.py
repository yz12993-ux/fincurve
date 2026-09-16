"""Multi-start bounded least squares on family-specific residuals."""
import warnings

import numpy as np
from scipy.optimize import least_squares


def as_bounds(bounds, k):
    if bounds is None:
        return np.full(k, -np.inf), np.full(k, np.inf)
    lb, ub = (np.asarray(b, float) for b in bounds)
    return lb, ub


def _inside(p0, lb, ub):
    p = np.asarray(p0, float).copy()
    p[~np.isfinite(p)] = 0.0
    for i in range(p.size):
        lo, hi = lb[i], ub[i]
        pad = 1e-6 * (abs(p[i]) + 1)
        if np.isfinite(lo) and np.isfinite(hi):
            pad = min(pad, (hi - lo) / 4)
        if p[i] <= lo:
            p[i] = lo + pad
        if p[i] >= hi:
            p[i] = hi - pad
    return p


def jitter(starts, n_extra, rng):
    """Add randomly perturbed copies of the first start to escape local minima."""
    out = [np.asarray(s, float) for s in starts]
    if not out or n_extra <= 0:
        return out
    base = out[0]
    for _ in range(n_extra):
        noise = rng.standard_normal(base.size)
        out.append(base * (1 + 0.5 * noise) + 0.1 * noise * (np.abs(base) < 1e-8))
    return out


def fit_params(func, x, y, w, family, starts, bounds=None, max_nfev=None):
    """Return (best_params, cost) over all starts, or (None, inf) if every start fails."""
    if not starts:
        return None, np.inf
    k = len(starts[0])
    lb, ub = as_bounds(bounds, k)
    max_nfev = max_nfev or 150 * (k + 1)

    def resid(p):
        with np.errstate(all="ignore"):
            r = family.fit_residuals(y, func(x, *p), w)
        return np.where(np.isfinite(r), r, 1e8)

    best, best_cost = None, np.inf
    for p0 in starts:
        p0 = _inside(p0, lb, ub)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol = least_squares(resid, p0, bounds=(lb, ub), x_scale="jac", max_nfev=max_nfev)
        except Exception:
            continue
        if np.isfinite(sol.cost) and sol.cost < best_cost:
            best, best_cost = sol.x, sol.cost
    return best, best_cost
