"""Numerical kernels: the compiled C extension when it is available, NumPy otherwise.

Set the environment variable FINCURVE_NO_C=1 to force the NumPy implementations.
"""
import os

import numpy as np

try:
    if os.environ.get("FINCURVE_NO_C"):
        raise ImportError
    from . import _kernels
except ImportError:
    _kernels = None

HAVE_C = _kernels is not None


def neighbours(n, frac):
    return int(min(n, max(np.ceil(frac * n), 5)))


def _local_linear_numpy(x, y, x_eval, m, loo):
    n = x.size
    span = np.ptp(x) or 1.0
    out = np.empty(x_eval.size)
    for i, x0 in enumerate(x_eval):
        d = np.abs(x - x0)
        h = np.partition(d, m - 1)[m - 1]
        if h <= 0:
            h = d.max() if d.max() > 0 else 1.0
        k = np.clip(1 - (d / (h * 1.0001)) ** 3, 0, None) ** 3
        if loo:
            k[i] = 0.0
        sw = k.sum()
        if sw <= 0:
            out[i] = np.nan
            continue
        xm, ym = k @ x / sw, k @ y / sw
        sxx = k @ (x - xm) ** 2
        b = (k @ ((x - xm) * (y - ym))) / sxx if sxx > 1e-12 * sw * span * span else 0.0
        out[i] = ym + b * (x0 - xm)
    return out


def local_linear(x, y, x_eval=None, frac=0.3, loo=False, force_numpy=False):
    """Tricube-weighted local linear regression (LOWESS without robustness iterations).

    With loo=True (only when evaluating at x itself) each point is predicted without its own weight.
    """
    x = np.ascontiguousarray(x, dtype=float)
    y = np.ascontiguousarray(y, dtype=float)
    loo = bool(loo and x_eval is None)
    m = neighbours(x.size, frac)
    if _kernels is None or force_numpy:
        xe = x if x_eval is None else np.asarray(x_eval, dtype=float)
        return _local_linear_numpy(x, y, xe, m, loo)
    order = np.argsort(x, kind="stable")
    xs, ys = np.ascontiguousarray(x[order]), np.ascontiguousarray(y[order])
    if x_eval is None:
        sorted_out = np.empty(x.size)
        _kernels.local_linear(xs, ys, xs, sorted_out, m, loo)
        out = np.empty(x.size)
        out[order] = sorted_out
        return out
    xe = np.ascontiguousarray(np.ravel(x_eval), dtype=float)
    out = np.empty(xe.size)
    _kernels.local_linear(xs, ys, xe, out, m, False)
    return out
