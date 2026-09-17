"""Fitting engines.

Most candidate curves are separable: mean = B(x, θ) @ β, where β are linear
coefficients and θ holds at most a few nonlinear parameters. For those we use
variable projection — β is solved exactly (weighted least squares, IRLS or
Gauss–Newton depending on the likelihood) and the optimiser only searches θ.
That removes the expensive finite-difference Jacobians over β and makes
purely linear forms a single matrix solve. Non-separable forms fall back to
multi-start bounded least squares over all parameters.
"""
import warnings

import numpy as np
from scipy.linalg import lstsq as _lstsq
from scipy.optimize import least_squares, lsq_linear


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
    """General path: multi-start least squares over every parameter. Returns (params, cost)."""
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


# ----------------------------------------------------------------------------- linear algebra
def wls(A, b, w, bounds=None):
    """Weighted least squares, optionally with box constraints on the coefficients."""
    sw = np.sqrt(np.maximum(w, 0.0))
    As, bs = A * sw[:, None], b * sw
    if bounds is None:
        coef = _lstsq(As, bs, lapack_driver="gelsy", check_finite=False)[0]
    else:
        coef = lsq_linear(As, bs, bounds=bounds, method="bvls").x
    return coef


def _objective(family, y, w, mu):
    with np.errstate(all="ignore"):
        r = family.fit_residuals(y, mu, w)
    r = np.where(np.isfinite(r), r, 1e8)
    return float(r @ r)


def _eta_limits(family):
    return (-35.0, 35.0) if family.name == "binomial" else (-700.0, 700.0)


def inner_fit(B, y, w, family, link=False, bounds=None, beta0=None, max_iter=40, tol=1e-10):
    """Linear coefficients that maximise the family likelihood for a fixed basis B.

    link=False: mean = B @ β (Gaussian: one solve; log-normal: Gauss–Newton; Poisson and
    binomial: identity-link IRLS). link=True: mean = inverse_link(B @ β), canonical IRLS.
    Every step is halved until the objective does not increase. Returns (β, objective).
    """
    lo, hi = _eta_limits(family)
    to_mean = (lambda eta: family.inverse_link(np.clip(eta, lo, hi))) if link else (lambda eta: eta)
    name = family.name
    if not link and name in ("binomial", "poisson"):
        # Identity-scale means get clipped into the valid range, which makes the likelihood flat and
        # non-convex; extra iterations buy little, and these forms are poor fits for such targets anyway.
        max_iter = min(max_iter, 15)
    with np.errstate(all="ignore"):
        if name == "gaussian" and not link:
            beta = wls(B, y, w, bounds)
            return beta, _objective(family, y, w, B @ beta)
        if beta0 is not None and np.all(np.isfinite(beta0)):
            beta = np.asarray(beta0, float)
        elif link:
            beta = wls(B, family.working_response(y), w, bounds)
        elif name == "lognormal":
            beta = wls(B, y, w / y ** 2, bounds)
        elif name == "poisson":
            beta = wls(B, y, w / np.maximum(y, 0.5), bounds)
        else:
            beta = wls(B, y, w, bounds)
        obj = _objective(family, y, w, to_mean(B @ beta))
        for _ in range(max_iter):
            eta = B @ beta
            mu = family.clip(to_mean(eta))
            shifted = None if bounds is None else (np.asarray(bounds[0]) - beta, np.asarray(bounds[1]) - beta)
            if link and name == "binomial":
                v = np.maximum(mu * (1 - mu), 1e-12)
                new = wls(B, eta + (y - mu) / v, w * v, bounds)
            elif link:
                new = wls(B, eta + (y - mu) / mu, w * mu, bounds)
            elif name == "lognormal":
                if np.any(mu <= 1e-300):
                    break
                new = beta + wls(B / mu[:, None], np.log(y) - np.log(mu), w, shifted)
            elif name == "poisson":
                new = wls(B, y, w / mu, bounds)
            else:
                new = wls(B, y, w / np.maximum(mu * (1 - mu), 1e-6), bounds)
            if not np.all(np.isfinite(new)):
                break
            step = new - beta
            for _ in range(20):
                cand = beta + step
                cobj = _objective(family, y, w, to_mean(B @ cand))
                if cobj <= obj:
                    break
                step = step / 2
            else:
                break
            gain = obj - cobj
            beta, obj = cand, cobj
            if gain <= tol * max(obj, 1e-300):
                break
    return beta, obj


# ----------------------------------------------------------------------------- variable projection
def _proxy_problem(family, y, w, link):
    """Weighted least-squares stand-in for the likelihood, used to screen nonlinear grids."""
    with np.errstate(all="ignore"):
        if link:
            return family.working_response(y), w
        if family.name == "lognormal":
            return y, w / y ** 2
        if family.name == "poisson":
            return y, w / np.maximum(y, 0.5)
        if family.name == "binomial":
            return y, w / np.maximum(y * (1 - y), 1e-3)
    return y, w


def fit_separable(c, x, y, w, family, theta_starts=None, use_grid=True, n_refine=2, max_nfev=None):
    """Variable-projection fit of a separable candidate. Returns (params, theta, cost) or (None, None, inf)."""
    link = bool(getattr(c, "link", False))
    lo, hi = _eta_limits(family)
    lb, ub = as_bounds(c.nl_bounds, c.q)
    iterative = link or family.name != "gaussian"
    state = {"beta": None}

    def solve(theta):
        with np.errstate(all="ignore"):
            B = np.column_stack([np.broadcast_to(col, x.shape) for col in c.basis(x, theta)])
        if not np.all(np.isfinite(B)):
            return None, np.inf, None
        beta, obj = inner_fit(B, y, w, family, link, c.coef_bounds, state["beta"] if iterative else None)
        if np.all(np.isfinite(beta)) and np.isfinite(obj):
            state["beta"] = beta
        return beta, obj, B

    if c.q == 0:
        beta, obj, _ = solve(())
        if beta is None or not np.isfinite(obj):
            return None, None, np.inf
        return list(c.assemble(beta, ())), (), obj / 2

    def resid(theta):
        beta, _, B = solve(theta)
        if beta is None:
            return np.full(y.shape, 1e4)
        with np.errstate(all="ignore"):
            eta = B @ beta
            mu = family.inverse_link(np.clip(eta, lo, hi)) if link else eta
            r = family.fit_residuals(y, mu, w)
        return np.where(np.isfinite(r), r, 1e8)

    starts = [tuple(t) for t in (theta_starts or [])]
    grid = [tuple(t) for t in c.nl_grid(x, y)] if use_grid else []
    # Screen the grid with a one-solve weighted least-squares proxy of the likelihood; only the
    # most promising settings get the full iterative inner fit during refinement.
    target, pw = _proxy_problem(family, y, w, link)
    scored = []
    for t in grid:
        with np.errstate(all="ignore"):
            B = np.column_stack([np.broadcast_to(col, x.shape) for col in c.basis(x, t)])
            if not np.all(np.isfinite(B)):
                continue
            beta = wls(B, target, pw, c.coef_bounds)
            r = (target - B @ beta) * np.sqrt(pw)
            obj = float(r @ r)
        if np.isfinite(obj):
            scored.append((obj, t))
    scored.sort(key=lambda s: s[0])
    queue = starts + [t for _, t in scored][:n_refine]
    if not queue:
        return None, None, np.inf

    best_theta, best_cost = None, np.inf
    warm = not use_grid
    max_nfev = max_nfev or (15 if warm else 60) * (c.q + 1)
    tol = 1e-6 if warm else 1e-8
    for t0 in queue:
        t0 = _inside(t0, lb, ub)
        state["beta"] = None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sol = least_squares(resid, t0, bounds=(lb, ub), x_scale="jac", max_nfev=max_nfev,
                                    diff_step=1e-6 if iterative else None, ftol=tol, xtol=tol)
        except Exception:
            continue
        if np.isfinite(sol.cost) and sol.cost < best_cost:
            best_theta, best_cost = sol.x, sol.cost
    if best_theta is None:
        return None, None, np.inf
    state["beta"] = None
    beta, obj, _ = solve(best_theta)
    if beta is None:
        return None, None, np.inf
    return list(c.assemble(beta, tuple(best_theta))), tuple(float(v) for v in best_theta), obj / 2
