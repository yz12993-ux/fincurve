"""Single-x mode: fit every applicable candidate and rank by held-out likelihood."""
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats

from .families import get_family
from .fitting import fit_params, fit_separable, jitter
from .parallel import pmap, resolve_jobs
from .library import GROUPS, Scale, build_library, domain_ok, link_versions
from .kernels import local_linear
from .profile import acf1
from .scoring import rank_table


@dataclass
class FittedCurve:
    id: str
    candidate: object
    family: object
    params: np.ndarray
    scale: object
    theta: tuple = ()

    @property
    def label(self):
        return self.candidate.label

    def location(self, x):
        with np.errstate(all="ignore"):
            return self.family.clip(self.candidate.func(np.asarray(x, float), *self.params))

    def predict(self, x):
        """Expected value of y at x."""
        return self.family.mean(self.location(x), self.scale)

    def param_dict(self):
        return dict(zip(self.candidate.params, (float(v) for v in self.params)))


def _selected(c, include, exclude):
    tags = {c.key, c.group}
    if include and not tags & set(include):
        return False
    return not (exclude and tags & set(exclude))


def _safe_init(c, x, y):
    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            return [list(s) for s in c.init(x, y) if len(s) == c.k]
    except Exception:
        return []


def _fit(c, fam, x, y, w, n_starts, rng, theta=None):
    """Fit a candidate; returns (params, theta). A theta warm start skips the grid search."""
    if c.separable:
        if theta is not None:
            p, th, _ = fit_separable(c, x, y, w, fam, theta_starts=[theta] if c.q else None,
                                     use_grid=False, n_refine=1)
        else:
            p, th, _ = fit_separable(c, x, y, w, fam)
        return p, th
    return None, None


def _cross_validate(c, fam, x, y, w, splits, p_full, theta_full):
    n = x.size
    test_nll, test_pred = np.full(n, np.nan), np.full(n, np.nan)
    for tr, te in splits:
        if c.separable:
            pt, _ = _fit(c, fam, x[tr], y[tr], w[tr], 0, None, theta=theta_full)
        else:
            starts = [p_full] + ([] if len(splits) > 8 else _safe_init(c, x[tr], y[tr])[:1])
            pt, _ = fit_params(c.func, x[tr], y[tr], w[tr], fam, starts, c.bounds, max_nfev=40 * (c.k + 1))
        if pt is None:
            return test_nll, test_pred, False
        with np.errstate(all="ignore"):
            mu_tr = fam.clip(c.func(x[tr], *pt))
            mu_te = fam.clip(c.func(x[te], *pt))
        if not (np.all(np.isfinite(mu_tr)) and np.all(np.isfinite(mu_te))):
            return test_nll, test_pred, False
        scale = fam.scale(y[tr], mu_tr, w[tr])
        test_nll[te] = fam.nll(y[te], mu_te, scale, w[te])
        test_pred[te] = fam.mean(mu_te, scale)
    return test_nll, test_pred, True


def _smooth_predict(xt, yt, xe, frac):
    if xt.size > 20000:
        idx = np.random.default_rng(0).choice(xt.size, 20000, replace=False)
        xt, yt = xt[idx], yt[idx]
    return local_linear(xt, yt, xe, frac)


def _reference(fam, x, y, w, splits, frac):
    """Non-parametric local-linear smoother: tells us whether the library misses a shape."""
    to_link, from_link = (np.log, np.exp) if fam.name == "lognormal" else (lambda v: v, lambda v: v)
    test_nll, test_pred = np.full(x.size, np.nan), np.full(x.size, np.nan)
    for tr, te in splits:
        g = to_link(y[tr])
        mu_tr = fam.clip(from_link(_smooth_predict(x[tr], g, x[tr], frac)))
        mu_te = fam.clip(from_link(_smooth_predict(x[tr], g, x[te], frac)))
        scale = fam.scale(y[tr], mu_tr, w[tr])
        test_nll[te] = fam.nll(y[te], mu_te, scale, w[te])
        test_pred[te] = fam.mean(mu_te, scale)
    return test_nll, test_pred


def _evaluate(c, fam, x, y, w, splits, n_starts, seed):
    """Fit one candidate on all rows and in every fold. Returns a picklable result."""
    rng = np.random.default_rng(seed)
    n = x.size
    theta = ()
    if c.separable:
        p, theta = _fit(c, fam, x, y, w, n_starts, rng)
    else:
        starts = _safe_init(c, x, y)
        p, _ = fit_params(c.func, x, y, w, fam, jitter(starts, n_starts, rng), c.bounds) if starts else (None, 0)
    if p is None:
        return {"status": "fit failed"}
    with np.errstate(all="ignore"):
        mu = fam.clip(c.func(x, *p))
    if not np.all(np.isfinite(mu)):
        return {"status": "fit failed"}
    scale = fam.scale(y, mu, w)
    nll = float(np.sum(fam.nll(y, mu, scale, w)))
    pred = fam.mean(mu, scale)
    k = c.k + fam.extra_params
    out = {"status": "ok", "params": np.asarray(p, float), "theta": theta, "scale": scale,
           "aic": 2 * k + 2 * nll, "bic": k * np.log(n) + 2 * nll,
           "r2": float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)) if np.ptp(y) > 0 else np.nan}
    tn, tp, ok = _cross_validate(c, fam, x, y, w, splits, p, theta)
    if ok:
        out.update(test_nll=tn, cv_rmse=float(np.sqrt(np.nanmean((y - tp) ** 2))))
    else:
        out["status"] = "prediction failed in cross-validation (usually extrapolation outside the domain)"
    return out


def _evaluate_remote(payload):
    key, family_name, x, y, w, splits, n_starts, seed = payload
    sc, fam = Scale.from_x(x), get_family(family_name)
    c = next(c for c in build_library(sc) + link_versions(sc, fam) if c.key == key)
    return _evaluate(c, fam, x, y, w, splits, n_starts, seed)


def run_race(x, y, w, families, splits, n_starts=2, include=None, exclude=None, seed=0, frac=0.3, n_jobs=1):
    sc = Scale.from_x(x)
    base = build_library(sc)
    multi = len(families) > 1
    tasks, skipped = [], []
    for fam in families:
        for c in base + link_versions(sc, fam):
            if not _selected(c, include, exclude):
                continue
            if not domain_ok(c, x, y):
                skipped.append(c.key)
                continue
            tasks.append((c, fam, f"{c.key}@{fam.name}" if multi else c.key))

    seeds = [seed * 1_000_003 + i for i in range(len(tasks))]
    workers = resolve_jobs(n_jobs, len(tasks))
    if workers > 1:
        results = pmap(_evaluate_remote, [(c.key, fam.name, x, y, w, splits, n_starts, s)
                                          for (c, fam, _), s in zip(tasks, seeds)], workers)
    else:
        results = [_evaluate(c, fam, x, y, w, splits, n_starts, s) for (c, fam, _), s in zip(tasks, seeds)]

    rows, test_nll, fitted = [], {}, {}
    for (c, fam, mid), res in zip(tasks, results):
        row = {"id": mid, "model": c.key, "label": c.label, "family": fam.name, "group": GROUPS[c.group],
               "formula": c.formula, "k": c.k + fam.extra_params, "status": res["status"]}
        rows.append(row)
        if "params" not in res:
            continue
        row.update(aic=res["aic"], bic=res["bic"], r2=res["r2"])
        fitted[mid] = FittedCurve(mid, c, fam, res["params"], res["scale"], res["theta"])
        if "test_nll" in res:
            test_nll[mid] = res["test_nll"]
            row["cv_rmse"] = res["cv_rmse"]

    reference_ids = []
    for fam in families:
        rid = f"smoother@{fam.name}" if multi else "smoother"
        tn, tp = _reference(fam, x, y, w, splits, frac)
        test_nll[rid] = tn
        reference_ids.append(rid)
        rows.append({"id": rid, "model": "smoother", "label": "Local-linear smoother (reference)", "family": fam.name,
                     "group": "reference", "formula": "local linear regression", "k": np.nan, "status": "reference",
                     "cv_rmse": float(np.sqrt(np.nanmean((y - tp) ** 2)))})

    table, best, recommended = rank_table(rows, test_nll, w, reference_ids)
    if "aic" in table:
        table["d_aic"] = table["aic"] - table["aic"].min()
    return {"table": table, "models": fitted, "best": best, "recommended": recommended,
            "test_nll": test_nll, "reference_ids": reference_ids, "skipped": sorted(set(skipped)), "scale": sc}


def extrapolation_spread(models, x, y, share=0.25):
    """Largest disagreement between models outside the data range, as a share of y's range."""
    lo, hi = x.min(), x.max()
    r = hi - lo or 1.0
    left = lo - share * r
    if lo >= 0:
        left = max(left, 0.0)      # nonnegative variables (tenor, size, rate) are not extrapolated below zero
    grid = np.linspace(left, hi + share * r, 240)
    outside = (grid < lo) | (grid > hi)
    preds = []
    for m in models:
        dom = m.candidate.domain
        valid = np.ones_like(grid, bool)
        if "x_pos" in dom:
            valid &= grid > 0
        if "x_nonneg" in dom:
            valid &= grid >= 0
        if "x_same_sign" in dom:
            valid &= np.sign(grid) == np.sign(lo)
        p = m.predict(grid)
        preds.append(np.where(valid & np.isfinite(p), p, np.nan))
    if len(preds) < 2:
        return 0.0
    P = np.array(preds)[:, outside]
    enough = np.sum(np.isfinite(P), axis=0) >= 2
    if not enough.any():
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spread = np.nanmax(P[:, enough], axis=0) - np.nanmin(P[:, enough], axis=0)
    return float(np.max(spread) / (np.ptp(y) or 1.0))


def residual_checks(model, x, y):
    o = np.argsort(x, kind="stable")
    pred = model.predict(x)[o]
    r = y[o] - pred
    out = {"resid_lag1": acf1(r)}
    if x.size >= 15 and np.ptp(pred) > 0:
        rho, p = stats.spearmanr(pred, np.abs(r))
        out.update(hetero_rho=float(rho), hetero_p=float(p))
    return out
