"""Multi-feature mode: link(E[y]) = β₀ + Σ f_j(x_j) + interactions, one shape per feature.

Once the few nonlinear shape parameters (sigmoid slope and centre, hinge location, ...)
are fixed, every model here is linear in its coefficients. Fits therefore use variable
projection: coefficients are solved exactly — least squares for Gaussian and log-normal
targets, IRLS for binary and count targets — and the optimiser only moves the nonlinear
parameters. Shapes are chosen greedily by AIC, and the search is repeated inside every
cross-validation fold so the reported score is not flattered by it.
"""
import itertools
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.special import expit

from .families import get_family
from .fitting import _inside, inner_fit, wls
from .parallel import pmap, resolve_jobs
from .scoring import rank_table

SHAPE_LABELS = {
    "linear": "linear", "quadratic": "quadratic (U / inverted U)", "cubic": "cubic",
    "log": "logarithmic (diminishing)", "sqrt": "square root", "reciprocal": "reciprocal", "power": "power law",
    "exp": "exponential (accelerating or decaying)", "sigmoid": "sigmoid (threshold)", "peak": "single peak",
    "hinge": "hinge (kink)",
}
INF = np.inf


# ----------------------------------------------------------------------------- features
@dataclass
class Feature:
    name: str
    kind: str                  # numeric | binary | categorical | excluded
    note: str = ""
    raw: np.ndarray = None
    z: np.ndarray = None
    onehot: np.ndarray = None
    levels: list = None
    stats: dict = None
    discrete: bool = False
    is_time: bool = False


def _numeric(name, v, discrete=False, is_time=False, t0=None):
    med = float(np.median(v))
    q1, q3 = np.percentile(v, [25, 75])
    scale = float((q3 - q1) / 1.349) or float(np.std(v)) or 1.0
    z = (v - med) / scale
    pos = v[v > 0]
    st = {"med": med, "scale": scale, "zq": np.percentile(z, [2, 10, 25, 50, 75, 90, 98]),
          "zmin": float(z.min()), "zmax": float(z.max()), "x_pos": bool(np.all(v > 0)),
          "x_nonneg": bool(np.all(v >= 0)), "xref": float(np.median(pos)) if pos.size else 1.0,
          "nunique": int(np.unique(v).size), "t0": t0,
          "grid": np.percentile(v, np.linspace(1, 99, 60))}
    return Feature(name, "numeric", raw=v, z=z, stats=st, discrete=discrete, is_time=is_time)


def _categorical(name, s, n):
    vals = s.astype(str).to_numpy()
    levels, counts = np.unique(vals, return_counts=True)
    if levels.size <= 1:
        return Feature(name, "excluded", "only one distinct value")
    if levels.size > max(20, 0.05 * n):
        return Feature(name, "excluded",
                       f"{levels.size} categories: a high-cardinality field (looks like an ID); one-hot "
                       "encoding would let the model memorise rows and leak")
    rare = counts < max(5, 0.01 * n)
    merged = set(levels[rare].tolist()) if rare.sum() >= 2 else set()
    if merged:
        vals = np.where(np.isin(vals, list(merged)), "other", vals)
        levels, counts = np.unique(vals, return_counts=True)
    if levels.size <= 1:
        return Feature(name, "excluded", "only one distinct value after merging rare levels")
    order = np.argsort(-counts, kind="stable")
    ordered = [str(levels[i]) for i in order]
    note = f"reference level: {ordered[0]}" + (f"; {len(merged)} rare levels merged into 'other'" if merged else "")
    onehot = np.column_stack([(vals == lv).astype(float) for lv in ordered[1:]])
    return Feature(name, "categorical", note, onehot=onehot, levels=ordered, stats={"merged": merged})


def prepare_features(df):
    n = len(df)
    feats = []
    for col in df.columns:
        s, name = df[col], str(col)
        if pd.api.types.is_datetime64_any_dtype(s):
            t0 = s.min()
            years = (s - t0).dt.total_seconds().to_numpy() / (365.25 * 86400)
            feats.append(_numeric(name, years, is_time=True, t0=t0))
        elif pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
            feats.append(_categorical(name, s, n))
        else:
            v = s.to_numpy(float)
            uniq = np.unique(v)
            if uniq.size <= 1:
                feats.append(Feature(name, "excluded", "only one distinct value"))
            elif uniq.size == 2:
                feats.append(Feature(name, "binary", raw=(v == uniq[1]).astype(float),
                                     levels=[uniq[0].item(), uniq[1].item()]))
            elif np.allclose(v, np.round(v)) and uniq.size == n and np.ptp(v) == n - 1:
                feats.append(Feature(name, "excluded",
                                     "consecutive unique integers: looks like a row number or ID and would leak"))
            else:
                feats.append(_numeric(name, v, discrete=uniq.size <= 6))
    return feats


def encode(template, df):
    """Encode new rows with the scaling and levels learned from the training features."""
    out = []
    for f in template:
        if f.kind == "excluded":
            out.append(f)
            continue
        s = df[f.name]
        if f.kind == "numeric":
            if f.is_time:
                v = (pd.to_datetime(s) - f.stats["t0"]).dt.total_seconds().to_numpy() / (365.25 * 86400)
            else:
                v = s.to_numpy(float)
            out.append(replace(f, raw=v, z=(v - f.stats["med"]) / f.stats["scale"]))
        elif f.kind == "binary":
            out.append(replace(f, raw=(s.to_numpy() == f.levels[1]).astype(float)))
        else:
            vals = s.astype(str).to_numpy()
            vals = np.where(np.isin(vals, list(f.stats["merged"])), "other", vals)
            out.append(replace(f, onehot=np.column_stack([(vals == lv).astype(float) for lv in f.levels[1:]])))
    return out


# ----------------------------------------------------------------------------- shape components
@dataclass(frozen=True)
class Component:
    """A shape: n_lin coefficient columns that depend on n_nl nonlinear parameters."""
    shape: str
    n_lin: int
    nl_lb: tuple
    nl_ub: tuple
    grid: tuple
    columns: object          # (z, x, nl) -> list of arrays

    @property
    def n_nl(self):
        return len(self.nl_lb)

    @property
    def n_params(self):
        return self.n_lin + self.n_nl


def make_component(shape, st):
    zq, xref = st["zq"], st["xref"]
    zmin, zr = st["zmin"], (st["zmax"] - st["zmin"]) or 1.0
    mids = (zq[2], zq[3], zq[4])

    def span(lo, hi):
        return (lo, hi if hi > lo else lo + 1e-6)

    if shape == "linear":
        return Component(shape, 1, (), (), ((),), lambda z, x, nl: [z])
    if shape == "quadratic":
        return Component(shape, 2, (), (), ((),), lambda z, x, nl: [z, z * z])
    if shape == "cubic":
        return Component(shape, 3, (), (), ((),), lambda z, x, nl: [z, z * z, z ** 3])
    if shape == "log":
        return Component(shape, 1, (), (), ((),), lambda z, x, nl: [np.log(x / xref)])
    if shape == "sqrt":
        return Component(shape, 1, (), (), ((),), lambda z, x, nl: [np.sqrt(np.maximum(x, 0) / xref)])
    if shape == "reciprocal":
        return Component(shape, 1, (), (), ((),), lambda z, x, nl: [xref / x])
    if shape == "power":
        return Component(shape, 1, (-3.0,), (3.0,), ((0.5,), (2.0,), (-1.0,)),
                         lambda z, x, nl: [np.power(x / xref, nl[0])])
    if shape == "exp":
        return Component(shape, 1, (-15.0,), (15.0,), ((-6.0,), (-2.0,), (2.0,), (6.0,)),
                         lambda z, x, nl: [np.exp(nl[0] * (z - zmin) / zr)])
    if shape == "sigmoid":
        lo, hi = span(zq[0], zq[6])
        return Component(shape, 1, (0.2, lo), (50.0, hi), tuple((r, m) for m in mids for r in (1.0, 4.0)),
                         lambda z, x, nl: [expit(nl[0] * (z - nl[1]))])
    if shape == "peak":
        lo, hi = span(zq[0], zq[6])
        return Component(shape, 1, (lo, 0.1), (hi, 5.0), tuple((m, w) for m in mids for w in (0.5, 1.5)),
                         lambda z, x, nl: [np.exp(-((z - nl[0]) ** 2) / (2 * nl[1] ** 2))])
    if shape == "hinge":
        lo, hi = span(zq[1], zq[5])
        return Component(shape, 2, (lo,), (hi,), tuple((m,) for m in mids),
                         lambda z, x, nl: [z, np.maximum(0, z - nl[0])])
    raise ValueError(shape)


def allowed_shapes(f):
    st = f.stats
    if f.discrete:
        shapes = ["linear"] + (["quadratic"] if st["nunique"] >= 3 else [])
        return shapes + (["log"] if st["x_pos"] else [])
    shapes = ["linear", "quadratic", "cubic", "exp", "sigmoid", "peak", "hinge"]
    if st["x_pos"]:
        shapes += ["log", "reciprocal", "power"]
    if st["x_nonneg"]:
        shapes.append("sqrt")
    return shapes


# ----------------------------------------------------------------------------- model design
@dataclass
class Block:
    name: str
    kind: str                # numeric | binary | categorical | interaction
    component: object
    lin: slice
    nl: slice
    cols: object             # (idx, nl) -> list of arrays


class Design:
    """Column layout of one additive specification: eta = M(nl) @ beta, column 0 is the intercept."""

    def __init__(self, feats, shapes, interactions, family):
        self.feats, self.family = feats, family
        self.shapes, self.interactions = dict(shapes), list(interactions)
        self.blocks = []
        p, q, lb, ub, zs = 1, 0, [], [], {}
        for f in feats:
            if f.kind == "numeric":
                comp = make_component(self.shapes[f.name], f.stats)
                cols = (lambda c, z, x: lambda idx, nl: c.columns(z[idx], x[idx], nl))(comp, f.z, f.raw)
                self.blocks.append(Block(f.name, "numeric", comp, slice(p, p + comp.n_lin), slice(q, q + comp.n_nl), cols))
                p, q = p + comp.n_lin, q + comp.n_nl
                lb += list(comp.nl_lb)
                ub += list(comp.nl_ub)
                zs[f.name] = f.z
            elif f.kind == "binary":
                cols = (lambda v: lambda idx, nl: [v[idx]])(f.raw)
                self.blocks.append(Block(f.name, "binary", None, slice(p, p + 1), slice(q, q), cols))
                p += 1
            elif f.kind == "categorical":
                width = f.onehot.shape[1]
                cols = (lambda M: lambda idx, nl: list(M[idx].T))(f.onehot)
                self.blocks.append(Block(f.name, "categorical", None, slice(p, p + width), slice(q, q), cols))
                p += width
        for a, b in self.interactions:
            cols = (lambda za, zb: lambda idx, nl: [za[idx] * zb[idx]])(zs[a], zs[b])
            self.blocks.append(Block(f"{a} × {b}", "interaction", None, slice(p, p + 1), slice(q, q), cols))
            p += 1
        self.p, self.q = p, q
        self.nl_lb, self.nl_ub = np.array(lb, float), np.array(ub, float)

    @property
    def k(self):
        return self.p + self.q

    def matrix(self, idx, nl):
        cols = [np.ones(len(idx))]
        for b in self.blocks:
            cols.extend(b.cols(idx, nl[b.nl]))
        return np.column_stack(cols)

    def block(self, name):
        return next((b for b in self.blocks if b.name == name), None)

    def with_features(self, feats):
        return Design(feats, self.shapes, self.interactions, self.family)

    def start_nl(self, previous=None, override=None):
        """Nonlinear start: overrides first, then values from a previous fit with the same shape, else grid."""
        override = override or {}
        out = np.empty(self.q)
        for b in self.blocks:
            if b.kind != "numeric" or b.component.n_nl == 0:
                continue
            if b.name in override:
                out[b.nl] = override[b.name]
                continue
            old = previous.design.block(b.name) if previous is not None else None
            if old is not None and old.kind == "numeric" and old.component.shape == b.component.shape:
                out[b.nl] = previous.nl[old.nl]
            else:
                out[b.nl] = b.component.grid[0]
        return out


@dataclass
class Fit:
    design: Design
    nl: np.ndarray
    beta: np.ndarray


def _eta_limits(family):
    return (-35.0, 35.0) if family.name == "binomial" else (-700.0, 700.0)


def _mean(family, eta):
    lo, hi = _eta_limits(family)
    return eta if family.name == "gaussian" else family.inverse_link(np.clip(eta, lo, hi))


def _solve(design, idx, y, w, nl, beta0=None):
    fam = design.family
    with np.errstate(all="ignore"):
        M = design.matrix(idx, nl)
        if not np.all(np.isfinite(M)):
            return None, None
        if fam.name == "gaussian":
            beta = wls(M, y[idx], w[idx])
        elif fam.name == "lognormal":
            beta = wls(M, np.log(y[idx]), w[idx])
        else:
            beta, _ = inner_fit(M, y[idx], w[idx], fam, link=True, beta0=beta0)
    if not np.all(np.isfinite(beta)):
        return None, None
    return beta, M


def fit_design(design, idx, y, w, starts=None, max_nfev=None):
    """Variable-projection fit of a design on rows idx. Returns a Fit or None."""
    fam = design.family
    if design.q == 0:
        beta, _ = _solve(design, idx, y, w, np.empty(0))
        return None if beta is None else Fit(design, np.empty(0), beta)
    state = {"beta": None}
    yy, ww = y[idx], w[idx]

    def resid(nl):
        beta, M = _solve(design, idx, y, w, nl, state["beta"])
        if beta is None:
            return np.full(len(idx), 1e4)
        state["beta"] = beta
        with np.errstate(all="ignore"):
            r = fam.fit_residuals(yy, _mean(fam, M @ beta), ww)
        return np.where(np.isfinite(r), r, 1e8)

    iterative = fam.name in ("binomial", "poisson")
    best = None
    for s in starts or [design.start_nl()]:
        state["beta"] = None
        try:
            sol = least_squares(resid, _inside(s, design.nl_lb, design.nl_ub), bounds=(design.nl_lb, design.nl_ub),
                                x_scale="jac", max_nfev=max_nfev or 40 * (design.q + 1), ftol=1e-8, xtol=1e-8,
                                diff_step=1e-6 if iterative else None)
        except Exception:
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        return None
    beta, _ = _solve(design, idx, y, w, best.x)
    return None if beta is None else Fit(design, best.x, beta)


def fit_mean(fit, idx):
    return _mean(fit.design.family, fit.design.matrix(idx, fit.nl) @ fit.beta)


def aic_of(fit, idx, y, w):
    fam = fit.design.family
    with np.errstate(all="ignore"):
        mu = fam.clip(fit_mean(fit, idx))
    if not np.all(np.isfinite(mu)):
        return INF
    scale = fam.scale(y[idx], mu, w[idx])
    nll = float(np.sum(fam.nll(y[idx], mu, scale, w[idx])))
    return 2 * (fit.design.k + fam.extra_params) + 2 * nll


def contribution(fit, name, idx):
    b = fit.design.block(name)
    if b is None:
        return np.zeros(len(idx))
    with np.errstate(all="ignore"):
        return np.column_stack(b.cols(idx, fit.nl[b.nl])) @ fit.beta[b.lin]


def _working(fam, y, eta, w):
    """Link-scale working response and weights for screening one feature's shape."""
    mu = fam.clip(_mean(fam, eta))
    if fam.name == "lognormal":
        return np.log(y), w
    if fam.name == "binomial":
        v = mu * (1 - mu)
        return eta + np.clip((y - mu) / v, -8, 8), w * v
    if fam.name == "poisson":
        return eta + np.clip((y - mu) / mu, -8, 8), w * mu
    return y, w


def fit_component(comp, z, x, target, wt):
    """Screen one shape against a partial residual: intercept + shape columns, weighted least squares."""
    sw = np.sqrt(wt)

    def resid(nl):
        with np.errstate(all="ignore"):
            M = np.column_stack([np.ones_like(z)] + comp.columns(z, x, nl))
            if not np.all(np.isfinite(M)):
                return np.full(z.shape, 1e4)
            beta = wls(M, target, wt)
            return (target - M @ beta) * sw

    if comp.n_nl == 0:
        r = resid(())
        return float(r @ r), ()
    scored = sorted(((float(r @ r), g) for g in comp.grid for r in [resid(g)]), key=lambda t: t[0])
    lb, ub = np.array(comp.nl_lb), np.array(comp.nl_ub)
    try:
        sol = least_squares(resid, _inside(scored[0][1], lb, ub), bounds=(lb, ub), x_scale="jac",
                            max_nfev=30 * (comp.n_nl + 1))
    except Exception:
        return scored[0][0], scored[0][1]
    return 2 * sol.cost, tuple(sol.x)


def fit_linear(feats, family, idx, y, w):
    design = Design(feats, {f.name: "linear" for f in feats if f.kind == "numeric"}, [], family)
    fit = fit_design(design, idx, y, w)
    if fit is None:
        raise RuntimeError("the all-linear baseline model failed to fit")
    return fit, aic_of(fit, idx, y, w)


def search_shapes(feats, family, idx, y, w, sweeps=2, margin=4.0):
    fit, aic = fit_linear(feats, family, idx, y, w)
    numeric = [f for f in feats if f.kind == "numeric"]
    history = []
    for _ in range(sweeps):
        changed = False
        for f in sorted(numeric, key=lambda f: -np.var(contribution(fit, f.name, idx))):
            with np.errstate(all="ignore"):
                eta = fit.design.matrix(idx, fit.nl) @ fit.beta
            target, wt = _working(family, y[idx], eta, w[idx])
            partial = target - (eta - contribution(fit, f.name, idx))
            current = fit.design.shapes[f.name]
            screened = []
            for shape in allowed_shapes(f):
                sse, nl = fit_component(make_component(shape, f.stats), f.z[idx], f.raw[idx], partial, wt)
                if np.isfinite(sse):
                    screened.append((sse, shape, nl))
            now = next((s for s, sh, _ in screened if sh == current), INF)
            options = sorted((o for o in screened if o[1] != current and o[0] < 0.995 * now), key=lambda o: o[0])[:3]
            best = None
            for _, shape, nl in options:
                trial = Design(feats, dict(fit.design.shapes, **{f.name: shape}), fit.design.interactions, family)
                tfit = fit_design(trial, idx, y, w, [trial.start_nl(fit, {f.name: nl})])
                if tfit is None:
                    continue
                taic = aic_of(tfit, idx, y, w)
                if taic < aic - margin and (best is None or taic < best[0]):
                    best = (taic, shape, tfit)
            if best:
                history.append({"feature": f.name, "from": current, "to": best[1], "aic_gain": aic - best[0]})
                aic, _, fit = best
                changed = True
        if not changed:
            break
    return fit, aic, history


def search_interactions(fit, aic, idx, y, w, top=4, max_terms=2, margin=10.0):
    design = fit.design
    numeric = [f for f in design.feats if f.kind == "numeric"]
    ranked = sorted(numeric, key=lambda f: -np.var(contribution(fit, f.name, idx)))[:top]
    added = []
    for _ in range(max_terms):
        best = None
        for a, b in itertools.combinations([f.name for f in ranked], 2):
            if (a, b) in fit.design.interactions:
                continue
            trial = Design(design.feats, fit.design.shapes, fit.design.interactions + [(a, b)], design.family)
            tfit = fit_design(trial, idx, y, w, [trial.start_nl(fit)])
            if tfit is None:
                continue
            taic = aic_of(tfit, idx, y, w)
            if taic < aic - margin and (best is None or taic < best[0]):
                best = (taic, (a, b), tfit)
        if not best:
            break
        added.append({"pair": best[1], "aic_gain": aic - best[0]})
        aic, _, fit = best
    return fit, aic, added


# ----------------------------------------------------------------------------- reference
def _knn_matrix(feats):
    cols = []
    for f in feats:
        if f.kind == "numeric":
            cols.append(f.z[:, None])
        elif f.kind == "binary":
            cols.append(f.raw[:, None])
        elif f.kind == "categorical":
            cols.append(f.onehot)
    return np.hstack(cols) if cols else None


def knn_reference(feats, family, y, w, splits):
    """k-nearest-neighbour predictions (KD-tree) as a non-parametric benchmark."""
    M = _knn_matrix(feats)
    if M is None:
        return None
    g = np.log(y) if family.name == "lognormal" else y
    back = np.exp if family.name == "lognormal" else (lambda v: v)
    test_nll = np.full(y.size, np.nan)
    for tr, te in splits:
        k = int(np.clip(np.sqrt(len(tr)), 5, 50))
        tree = cKDTree(M[tr])
        _, nn_tr = tree.query(M[tr], k=k + 1, workers=-1)
        _, nn_te = tree.query(M[te], k=k, workers=-1)
        g_tr = g[tr]
        mu_tr = family.clip(back(g_tr[nn_tr[:, 1:]].mean(axis=1)))
        mu_te = family.clip(back(g_tr[np.atleast_2d(nn_te)].mean(axis=1)))
        scale = family.scale(y[tr], mu_tr, w[tr])
        test_nll[te] = family.nll(y[te], mu_te, scale, w[te])
    return test_nll


# ----------------------------------------------------------------------------- driver
def effect_curve(fit, f):
    """Centred contribution of numeric feature f over its 1st–99th percentile grid."""
    b = fit.design.block(f.name)
    grid = f.stats["grid"]
    nl, beta = fit.nl[b.nl], fit.beta[b.lin]
    with np.errstate(all="ignore"):
        curve = np.column_stack(b.component.columns((grid - f.stats["med"]) / f.stats["scale"], grid, nl)) @ beta
        centre = np.mean(np.column_stack(b.component.columns(f.z, f.raw, nl)) @ beta)
    return grid, curve - centre


def describe_effect(grid, eff):
    span = np.ptp(eff)
    if not np.isfinite(span) or span < 1e-12:
        return "negligible"
    d = np.diff(eff)
    sig = np.sign(np.where(np.abs(d) < 0.002 * span, 0, d))
    nz = sig[sig != 0]
    turns = int(np.sum(nz[1:] != nz[:-1])) if nz.size > 1 else 0
    if turns >= 2:
        return "oscillating"
    if turns == 1:
        return "rise then fall" if nz[0] > 0 else "fall then rise"
    t = (grid - grid.min()) / (np.ptp(grid) or 1.0)     # real x spacing, not percentile rank
    curv = np.polyfit(t, eff / span, 2)[0]
    rising = nz.size and nz[0] > 0
    with np.errstate(all="ignore"):
        slope = np.abs(np.gradient(eff, t))
    if np.all(np.isfinite(slope)) and slope.max() > 0:
        steepest = t[np.argmax(slope)]
        ends = max(slope[t <= t[0] + 0.1].mean(), slope[t >= t[-1] - 0.1].mean())
        if 0.1 < steepest < 0.9 and ends < 0.25 * slope.max():
            return "S-shaped rise" if rising else "S-shaped fall"
    if abs(curv) < 0.15:
        return "roughly linear rise" if rising else "roughly linear fall"
    if rising:
        return "accelerating rise" if curv > 0 else "rising, flattening"
    return "falling, flattening" if curv > 0 else "accelerating fall"


@dataclass
class AdditiveModel:
    id: str
    label: str
    fit: Fit
    scale: object
    template: list

    @property
    def design(self):
        return self.fit.design

    def predict(self, df):
        feats = encode(self.template, df)
        design = self.fit.design.with_features(feats)
        fam = design.family
        with np.errstate(all="ignore"):
            eta = design.matrix(np.arange(len(df)), self.fit.nl) @ self.fit.beta
        return fam.mean(fam.clip(_mean(fam, eta)), self.scale)

    def partial_effects(self):
        """Centred contribution of each term on the link scale, for plotting."""
        effects = {}
        by_name = {f.name: f for f in self.fit.design.feats}
        for b in self.fit.design.blocks:
            f = by_name.get(b.name)
            if f is None:
                continue
            beta = self.fit.beta[b.lin]
            if b.kind == "numeric":
                grid, eff = effect_curve(self.fit, f)
                effects[f.name] = ("numeric", grid, eff, f.raw)
            elif b.kind == "binary":
                effects[f.name] = ("levels", [str(v) for v in f.levels], np.array([0.0, beta[0]]), None)
            else:
                effects[f.name] = ("levels", f.levels, np.concatenate([[0.0], beta]), None)
        return effects


def _fold_scores(fit, family, tr, te, y, w):
    with np.errstate(all="ignore"):
        mu_tr = family.clip(fit_mean(fit, tr))
        mu_te = family.clip(fit_mean(fit, te))
    scale = family.scale(y[tr], mu_tr, w[tr])
    return family.nll(y[te], mu_te, scale, w[te]), family.mean(mu_te, scale)


def _nested_fold(usable, family, y, w, tr, te, variant_ids, full_curves):
    """Repeat the whole shape (and interaction) search on one training fold and score its test rows."""
    fits = {"linear": fit_linear(usable, family, tr, y, w)[0]}
    agreement = {}
    if "shapes" in variant_ids:
        s_fit, s_aic, _ = search_shapes(usable, family, tr, y, w)
        fits["shapes"] = s_fit
        for f in usable:
            if f.kind != "numeric":
                continue
            a, b = full_curves[f.name], effect_curve(s_fit, f)[1]
            if np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
                agreement[f.name] = float(np.ptp(a) < 1e-12 and np.ptp(b) < 1e-12)
            else:
                agreement[f.name] = float(np.corrcoef(a, b)[0, 1])
        if "interactions" in variant_ids:
            fits["interactions"] = search_interactions(s_fit, s_aic, tr, y, w)[0]
    scores = {vid: _fold_scores(fit, family, tr, te, y, w) for vid, fit in fits.items() if vid in variant_ids}
    return scores, agreement


def _nested_fold_remote(payload):
    usable, family_name, y, w, tr, te, variant_ids, full_curves = payload
    return _nested_fold(usable, get_family(family_name), y, w, tr, te, variant_ids, full_curves)


def run_additive(feats, family, y, w, splits, nested=True, interactions=True, n_jobs=1):
    n = y.size
    idx = np.arange(n)
    usable = [f for f in feats if f.kind != "excluded"]
    numeric = [f for f in usable if f.kind == "numeric"]

    lin_fit, lin_aic = fit_linear(usable, family, idx, y, w)
    variants = {"linear": ("All-linear GLM (baseline)" if numeric else "Category / binary offsets", lin_fit, lin_aic)}
    shp_fit, shp_aic, history, added = lin_fit, lin_aic, [], []
    if numeric:
        shp_fit, shp_aic, history = search_shapes(usable, family, idx, y, w)
        variants["shapes"] = ("Additive shape model", shp_fit, shp_aic)
        if interactions and len(numeric) >= 2:
            int_fit, int_aic, added = search_interactions(shp_fit, shp_aic, idx, y, w)
            if added:
                variants["interactions"] = ("Shapes + interactions", int_fit, int_aic)

    test_nll = {vid: np.full(n, np.nan) for vid in variants}
    test_pred = {vid: np.full(n, np.nan) for vid in variants}
    full_curves = {f.name: effect_curve(shp_fit, f)[1] for f in numeric}
    fold_agreement = {f.name: [] for f in numeric}
    if nested:
        variant_ids = tuple(variants)
        workers = resolve_jobs(n_jobs, len(splits))
        if workers > 1:
            outcomes = pmap(_nested_fold_remote, [(usable, family.name, y, w, tr, te, variant_ids, full_curves)
                                                  for tr, te in splits], workers)
        else:
            outcomes = [_nested_fold(usable, family, y, w, tr, te, variant_ids, full_curves) for tr, te in splits]
        for (tr, te), (scores, agreement) in zip(splits, outcomes):
            for vid, (nll, pred) in scores.items():
                test_nll[vid][te], test_pred[vid][te] = nll, pred
            for name, value in agreement.items():
                fold_agreement[name].append(value)
    else:
        for tr, te in splits:
            for vid, (_, vfit, _) in variants.items():
                refit = fit_design(vfit.design, tr, y, w, [vfit.nl] if vfit.design.q else None) or vfit
                test_nll[vid][te], test_pred[vid][te] = _fold_scores(refit, family, tr, te, y, w)

    models, rows = {}, []
    for vid, (label, vfit, aic) in variants.items():
        with np.errstate(all="ignore"):
            mu = family.clip(fit_mean(vfit, idx))
        scale = family.scale(y, mu, w)
        models[vid] = AdditiveModel(vid, label, vfit, scale, feats)
        pred = family.mean(mu, scale)
        rows.append({"id": vid, "model": vid, "label": label, "family": family.name, "k": vfit.design.k + family.extra_params,
                     "aic": aic, "r2": float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)),
                     "cv_rmse": float(np.sqrt(np.nanmean((y - test_pred[vid]) ** 2))), "status": "ok"})

    reference_ids = []
    if n >= 50:
        ref = knn_reference(usable, family, y, w, splits)
        if ref is not None:
            test_nll["knn"] = ref
            reference_ids.append("knn")
            rows.append({"id": "knn", "model": "knn", "label": "k-nearest neighbours (reference)",
                         "family": family.name, "k": np.nan, "status": "reference"})

    table, best, recommended = rank_table(rows, test_nll, w, reference_ids)
    table["d_aic"] = table["aic"] - table["aic"].min()

    final = models[recommended].fit
    variance = {b.name: float(np.var(contribution(final, b.name, idx))) for b in final.design.blocks}
    total = sum(variance.values()) or 1.0
    feature_rows = []
    for f in feats:
        row = {"feature": f.name, "type": f.kind, "note": f.note}
        if f.kind != "excluded":
            row["importance"] = variance.get(f.name, 0.0) / total
            if f.kind == "numeric":
                row["shape"] = SHAPE_LABELS[final.design.block(f.name).component.shape]
                row["trend"] = ("negligible" if row["importance"] < 0.005
                                else describe_effect(*effect_curve(final, f)))
                if fold_agreement.get(f.name):
                    row["stability"] = float(np.mean(fold_agreement[f.name]))
            else:
                row["shape"] = "offset"
        feature_rows.append(row)
    for b in final.design.blocks:
        if b.kind == "interaction":
            feature_rows.append({"feature": b.name, "type": "interaction", "shape": "product",
                                 "importance": variance[b.name] / total, "note": ""})

    return {"table": table, "models": models, "best": best, "recommended": recommended,
            "features": pd.DataFrame(feature_rows), "history": history, "interactions": added,
            "test_nll": test_nll, "reference_ids": reference_ids, "nested": nested,
            "full_shapes": dict(shp_fit.design.shapes)}
