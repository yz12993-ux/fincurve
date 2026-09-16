"""Multi-feature mode: link(E[y]) = β₀ + Σ f_j(x_j) + interactions, one shape per feature.

Every numeric feature starts linear. A greedy search swaps in a nonlinear shape
(log, saturating, sigmoid, peak, kink, ...) only when it lowers AIC by a clear
margin, then products of the most important features are tested. Categorical
features enter as one-hot offsets. Shape selection is repeated inside every
cross-validation fold so the reported score is not flattered by the search.
"""
import itertools
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.special import expit

from .families import get_family
from .fitting import fit_params
from .scoring import rank_table

GAUSSIAN = get_family("gaussian")
SHAPE_LABELS = {
    "linear": "线性", "quadratic": "二次（U 形/倒 U 形）", "cubic": "三次", "log": "对数（边际递减）",
    "sqrt": "平方根", "reciprocal": "反比", "power": "幂律", "exp": "指数（加速或衰减）",
    "sigmoid": "S 形（阈值切换）", "peak": "单峰", "hinge": "折线（有拐点）",
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
        return Feature(name, "excluded", "只有一个取值")
    if levels.size > max(20, 0.05 * n):
        return Feature(name, "excluded",
                       f"{levels.size} 个类别：高基数字段（像 ID），one-hot 会让模型记住样本、造成泄漏")
    rare = counts < max(5, 0.01 * n)
    merged = set(levels[rare].tolist()) if rare.sum() >= 2 else set()
    if merged:
        vals = np.where(np.isin(vals, list(merged)), "其他", vals)
        levels, counts = np.unique(vals, return_counts=True)
    if levels.size <= 1:
        return Feature(name, "excluded", "合并稀有类别后只剩一个取值")
    order = np.argsort(-counts, kind="stable")
    ordered = [str(levels[i]) for i in order]
    note = f"基准类别：{ordered[0]}" + (f"；{len(merged)} 个稀有类别并入“其他”" if merged else "")
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
                feats.append(Feature(name, "excluded", "只有一个取值"))
            elif uniq.size == 2:
                feats.append(Feature(name, "binary", raw=(v == uniq[1]).astype(float),
                                     levels=[uniq[0].item(), uniq[1].item()]))
            elif np.allclose(v, np.round(v)) and uniq.size == n and np.ptp(v) == n - 1:
                feats.append(Feature(name, "excluded", "连续整数且每行唯一：像行号或 ID，放进模型会造成泄漏"))
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
            vals = np.where(np.isin(vals, list(f.stats["merged"])), "其他", vals)
            out.append(replace(f, onehot=np.column_stack([(vals == lv).astype(float) for lv in f.levels[1:]])))
    return out


# ----------------------------------------------------------------------------- shape components
@dataclass
class Component:
    shape: str
    n: int
    f: object
    bounds: tuple
    init: object


def _grid_init(cols_fn, grid, assemble):
    """Weighted least squares for the linear coefficients at each nonlinear setting."""
    def init(z, x, target, wt):
        sw = np.sqrt(wt)
        found = []
        for g in grid:
            with np.errstate(all="ignore"):
                A = np.column_stack([np.ones_like(z)] + cols_fn(z, x, g))
                if not np.all(np.isfinite(A)):
                    continue
                coef, *_ = np.linalg.lstsq(A * sw[:, None], target * sw, rcond=None)
                rss = float(np.sum(wt * (A @ coef - target) ** 2))
            found.append((rss, [float(coef[0])] + [float(v) for v in assemble(coef[1:], g)]))
        found.sort(key=lambda t: t[0])
        return [p for _, p in found[:3]]
    return init


def make_component(shape, st):
    zq, xref = st["zq"], st["xref"]
    zmin, zr = st["zmin"], (st["zmax"] - st["zmin"]) or 1.0
    mids = (zq[2], zq[3], zq[4])
    if shape == "linear":
        return Component(shape, 1, lambda z, x, b: b * z, None,
                         _grid_init(lambda z, x, g: [z], [0], lambda c, g: [c[0]]))
    if shape == "quadratic":
        return Component(shape, 2, lambda z, x, b, c: b * z + c * z * z, None,
                         _grid_init(lambda z, x, g: [z, z * z], [0], lambda c, g: list(c)))
    if shape == "cubic":
        return Component(shape, 3, lambda z, x, b, c, d: b * z + c * z * z + d * z ** 3, None,
                         _grid_init(lambda z, x, g: [z, z * z, z ** 3], [0], lambda c, g: list(c)))
    if shape == "log":
        return Component(shape, 1, lambda z, x, b: b * np.log(x / xref), None,
                         _grid_init(lambda z, x, g: [np.log(x / xref)], [0], lambda c, g: [c[0]]))
    if shape == "sqrt":
        return Component(shape, 1, lambda z, x, b: b * np.sqrt(np.maximum(x, 0) / xref), None,
                         _grid_init(lambda z, x, g: [np.sqrt(np.maximum(x, 0) / xref)], [0], lambda c, g: [c[0]]))
    if shape == "reciprocal":
        return Component(shape, 1, lambda z, x, b: b * xref / x, None,
                         _grid_init(lambda z, x, g: [xref / x], [0], lambda c, g: [c[0]]))
    if shape == "power":
        return Component(shape, 2, lambda z, x, b, c: b * np.power(x / xref, c), ([-INF, -3], [INF, 3]),
                         _grid_init(lambda z, x, c: [np.power(x / xref, c)], [0.5, 2.0, -1.0],
                                    lambda k, c: [k[0], c]))
    if shape == "exp":
        return Component(shape, 2, lambda z, x, a, k: a * np.exp(k * (z - zmin) / zr), ([-INF, -15], [INF, 15]),
                         _grid_init(lambda z, x, k: [np.exp(k * (z - zmin) / zr)], [-6.0, -2.0, 2.0, 6.0],
                                    lambda c, k: [c[0], k]))
    if shape == "sigmoid":
        return Component(shape, 3, lambda z, x, K, r, m: K * expit(r * (z - m)),
                         ([-INF, 0.2, zq[0]], [INF, 50, zq[6]]),
                         _grid_init(lambda z, x, t: [expit(t[1] * (z - t[0]))],
                                    [(m, r) for m in mids for r in (1.0, 4.0)], lambda c, t: [c[0], t[1], t[0]]))
    if shape == "peak":
        return Component(shape, 3, lambda z, x, a, m, w: a * np.exp(-((z - m) ** 2) / (2 * w * w)),
                         ([-INF, zq[0], 0.1], [INF, zq[6], 5]),
                         _grid_init(lambda z, x, t: [np.exp(-((z - t[0]) ** 2) / (2 * t[1] ** 2))],
                                    [(m, w) for m in mids for w in (0.5, 1.5)], lambda c, t: [c[0], t[0], t[1]]))
    if shape == "hinge":
        return Component(shape, 3, lambda z, x, b, c, k: b * z + c * np.maximum(0, z - k),
                         ([-INF, -INF, zq[1]], [INF, INF, zq[5]]),
                         _grid_init(lambda z, x, k: [z, np.maximum(0, z - k)], mids, lambda c, k: [c[0], c[1], k]))
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
class Design:
    """Parameter layout and evaluation of one additive specification."""

    def __init__(self, feats, shapes, interactions, family):
        self.feats, self.family = feats, family
        self.shapes, self.interactions = dict(shapes), list(interactions)
        self.blocks = []
        lb, ub, k = [-INF], [INF], 1
        zs = {}
        for f in feats:
            if f.kind == "numeric":
                comp = make_component(self.shapes[f.name], f.stats)
                ev = (lambda c, z, x: lambda idx, p: c.f(z[idx], x[idx], *p))(comp, f.z, f.raw)
                n_p, b = comp.n, comp.bounds or ([-INF] * comp.n, [INF] * comp.n)
                zs[f.name] = f.z
            elif f.kind == "binary":
                ev, n_p, b = (lambda v: lambda idx, p: p[0] * v[idx])(f.raw), 1, ([-INF], [INF])
            elif f.kind == "categorical":
                n_p = f.onehot.shape[1]
                ev, b = (lambda M: lambda idx, p: M[idx] @ p)(f.onehot), ([-INF] * n_p, [INF] * n_p)
            else:
                continue
            self.blocks.append((f.name, f.kind, slice(k, k + n_p), ev))
            lb += list(b[0])
            ub += list(b[1])
            k += n_p
        for a, c in self.interactions:
            ev = (lambda za, zc: lambda idx, p: p[0] * za[idx] * zc[idx])(zs[a], zs[c])
            self.blocks.append((f"{a} × {c}", "interaction", slice(k, k + 1), ev))
            lb.append(-INF)
            ub.append(INF)
            k += 1
        self.k = k
        self.bounds = (np.array(lb), np.array(ub))

    def eta(self, idx, theta):
        theta = np.asarray(theta, float)
        out = np.full(len(idx), theta[0])
        for _, _, sl, ev in self.blocks:
            out = out + ev(idx, theta[sl])
        return out

    def mean_fn(self, idx, *theta):
        with np.errstate(all="ignore"):
            return self.family.inverse_link(self.eta(idx, theta))

    def contribution(self, name, idx, theta):
        for bname, _, sl, ev in self.blocks:
            if bname == name:
                with np.errstate(all="ignore"):
                    return ev(idx, np.asarray(theta, float)[sl])
        return np.zeros(len(idx))

    def with_features(self, feats):
        return Design(feats, self.shapes, self.interactions, self.family)


def _transfer(src, theta, dst, override=None, shift=0.0):
    override = override or {}
    out = np.zeros(dst.k)
    out[0] = theta[0] + shift
    src_blocks = {name: sl for name, _, sl, _ in src.blocks}
    for name, _, sl, _ in dst.blocks:
        if name in override:
            out[sl] = override[name]
        elif name in src_blocks and src.shapes.get(name) == dst.shapes.get(name):
            out[sl] = np.asarray(theta)[src_blocks[name]]
    return out


def _evaluate(design, theta, idx, y, w):
    fam = design.family
    mu = fam.clip(design.mean_fn(idx, *theta))
    if not np.all(np.isfinite(mu)):
        return np.inf, np.inf, None
    scale = fam.scale(y[idx], mu, w[idx])
    nll = float(np.sum(fam.nll(y[idx], mu, scale, w[idx])))
    return nll, 2 * (design.k + fam.extra_params) + 2 * nll, scale


def _working(fam, y, eta, w):
    mu = fam.clip(fam.inverse_link(eta))
    if fam.name == "lognormal":
        return np.log(y), w
    if fam.name == "binomial":
        v = mu * (1 - mu)
        return eta + np.clip((y - mu) / v, -8, 8), w * v
    if fam.name == "poisson":
        return eta + np.clip((y - mu) / mu, -8, 8), w * mu
    return y, w


def fit_linear(feats, family, idx, y, w):
    design = Design(feats, {f.name: "linear" for f in feats if f.kind == "numeric"}, [], family)
    cols = [np.ones(len(idx))]
    for f in feats:
        if f.kind == "numeric":
            cols.append(f.z[idx])
        elif f.kind == "binary":
            cols.append(f.raw[idx])
        elif f.kind == "categorical":
            cols += list(f.onehot[idx].T)
    A = np.column_stack(cols)
    sw = np.sqrt(w[idx])
    start, *_ = np.linalg.lstsq(A * sw[:, None], family.working_response(y[idx]) * sw, rcond=None)
    theta, _ = fit_params(design.mean_fn, idx, y[idx], w[idx], family, [start], design.bounds)
    if theta is None:
        raise RuntimeError("线性基准模型拟合失败")
    return design, theta, _evaluate(design, theta, idx, y, w)[1]


def search_shapes(feats, family, idx, y, w, sweeps=2, margin=4.0):
    design, theta, aic = fit_linear(feats, family, idx, y, w)
    numeric = [f for f in feats if f.kind == "numeric"]
    history = []
    for _ in range(sweeps):
        changed = False
        for f in sorted(numeric, key=lambda f: -np.var(design.contribution(f.name, idx, theta))):
            eta = design.eta(idx, theta)
            target, wt = _working(family, y[idx], eta, w[idx])
            current = design.contribution(f.name, idx, theta)
            partial = target - (eta - current)
            screened = []
            for shape in allowed_shapes(f):
                comp = make_component(shape, f.stats)
                starts = comp.init(f.z[idx], f.raw[idx], partial, wt)
                lb, ub = comp.bounds or ([-INF] * comp.n, [INF] * comp.n)
                func = (lambda c, z, x: lambda i, c0, *p: c0 + c.f(z[i], x[i], *p))(comp, f.z, f.raw)
                p, cost = fit_params(func, idx, partial, wt, GAUSSIAN, starts, ([-INF] + list(lb), [INF] + list(ub)))
                if p is not None:
                    screened.append((cost, shape, p))
            now = next((c for c, s, _ in screened if s == design.shapes[f.name]), INF)
            options = sorted((o for o in screened if o[1] != design.shapes[f.name] and o[0] < 0.995 * now),
                             key=lambda o: o[0])[:3]
            best = None
            for _, shape, p in options:
                trial = Design(feats, dict(design.shapes, **{f.name: shape}), design.interactions, family)
                start = _transfer(design, theta, trial, {f.name: p[1:]}, shift=p[0])
                t2, _ = fit_params(trial.mean_fn, idx, y[idx], w[idx], family, [start], trial.bounds)
                if t2 is None:
                    continue
                aic2 = _evaluate(trial, t2, idx, y, w)[1]
                if aic2 < aic - margin and (best is None or aic2 < best[0]):
                    best = (aic2, shape, trial, t2)
            if best:
                history.append({"feature": f.name, "from": design.shapes[f.name], "to": best[1],
                                "aic_gain": aic - best[0]})
                aic, _, design, theta = best
                changed = True
        if not changed:
            break
    return design, theta, aic, history


def search_interactions(design, theta, aic, idx, y, w, top=4, max_terms=2, margin=10.0):
    numeric = [f for f in design.feats if f.kind == "numeric"]
    ranked = sorted(numeric, key=lambda f: -np.var(design.contribution(f.name, idx, theta)))[:top]
    added = []
    for _ in range(max_terms):
        best = None
        for a, b in itertools.combinations([f.name for f in ranked], 2):
            if (a, b) in design.interactions:
                continue
            trial = Design(design.feats, design.shapes, design.interactions + [(a, b)], design.family)
            t2, _ = fit_params(trial.mean_fn, idx, y[idx], w[idx], design.family,
                               [_transfer(design, theta, trial)], trial.bounds)
            if t2 is None:
                continue
            aic2 = _evaluate(trial, t2, idx, y, w)[1]
            if aic2 < aic - margin and (best is None or aic2 < best[0]):
                best = (aic2, (a, b), trial, t2)
        if not best:
            break
        added.append({"pair": best[1], "aic_gain": aic - best[0]})
        aic, _, design, theta = best
    return design, theta, aic, added


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


def _knn_predict(M, g, tr, te, k, exclude_self=False):
    out = np.empty(len(te))
    for start in range(0, len(te), 500):
        chunk = te[start:start + 500]
        d = ((M[chunk][:, None, :] - M[tr][None, :, :]) ** 2).sum(-1)
        kk = min(k + exclude_self, len(tr))
        nn = np.argpartition(d, kk - 1, axis=1)[:, :kk]
        if exclude_self:
            order = np.argsort(np.take_along_axis(d, nn, axis=1), axis=1)
            nn = np.take_along_axis(nn, order, axis=1)[:, 1:]
        out[start:start + len(chunk)] = g[tr][nn].mean(axis=1)
    return out


def knn_reference(feats, family, y, w, splits):
    M = _knn_matrix(feats)
    if M is None:
        return None
    link = np.log if family.name == "lognormal" else (lambda v: v)
    inv = np.exp if family.name == "lognormal" else (lambda v: v)
    g = link(y)
    test_nll = np.full(y.size, np.nan)
    for tr, te in splits:
        k = int(np.clip(np.sqrt(len(tr)), 5, 50))
        mu_tr = family.clip(inv(_knn_predict(M, g, tr, tr, k, exclude_self=True)))
        mu_te = family.clip(inv(_knn_predict(M, g, tr, te, k)))
        scale = family.scale(y[tr], mu_tr, w[tr])
        test_nll[te] = family.nll(y[te], mu_te, scale, w[te])
    return test_nll


def effect_curve(design, theta, f):
    """Centred contribution of numeric feature f over its 1st–99th percentile grid."""
    sl = next(sl for name, _, sl, _ in design.blocks if name == f.name)
    comp = make_component(design.shapes[f.name], f.stats)
    grid = f.stats["grid"]
    with np.errstate(all="ignore"):
        curve = comp.f((grid - f.stats["med"]) / f.stats["scale"], grid, *np.asarray(theta)[sl])
        centre = np.mean(comp.f(f.z, f.raw, *np.asarray(theta)[sl]))
    return grid, curve - centre


def describe_effect(grid, eff):
    span = np.ptp(eff)
    if not np.isfinite(span) or span < 1e-12:
        return "几乎无影响"
    d = np.diff(eff)
    sig = np.sign(np.where(np.abs(d) < 0.002 * span, 0, d))
    nz = sig[sig != 0]
    turns = int(np.sum(nz[1:] != nz[:-1])) if nz.size > 1 else 0
    if turns >= 2:
        return "多次起伏"
    if turns == 1:
        return "先升后降" if nz[0] > 0 else "先降后升"
    t = (grid - grid.min()) / (np.ptp(grid) or 1.0)     # real x spacing, not percentile rank
    curv = np.polyfit(t, eff / span, 2)[0]
    rising = nz.size and nz[0] > 0
    with np.errstate(all="ignore"):
        slope = np.abs(np.gradient(eff, t))
    if np.all(np.isfinite(slope)) and slope.max() > 0:
        steepest = t[np.argmax(slope)]
        ends = max(slope[t <= t[0] + 0.1].mean(), slope[t >= t[-1] - 0.1].mean())
        if 0.1 < steepest < 0.9 and ends < 0.25 * slope.max():
            return "S 形上升（中间陡、两端平）" if rising else "S 形下降（中间陡、两端平）"
    if abs(curv) < 0.15:
        return "近似线性上升" if rising else "近似线性下降"
    if rising:
        return "加速上升" if curv > 0 else "上升趋缓"
    return "下降趋缓" if curv > 0 else "加速下降"


# ----------------------------------------------------------------------------- driver
@dataclass
class AdditiveModel:
    id: str
    label: str
    design: Design
    theta: np.ndarray
    scale: object
    template: list

    def predict(self, df):
        feats = encode(self.template, df)
        d = self.design.with_features(feats)
        idx = np.arange(len(df))
        return self.design.family.mean(self.design.family.clip(d.mean_fn(idx, *self.theta)), self.scale)

    def partial_effects(self):
        """Centred contribution of each term on the link scale, for plotting."""
        effects = {}
        comps = {name: sl for name, _, sl, _ in self.design.blocks}
        for f in self.design.feats:
            if f.name not in comps:
                continue
            p = self.theta[comps[f.name]]
            if f.kind == "numeric":
                comp = make_component(self.design.shapes[f.name], f.stats)
                grid = f.stats["grid"]
                with np.errstate(all="ignore"):
                    curve = comp.f((grid - f.stats["med"]) / f.stats["scale"], grid, *p)
                    centre = np.mean(comp.f(f.z, f.raw, *p))
                effects[f.name] = ("numeric", grid, curve - centre, f.raw)
            elif f.kind == "binary":
                effects[f.name] = ("levels", [str(v) for v in f.levels], np.array([0.0, p[0]]), None)
            else:
                effects[f.name] = ("levels", f.levels, np.concatenate([[0.0], p]), None)
        return effects


def run_additive(feats, family, y, w, splits, nested=True, interactions=True):
    n = y.size
    idx = np.arange(n)
    usable = [f for f in feats if f.kind != "excluded"]
    lin_design, lin_theta, lin_aic = fit_linear(usable, family, idx, y, w)
    shp_design, shp_theta, shp_aic, history = search_shapes(usable, family, idx, y, w)
    has_numeric = any(f.kind == "numeric" for f in usable)
    variants = {"linear": ("全线性（GLM 基准）" if has_numeric else "类别/二元偏移模型", lin_design, lin_theta, lin_aic)}
    if has_numeric:
        variants["shapes"] = ("加性形状模型", shp_design, shp_theta, shp_aic)
    added = []
    if interactions and sum(f.kind == "numeric" for f in usable) >= 2:
        int_design, int_theta, int_aic, added = search_interactions(shp_design, shp_theta, shp_aic, idx, y, w)
        if added:
            variants["interactions"] = ("加性形状 + 交互项", int_design, int_theta, int_aic)

    rows = []
    test_nll = {vid: np.full(n, np.nan) for vid in variants}
    test_pred = {vid: np.full(n, np.nan) for vid in variants}
    numeric = [f for f in usable if f.kind == "numeric"]
    full_curves = {f.name: effect_curve(shp_design, shp_theta, f)[1] for f in numeric}
    fold_agreement = {f.name: [] for f in numeric}
    for tr, te in splits:
        if nested:
            d_s, t_s, a_s, _ = search_shapes(usable, family, tr, y, w)
            for f in numeric:
                a, b = full_curves[f.name], effect_curve(d_s, t_s, f)[1]
                if np.ptp(a) < 1e-12 or np.ptp(b) < 1e-12:
                    fold_agreement[f.name].append(float(np.ptp(a) < 1e-12 and np.ptp(b) < 1e-12))
                else:
                    fold_agreement[f.name].append(float(np.corrcoef(a, b)[0, 1]))
            fold_models = {"linear": fit_linear(usable, family, tr, y, w)[:2], "shapes": (d_s, t_s)}
            if "interactions" in variants:
                d_i, t_i, _, _ = search_interactions(d_s, t_s, a_s, tr, y, w)
                fold_models["interactions"] = (d_i, t_i)
        else:
            fold_models = {}
            for vid, (_, d, th, _) in variants.items():
                t_fold, _ = fit_params(d.mean_fn, tr, y[tr], w[tr], family, [th], d.bounds)
                fold_models[vid] = (d, th if t_fold is None else t_fold)
        for vid, (d, th) in fold_models.items():
            if vid not in variants:
                continue
            mu_tr = family.clip(d.mean_fn(tr, *th))
            mu_te = family.clip(d.mean_fn(te, *th))
            scale = family.scale(y[tr], mu_tr, w[tr])
            test_nll[vid][te] = family.nll(y[te], mu_te, scale, w[te])
            test_pred[vid][te] = family.mean(mu_te, scale)

    models = {}
    for vid, (label, d, th, aic) in variants.items():
        scale = _evaluate(d, th, idx, y, w)[2]
        models[vid] = AdditiveModel(vid, label, d, th, scale, feats)
        pred = family.mean(family.clip(d.mean_fn(idx, *th)), scale)
        rows.append({"id": vid, "model": vid, "label": label, "family": family.name, "k": d.k + family.extra_params,
                     "aic": aic, "r2": float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)),
                     "cv_rmse": float(np.sqrt(np.nanmean((y - test_pred[vid]) ** 2))), "status": "ok"})

    reference_ids = []
    if n >= 50:
        ref = knn_reference(usable, family, y, w, splits)
        if ref is not None:
            test_nll["knn"] = ref
            reference_ids.append("knn")
            rows.append({"id": "knn", "model": "knn", "label": "K 近邻（非参数参照）", "family": family.name,
                         "k": np.nan, "status": "reference"})

    table, best, recommended = rank_table(rows, test_nll, w, reference_ids)
    table["d_aic"] = table["aic"] - table["aic"].min()

    final = models[recommended]
    effects_var = {name: float(np.var(final.design.contribution(name, idx, final.theta)))
                   for name, *_ in final.design.blocks}
    total = sum(effects_var.values()) or 1.0
    feature_rows = []
    for f in feats:
        row = {"feature": f.name, "type": {"numeric": "数值", "binary": "二元", "categorical": "类别",
                                            "excluded": "已剔除"}[f.kind], "note": f.note}
        if f.kind != "excluded":
            shape = final.design.shapes.get(f.name)
            row["shape"] = SHAPE_LABELS.get(shape, "偏移项") if f.kind == "numeric" else "偏移项"
            row["importance"] = effects_var.get(f.name, 0.0) / total
            if f.kind == "numeric":
                row["trend"] = ("几乎无影响" if row["importance"] < 0.005
                                else describe_effect(*effect_curve(final.design, final.theta, f)))
                if fold_agreement.get(f.name):
                    row["stability"] = float(np.mean(fold_agreement[f.name]))
        feature_rows.append(row)
    for name, kind, *_ in final.design.blocks:
        if kind == "interaction":
            feature_rows.append({"feature": name, "type": "交互项", "shape": "乘积",
                                 "importance": effects_var[name] / total, "note": ""})

    return {"table": table, "models": models, "best": best, "recommended": recommended,
            "features": pd.DataFrame(feature_rows), "history": history, "interactions": added,
            "test_nll": test_nll, "reference_ids": reference_ids, "nested": nested,
            "full_shapes": dict(shp_design.shapes)}
