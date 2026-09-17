"""Library of candidate y = f(x) forms, including finance-specific curves.

Each candidate knows its formula, parameter bounds and the data domain it needs
(e.g. x > 0). Separable candidates — mean = B(x, θ) @ β — also declare their basis
columns, the nonlinear parameters θ with bounds and a search grid, and how (β, θ)
map back to the published parameter order. Those are fitted by variable projection;
the two non-separable forms (Heston term structure, SVI) use general least squares.
"""
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.signal import lombscargle
from scipy.special import expit, ndtr

GROUPS = {
    "baseline": "baseline",
    "polynomial": "polynomial",
    "growth_decay": "growth / decay",
    "power_law": "power law",
    "diminishing": "diminishing returns",
    "saturation": "saturation",
    "sigmoid": "sigmoid",
    "peak": "peak",
    "piecewise": "piecewise / kink",
    "periodic": "periodic",
    "term_structure": "term structure",
    "volatility": "volatility",
    "fixed_income": "fixed income",
    "credit": "credit",
    "options": "options",
    "link": "GLM (link function)",
}


@dataclass(frozen=True)
class Scale:
    """Constants fixed from the full x sample. They only re-parameterise the curves."""
    x0: float
    s: float
    xmed: float
    unit: float

    @classmethod
    def from_x(cls, x):
        x = np.asarray(x, float)
        s = float(np.ptp(x)) or 1.0
        pos = x[x > 0]
        xmed = float(np.median(pos)) if pos.size else s
        unit = 0.01 if np.max(np.abs(x)) > 1.5 else 1.0
        return cls(float(x.min()), s, xmed, unit)


@dataclass(frozen=True)
class Candidate:
    key: str
    label: str
    formula: str
    group: str
    meaning: str
    params: tuple
    func: Callable
    init: Callable
    bounds: Optional[tuple] = None
    domain: frozenset = frozenset()
    tags: frozenset = frozenset()
    min_n: int = 0
    uses_u: bool = False
    basis: Optional[Callable] = None        # (x, theta) -> list of columns
    assemble: Optional[Callable] = None     # (beta, theta) -> params in published order
    nl_grid: Optional[Callable] = None      # (x, y) -> list of theta tuples
    nl_bounds: Optional[tuple] = None       # bounds on theta
    coef_bounds: Optional[tuple] = None     # bounds on beta
    link: bool = False                      # mean = inverse_link(B @ beta)

    @property
    def k(self):
        return len(self.params)

    @property
    def separable(self):
        return self.basis is not None

    @property
    def q(self):
        return 0 if self.nl_bounds is None else len(self.nl_bounds[0])


# ----------------------------------------------------------------------------- helpers
def _lstsq(cols, y):
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef, float(np.sum((A @ coef - y) ** 2))


def _grid(build, grid, y, top=3):
    """For each nonlinear setting, solve the linear coefficients; keep the best `top` starts."""
    found = []
    for theta in grid:
        with np.errstate(all="ignore"):
            try:
                cols, assemble = build(theta)
                cols = [np.broadcast_to(c, y.shape) for c in cols]
                if not all(np.all(np.isfinite(c)) for c in cols):
                    continue
                coef, rss = _lstsq(cols, y)
                p = [float(v) for v in assemble(coef)]
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                continue
        if np.isfinite(rss) and np.all(np.isfinite(p)):
            found.append((rss, p))
    found.sort(key=lambda t: t[0])
    return [p for _, p in found[:top]]


def _smoothed(x, y):
    o = np.argsort(x, kind="stable")
    win = max(3, len(x) // 8)
    ys = pd.Series(y[o]).rolling(win, center=True, min_periods=1).mean().to_numpy()
    return x[o], ys


def _ns_basis(x, tau):
    """Nelson–Siegel loadings L(z) = (1 − e^−z)/z and L(z) − e^−z with z = x/τ."""
    z = np.maximum(np.asarray(x, float), 0) / tau
    small = z < 1e-6
    zz = np.where(small, 1.0, z)
    L = np.where(small, 1 - z / 2, -np.expm1(-zz) / zz)
    return L, L - np.exp(-z)


def _bs_call(x, K, v):
    xx = np.maximum(np.asarray(x, float), 1e-300)
    d1 = (np.log(xx / K) + 0.5 * v * v) / v
    return xx * ndtr(d1) - K * ndtr(d1 - v)


def _annuity(x, N, unit):
    r = np.asarray(x, float) * unit
    small = np.abs(r) < 1e-10
    rs = np.where(small, 1.0, r)
    return np.where(small, N, -np.expm1(-r * N) / rs), np.exp(-r * N)


# ----------------------------------------------------------------------------- library
def build_library(sc: Scale):
    x0, s, xmed, unit = sc.x0, sc.s, sc.xmed, sc.unit
    inf = np.inf

    def U(x):
        return (np.asarray(x, float) - x0) / s

    def one(x):
        return np.ones_like(np.asarray(x, float))

    lib = []

    def add(key, label, formula, group, meaning, params, func, init=None, bounds=None,
            domain=(), tags=(), min_n=0, uses_u=False, basis=None, assemble=None, grid=None,
            nl_bounds=None, coef_bounds=None):
        if basis is not None:
            nl_bounds = nl_bounds or ([], [])
            grid = grid or (lambda x, y: [()])
            if init is None:
                init = (lambda b, a, g: lambda x, y: _grid(lambda th: (b(x, th), lambda c: a(c, th)), g(x, y), y))(
                    basis, assemble, grid)
        lib.append(Candidate(key, label, formula, group, meaning, tuple(params), func, init, bounds,
                             frozenset(domain), frozenset(tags), min_n, uses_u, basis, assemble, grid,
                             nl_bounds, coef_bounds))

    # ---- baseline and polynomials
    add("constant", "Constant", "a", "baseline", "y does not depend on x (baseline)", ["a"],
        lambda x, a: a * one(x),
        basis=lambda x, th: [one(x)], assemble=lambda c, th: [c[0]])
    add("linear", "Linear", "a + b·u", "polynomial", "constant slope: beta, duration approximation, linear pricing", ["a", "b"],
        lambda x, a, b: a + b * U(x),
        basis=lambda x, th: [one(x), U(x)], assemble=lambda c, th: [c[0], c[1]], uses_u=True)
    add("quadratic", "Quadratic", "a + b·u + c·u²", "polynomial", "U / inverted U: optima, convexity", ["a", "b", "c"],
        lambda x, a, b, c: a + b * U(x) + c * U(x) ** 2,
        basis=lambda x, th: [one(x), U(x), U(x) ** 2], assemble=lambda c, th: list(c[:3]),
        tags=("peak",), uses_u=True)
    add("cubic", "Cubic", "a + b·u + c·u² + d·u³", "polynomial", "flexible curve with an inflection (dangerous to extrapolate)", ["a", "b", "c", "d"],
        lambda x, a, b, c, d: a + b * U(x) + c * U(x) ** 2 + d * U(x) ** 3,
        basis=lambda x, th: [one(x), U(x), U(x) ** 2, U(x) ** 3], assemble=lambda c, th: list(c[:4]),
        tags=("peak", "flexible"), uses_u=True)

    # ---- exponential family
    def grid_exp(x, y):
        g = [(-3.0,), (-1.0,), (1.0,), (3.0,)]
        if np.all(y > 0):
            b, _ = np.polyfit(U(x), np.log(y), 1)
            g.insert(0, (float(np.clip(b, -49, 49)),))
        return g

    add("exponential", "Exponential", "a·e^(b·u)", "growth_decay", "compounding growth, exponential decay", ["a", "b"],
        lambda x, a, b: a * np.exp(b * U(x)), bounds=([-inf, -50], [inf, 50]), uses_u=True,
        basis=lambda x, th: [np.exp(th[0] * U(x))], assemble=lambda c, th: [c[0], th[0]],
        grid=grid_exp, nl_bounds=([-50.0], [50.0]))

    add("exp_offset", "Exponential approach", "c + a·e^(−κ·u)", "growth_decay",
        "mean reversion to a long-run level (vol and spread term structures)", ["c", "a", "κ"],
        lambda x, c, a, k: c + a * np.exp(-k * U(x)), bounds=([-inf, -inf, -30], [inf, inf, 300]),
        tags=("saturating", "term_axis"), uses_u=True,
        basis=lambda x, th: [one(x), np.exp(-th[0] * U(x))], assemble=lambda c, th: [c[0], c[1], th[0]],
        grid=lambda x, y: [(1.0,), (3.0,), (10.0,), (30.0,), (-2.0,)], nl_bounds=([-30.0], [300.0]))

    add("double_exp", "Double exponential", "c + a₁·e^(−(κ₂+Δ)·u) + a₂·e^(−κ₂·u)", "growth_decay",
        "two-factor mean reversion with fast and slow speeds", ["c", "a1", "a2", "κ2", "Δ"],
        lambda x, c, a1, a2, k2, dk: c + a1 * np.exp(-(k2 + dk) * U(x)) + a2 * np.exp(-k2 * U(x)),
        bounds=([-inf, -inf, -inf, 0, 0], [inf, inf, inf, 100, 500]), tags=("term_axis",), min_n=12, uses_u=True,
        basis=lambda x, th: [one(x), np.exp(-(th[0] + th[1]) * U(x)), np.exp(-th[0] * U(x))],
        assemble=lambda c, th: [c[0], c[1], c[2], th[0], th[1]],
        grid=lambda x, y: [(k2, dk) for k2 in (0.5, 2.0, 6.0) for dk in (5.0, 20.0, 60.0)],
        nl_bounds=([0.0, 0.0], [100.0, 500.0]))

    # ---- power laws
    def grid_power(x, y):
        g = [(0.5,), (1.0,), (-0.5,), (2.0,)]
        for sgn in (1, -1):
            if np.all(sgn * y > 0):
                b, _ = np.polyfit(np.log(x), np.log(sgn * y), 1)
                g.insert(0, (float(np.clip(b, -9.9, 9.9)),))
        return g

    add("power", "Power law", "a·x^b", "power_law", "scaling laws: square-root impact, volatility time scaling, tails", ["a", "b"],
        lambda x, a, b: a * np.power(x, b), bounds=([-inf, -10], [inf, 10]), domain=("x_pos",),
        basis=lambda x, th: [np.power(x, th[0])], assemble=lambda c, th: [c[0], th[0]],
        grid=grid_power, nl_bounds=([-10.0], [10.0]))
    add("power_offset", "Power law + offset", "a + b·x^c", "power_law", "power law with a floor (fixed cost plus scale term)", ["a", "b", "c"],
        lambda x, a, b, c: a + b * np.power(x, c), bounds=([-inf, -inf, -5], [inf, inf, 5]), domain=("x_pos",),
        basis=lambda x, th: [one(x), np.power(x, th[0])], assemble=lambda c, th: [c[0], c[1], th[0]],
        grid=lambda x, y: [(0.25,), (0.5,), (1.0,), (2.0,), (-1.0,)], nl_bounds=([-5.0], [5.0]))
    add("tempered_power", "Tempered power law", "a·x^b·e^(−c·x/s)", "power_law", "power-law growth with an exponential cut-off", ["a", "b", "c"],
        lambda x, a, b, c: a * np.power(x, b) * np.exp(-c * np.asarray(x, float) / s),
        bounds=([-inf, -10, -20], [inf, 10, 50]), domain=("x_pos",), tags=("peak",),
        basis=lambda x, th: [np.power(x, th[0]) * np.exp(-th[1] * np.asarray(x, float) / s)],
        assemble=lambda c, th: [c[0], th[0], th[1]],
        grid=lambda x, y: [(b, c) for b in (0.5, 1.0, 2.0, -0.5) for c in (0.0, 1.0, 3.0)],
        nl_bounds=([-10.0, -20.0], [10.0, 50.0]))

    # ---- diminishing returns
    add("log", "Logarithmic", "a + b·ln x", "diminishing", "diminishing returns, learning curves, log returns", ["a", "b"],
        lambda x, a, b: a + b * np.log(x), domain=("x_pos",),
        basis=lambda x, th: [one(x), np.log(x)], assemble=lambda c, th: [c[0], c[1]])
    add("log_shift", "Shifted log", "a + b·ln(x + c)", "diminishing", "diminishing returns when x can be zero", ["a", "b", "c"],
        lambda x, a, b, c: a + b * np.log(np.asarray(x, float) + c),
        bounds=([-inf, -inf, -x0 + 1e-3 * s], [inf, inf, 10 * s]),
        basis=lambda x, th: [one(x), np.log(np.asarray(x, float) + th[0])], assemble=lambda c, th: [c[0], c[1], th[0]],
        grid=lambda x, y: [(-x0 + s * f,) for f in (0.01, 0.1, 0.5, 2.0)],
        nl_bounds=([-x0 + 1e-3 * s], [10 * s]))
    add("sqrt", "Square root", "a + b·√x", "diminishing", "diffusion, √t scaling, market impact", ["a", "b"],
        lambda x, a, b: a + b * np.sqrt(x), domain=("x_nonneg",),
        basis=lambda x, th: [one(x), np.sqrt(x)], assemble=lambda c, th: [c[0], c[1]])
    add("hyperbolic", "Hyperbolic", "a + b/x", "diminishing", "unit-cost amortisation", ["a", "b"],
        lambda x, a, b: a + b / np.asarray(x, float), domain=("x_same_sign",), tags=("saturating",),
        basis=lambda x, th: [one(x), 1 / np.asarray(x, float)], assemble=lambda c, th: [c[0], c[1]])
    add("rational", "Rational", "(a + b·u)/(1 + c·u)", "diminishing", "flexible monotone shape: saturating or hyperbolic", ["a", "b", "c"],
        lambda x, a, b, c: (a + b * U(x)) / (1 + c * U(x)), bounds=([-inf, -inf, -0.9], [inf, inf, 200]),
        tags=("saturating",), uses_u=True,
        basis=lambda x, th: [1 / (1 + th[0] * U(x)), U(x) / (1 + th[0] * U(x))],
        assemble=lambda c, th: [c[0], c[1], th[0]],
        grid=lambda x, y: [(0.0,), (0.5,), (2.0,), (10.0,)], nl_bounds=([-0.9], [200.0]))

    # ---- saturation
    add("michaelis_menten", "Michaelis–Menten", "V·x/(K + x)", "saturation",
        "capacity limits: strategy capacity, AUM vs return", ["V", "K"],
        lambda x, V, K: V * np.asarray(x, float) / (K + np.asarray(x, float)),
        bounds=([-inf, 1e-6 * s], [inf, 100 * s]), domain=("x_nonneg",), tags=("saturating",),
        basis=lambda x, th: [np.asarray(x, float) / (th[0] + np.asarray(x, float))],
        assemble=lambda c, th: [c[0], th[0]],
        grid=lambda x, y: [(K,) for K in (xmed / 2, xmed, 2 * xmed, s)], nl_bounds=([1e-6 * s], [100 * s]))
    add("hill", "Hill saturation", "d + V/(1 + (K/x)ⁿ)", "saturation", "saturating S-curve with adjustable steepness", ["d", "V", "K", "n"],
        lambda x, d, V, K, n: d + V / (1 + np.power(K / np.maximum(np.asarray(x, float), 1e-300), n)),
        bounds=([-inf, -inf, 1e-6 * s, 0.2], [inf, inf, 100 * s, 12]), domain=("x_nonneg",),
        tags=("sigmoid", "saturating"), min_n=10,
        basis=lambda x, th: [one(x), 1 / (1 + np.power(th[0] / np.maximum(np.asarray(x, float), 1e-300), th[1]))],
        assemble=lambda c, th: [c[0], c[1], th[0], th[1]],
        grid=lambda x, y: [(K, n) for K in (xmed / 2, xmed, 2 * xmed) for n in (1.0, 2.0, 4.0)],
        nl_bounds=([1e-6 * s, 0.2], [100 * s, 12.0]))

    # ---- sigmoids (θ = (r, m))
    sig_grid = [(r, m) for m in (0.2, 0.35, 0.5, 0.65, 0.8) for r in (5.0, 15.0, 40.0)]
    add("logistic", "Logistic", "d + K/(1 + e^(−r·(u − m)))", "sigmoid",
        "S-curve: default rate vs score, adoption, regime switches", ["d", "K", "r", "m"],
        lambda x, d, K, r, m: d + K * expit(r * (U(x) - m)),
        bounds=([-inf, -inf, 0.1, -0.5], [inf, inf, 500, 1.5]), tags=("sigmoid",), min_n=8, uses_u=True,
        basis=lambda x, th: [one(x), expit(th[0] * (U(x) - th[1]))], assemble=lambda c, th: [c[0], c[1], th[0], th[1]],
        grid=lambda x, y: sig_grid, nl_bounds=([0.1, -0.5], [500.0, 1.5]))
    add("gompertz", "Gompertz", "d + K·exp(−e^(−r·(u − m)))", "sigmoid", "asymmetric S-curve", ["d", "K", "r", "m"],
        lambda x, d, K, r, m: d + K * np.exp(-np.exp(-r * (U(x) - m))),
        bounds=([-inf, -inf, 0.1, -0.5], [inf, inf, 500, 1.5]), tags=("sigmoid",), min_n=8, uses_u=True,
        basis=lambda x, th: [one(x), np.exp(-np.exp(-th[0] * (U(x) - th[1])))],
        assemble=lambda c, th: [c[0], c[1], th[0], th[1]],
        grid=lambda x, y: sig_grid, nl_bounds=([0.1, -0.5], [500.0, 1.5]))

    # ---- peaks (θ = (m, w))
    def peak_grid(x, y):
        xs, ys = _smoothed(x, y)
        centres = {float(np.clip(U(xs[np.argmax(ys)]), -0.5, 1.5)), float(np.clip(U(xs[np.argmin(ys)]), -0.5, 1.5)), 0.5}
        return [(m, w) for m in sorted(centres) for w in (0.05, 0.15, 0.4)]

    add("gaussian_peak", "Gaussian peak", "d + a·exp(−(u − m)²/(2w²))", "peak", "single peak around an optimum", ["d", "a", "m", "w"],
        lambda x, d, a, m, w: d + a * np.exp(-((U(x) - m) ** 2) / (2 * w * w)),
        bounds=([-inf, -inf, -0.5, 0.005], [inf, inf, 1.5, 5]), tags=("peak",), min_n=8, uses_u=True,
        basis=lambda x, th: [one(x), np.exp(-((U(x) - th[0]) ** 2) / (2 * th[1] * th[1]))],
        assemble=lambda c, th: [c[0], c[1], th[0], th[1]], grid=peak_grid, nl_bounds=([-0.5, 0.005], [1.5, 5.0]))
    add("lorentzian", "Lorentzian peak", "d + a/(1 + ((u − m)/w)²)", "peak", "sharp peak with heavy shoulders", ["d", "a", "m", "w"],
        lambda x, d, a, m, w: d + a / (1 + ((U(x) - m) / w) ** 2),
        bounds=([-inf, -inf, -0.5, 0.005], [inf, inf, 1.5, 5]), tags=("peak",), min_n=8, uses_u=True,
        basis=lambda x, th: [one(x), 1 / (1 + ((U(x) - th[0]) / th[1]) ** 2)],
        assemble=lambda c, th: [c[0], c[1], th[0], th[1]], grid=peak_grid, nl_bounds=([-0.5, 0.005], [1.5, 5.0]))

    # ---- piecewise
    add("hinge", "Hinge", "a + b·u + c·max(0, u − k)", "piecewise",
        "threshold effects: tax brackets, leverage and margin limits", ["a", "b", "c", "k"],
        lambda x, a, b, c, k: a + b * U(x) + c * np.maximum(0, U(x) - k),
        bounds=([-inf, -inf, -inf, 0.02], [inf, inf, inf, 0.98]), tags=("kink",), min_n=8, uses_u=True,
        basis=lambda x, th: [one(x), U(x), np.maximum(0, U(x) - th[0])],
        assemble=lambda c, th: [c[0], c[1], c[2], th[0]],
        grid=lambda x, y: [(k,) for k in (0.15, 0.3, 0.45, 0.6, 0.75, 0.85)], nl_bounds=([0.02], [0.98]))
    add("softplus", "Soft hockey stick", "a + b·u + c·w·ln(1 + e^((u − k)/w))", "piecewise",
        "option-like payoffs, floors and stops", ["a", "b", "c", "k", "w"],
        lambda x, a, b, c, k, w: a + b * U(x) + c * w * np.logaddexp(0, (U(x) - k) / w),
        bounds=([-inf, -inf, -inf, -0.2, 0.002], [inf, inf, inf, 1.2, 1.0]), tags=("kink",), min_n=10, uses_u=True,
        basis=lambda x, th: [one(x), U(x), th[1] * np.logaddexp(0, (U(x) - th[0]) / th[1])],
        assemble=lambda c, th: [c[0], c[1], c[2], th[0], th[1]],
        grid=lambda x, y: [(k, w) for k in (0.2, 0.4, 0.6, 0.8) for w in (0.02, 0.1)],
        nl_bounds=([-0.2, 0.002], [1.2, 1.0]))

    # ---- periodic
    def grid_sine(x, y):
        u = U(x)
        coef, _ = _lstsq([one(x), u], y)
        r = y - (coef[0] + coef[1] * u)
        periods = np.geomspace(max(2.5 * s / len(x), s / 400), 1.5 * s, 300)
        with np.errstate(all="ignore"):
            power = lombscargle(x - x0, r - r.mean(), 2 * np.pi / periods)
        picks = []
        for i in np.argsort(power)[::-1]:
            if all(abs(np.log(periods[i] / p)) > 0.15 for p in picks):
                picks.append(float(periods[i]))
            if len(picks) == 3:
                break
        return [(P,) for P in picks]

    def sine_cols(x, th):
        t = 2 * np.pi * (np.asarray(x, float) - x0) / th[0]
        return [one(x), U(x), np.sin(t), np.cos(t)]

    add("sine_trend", "Sine + trend", "a + b·u + c·sin(2π·x/P + φ)", "periodic", "seasonality, intraday and weekday effects",
        ["a", "b", "c", "P", "φ"],
        lambda x, a, b, c, P, ph: a + b * U(x) + c * np.sin(2 * np.pi * (np.asarray(x, float) - x0) / P + ph),
        bounds=([-inf, -inf, -inf, s / 400, -10], [inf, inf, inf, 2 * s, 10]), tags=("periodic",), min_n=20, uses_u=True,
        basis=sine_cols,
        assemble=lambda c, th: [c[0], c[1], float(np.hypot(c[2], c[3])), th[0], float(np.arctan2(c[3], c[2]))],
        grid=grid_sine, nl_bounds=([s / 400], [2 * s]))

    # ---- term structures
    tau_grid = s * np.array([0.01, 0.03, 0.07, 0.15, 0.3, 0.6])

    def f_ns(x, b0, b1, b2, tau):
        L, H = _ns_basis(x, tau)
        return b0 + b1 * L + b2 * H

    def ns_cols(x, th):
        L, H = _ns_basis(x, th[0])
        return [one(x), L, H]

    add("nelson_siegel", "Nelson–Siegel", "β₀ + β₁·L(x/τ) + β₂·[L(x/τ) − e^(−x/τ)]", "term_structure",
        "yield / forward curve: level, slope and curvature", ["β0", "β1", "β2", "τ"], f_ns,
        bounds=([-inf, -inf, -inf, 0.002 * s], [inf, inf, inf, 5 * s]),
        domain=("x_nonneg",), tags=("saturating", "peak", "term_axis"), min_n=6,
        basis=ns_cols, assemble=lambda c, th: [c[0], c[1], c[2], th[0]],
        grid=lambda x, y: [(float(t),) for t in tau_grid], nl_bounds=([0.002 * s], [5 * s]))

    def f_sv(x, b0, b1, b2, b3, t1, t2):
        L1, H1 = _ns_basis(x, t1)
        _, H2 = _ns_basis(x, t2)
        return b0 + b1 * L1 + b2 * H1 + b3 * H2

    def sv_cols(x, th):
        L1, H1 = _ns_basis(x, th[0])
        _, H2 = _ns_basis(x, th[1])
        return [one(x), L1, H1, H2]

    add("svensson", "Svensson", "Nelson–Siegel + β₃·[L(x/τ₂) − e^(−x/τ₂)]", "term_structure",
        "yield curve with a second hump (central-bank standard)", ["β0", "β1", "β2", "β3", "τ1", "τ2"], f_sv,
        bounds=([-inf] * 4 + [0.002 * s] * 2, [inf] * 4 + [5 * s] * 2),
        domain=("x_nonneg",), tags=("peak", "term_axis"), min_n=12,
        basis=sv_cols, assemble=lambda c, th: [c[0], c[1], c[2], c[3], th[0], th[1]],
        grid=lambda x, y: [(float(a), float(b)) for i, a in enumerate(tau_grid) for b in tau_grid[i + 1:]],
        nl_bounds=([0.002 * s] * 2, [5 * s] * 2))

    def f_heston(x, v0, theta, kappa):
        L, _ = _ns_basis(x, 1.0 / kappa)
        return np.sqrt(np.maximum(theta + (v0 - theta) * L, 0))

    def init_heston(x, y):
        def build(kappa):
            L, _ = _ns_basis(x, 1.0 / kappa)
            return [1 - L, L], lambda c: [max(c[1], 1e-8), max(c[0], 1e-8), kappa]
        return _grid(build, np.array([0.5, 2.0, 8.0, 30.0]) / s, y ** 2)

    add("heston_vol_term", "Heston vol term structure", "√(θ + (v₀ − θ)·(1 − e^(−κx))/(κx))", "term_structure",
        "ATM implied-vol term structure: short-end variance v₀ reverts to θ at speed κ", ["v0", "θ", "κ"], f_heston, init_heston,
        ([0, 0, 0.01 / s], [inf, inf, 1000 / s]), domain=("x_nonneg", "y_pos"),
        tags=("saturating", "term_axis"), min_n=5)

    # ---- volatility smile (not separable: the square root wraps the linear part)
    def f_svi(x, a, b, rho, m, sig):
        d = np.asarray(x, float) - m
        return np.sqrt(np.maximum(a + b * (rho * d + np.sqrt(d * d + sig * sig)), 1e-12))

    def init_svi(x, y):
        xs, ys = _smoothed(x, y)
        grid = [(m, sg, r) for m in (float(xs[np.argmin(ys)]), x0 + s / 2)
                for sg in (s / 20, s / 6, s / 2) for r in (-0.6, 0.0, 0.6)]

        def build(t):
            m, sg, r = t
            d = x - m
            return [one(x), r * d + np.sqrt(d * d + sg * sg)], lambda c: [c[0], max(c[1], 1e-8), r, m, sg]
        return _grid(build, grid, y ** 2)

    add("svi_smile", "SVI smile", "√(a + b·(ρ(x − m) + √((x − m)² + σ²)))", "volatility",
        "implied-volatility smile / skew (x is log-moneyness)", ["a", "b", "ρ", "m", "σ"], f_svi, init_svi,
        ([-inf, 0, -0.999, x0 - s, 1e-4 * s], [inf, inf, 0.999, x0 + 2 * s, 10 * s]),
        domain=("y_pos",), tags=("peak",), min_n=8)

    # ---- fixed income
    def f_dcf(x, A, B, N):
        af, df = _annuity(x, N, unit)
        return A * af + B * df

    add("dcf_price", "DCF price", "A·(1 − e^(−rN))/r + B·e^(−rN)", "fixed_income",
        f"bond price vs yield (convexity): coupon A, face B, maturity N (r = x × {unit:g})", ["A", "B", "N"], f_dcf,
        bounds=([0, 0, 0.05], [inf, inf, 100]), domain=("y_pos",), min_n=5,
        basis=lambda x, th: list(_annuity(x, th[0], unit)), assemble=lambda c, th: [c[0], c[1], th[0]],
        grid=lambda x, y: [(N,) for N in (0.5, 2.0, 5.0, 10.0, 20.0, 30.0)], nl_bounds=([0.05], [100.0]),
        coef_bounds=([0.0, 0.0], [inf, inf]))

    # ---- credit
    add("hazard_cdf", "Constant hazard", "a·(1 − e^(−λx))", "credit", "cumulative default probability under a constant hazard", ["a", "λ"],
        lambda x, a, lam: a * -np.expm1(-lam * np.asarray(x, float)),
        bounds=([-inf, 1e-4 / s], [inf, 1e4 / s]), domain=("x_nonneg",), tags=("saturating", "term_axis"),
        basis=lambda x, th: [-np.expm1(-th[0] * np.asarray(x, float))], assemble=lambda c, th: [c[0], th[0]],
        grid=lambda x, y: [(lam / s,) for lam in (0.3, 1.0, 3.0, 10.0)], nl_bounds=([1e-4 / s], [1e4 / s]))
    add("weibull_cdf", "Weibull default curve", "a·(1 − exp(−(x/c)^k))", "credit",
        "cumulative default probability with a time-varying hazard (k > 1: rising hazard)", ["a", "c", "k"],
        lambda x, a, c, k: a * -np.expm1(-np.power(np.maximum(np.asarray(x, float), 0) / c, k)),
        bounds=([-inf, 1e-3 * s, 0.2], [inf, 100 * s, 10]), domain=("x_nonneg",),
        tags=("sigmoid", "saturating", "term_axis"), min_n=6,
        basis=lambda x, th: [-np.expm1(-np.power(np.maximum(np.asarray(x, float), 0) / th[0], th[1]))],
        assemble=lambda c, th: [c[0], th[0], th[1]],
        grid=lambda x, y: [(c, k) for c in (xmed, s / 2, s) for k in (0.7, 1.5, 3.0)],
        nl_bounds=([1e-3 * s, 0.2], [100 * s, 10.0]))

    # ---- options
    add("black_scholes", "Black–Scholes shape", "a·C_BS(x; K, v) + b·(x − K)", "options",
        "option price vs underlying: call (b ≈ 0) or put (b ≈ −a); v = σ√T", ["a", "b", "K", "v"],
        lambda x, a, b, K, v: a * _bs_call(x, K, v) + b * (np.asarray(x, float) - K),
        bounds=([-inf, -inf, max(x0, 1e-9) * 0.1, 1e-3], [inf, inf, (x0 + s) * 10, 5]),
        domain=("x_pos",), tags=("kink",), min_n=8,
        basis=lambda x, th: [_bs_call(x, th[0], th[1]), np.asarray(x, float) - th[0]],
        assemble=lambda c, th: [c[0], c[1], th[0], th[1]],
        grid=lambda x, y: [(float(K), v) for K in np.quantile(x, [0.25, 0.5, 0.75]) for v in (0.05, 0.15, 0.4)],
        nl_bounds=([max(x0, 1e-9) * 0.1, 1e-3], [(x0 + s) * 10, 5.0]))
    return lib


def link_versions(sc, family):
    """GLM-style candidates: mean = inverse_link(g(x)) for binomial and Poisson targets."""
    if family.link not in ("logit", "log") or family.name == "lognormal":
        return []
    outer = "expit" if family.link == "logit" else "exp"
    base = {c.key: c for c in build_library(sc)}
    out = []
    for key in ("linear", "quadratic", "cubic", "log", "sqrt", "hinge", "softplus"):
        c = base[key]
        out.append(replace(
            c, key=f"{family.link}_{key}", label=f"{family.link}–{c.label.lower()}", group="link",
            formula=f"{outer}({c.formula})",
            meaning=f"{c.label.lower()} shape on the {family.link} scale",
            func=(lambda f: lambda x, *p: family.inverse_link(f(x, *p)))(c.func),
            init=(lambda g: lambda x, y: g(x, family.working_response(y)))(c.init),
            link=True))
    return out


def domain_ok(c, x, y):
    checks = {
        "x_pos": lambda: np.all(x > 0),
        "x_nonneg": lambda: np.all(x >= 0),
        "x_same_sign": lambda: np.all(x > 0) or np.all(x < 0),
        "y_pos": lambda: np.all(y > 0),
    }
    if not all(checks[d]() for d in c.domain):
        return False
    return len(x) >= max(c.min_n, 2 * c.k + 2)


def list_candidates():
    """Table of every built-in candidate (for browsing the library)."""
    lib = build_library(Scale(0.0, 1.0, 0.5, 1.0))
    return pd.DataFrame([{
        "key": c.key, "name": c.label, "group": GROUPS[c.group], "formula": c.formula,
        "parameters": c.k, "meaning": c.meaning,
        "domain": ", ".join(sorted(c.domain)) or "-",
    } for c in lib])
