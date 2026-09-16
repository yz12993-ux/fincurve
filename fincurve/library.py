"""Library of candidate y = f(x) forms, including finance-specific curves.

Each candidate knows its formula, parameter bounds, the data domain it needs
(e.g. x > 0), and how to build good starting values. Starting values come from
grids over the nonlinear parameters with the linear ones solved by least squares,
which is what makes multi-parameter curves such as Svensson or SVI fit reliably.
"""
from dataclasses import dataclass, replace
from typing import Callable, Optional

import numpy as np
import pandas as pd
from scipy.signal import lombscargle
from scipy.special import expit, ndtr

GROUPS = {
    "baseline": "基准",
    "polynomial": "多项式",
    "growth_decay": "指数增长/衰减",
    "power_law": "幂律",
    "diminishing": "边际递减",
    "saturation": "饱和",
    "sigmoid": "S 形",
    "peak": "峰形",
    "piecewise": "分段/拐点",
    "periodic": "周期",
    "term_structure": "期限结构",
    "volatility": "波动率",
    "fixed_income": "固定收益",
    "credit": "信用/违约",
    "options": "期权",
    "link": "广义线性（带连接函数）",
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

    @property
    def k(self):
        return len(self.params)


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

    def add(key, label, formula, group, meaning, params, func, init, bounds=None,
            domain=(), tags=(), min_n=0, uses_u=False):
        lib.append(Candidate(key, label, formula, group, meaning, tuple(params), func, init,
                             bounds, frozenset(domain), frozenset(tags), min_n, uses_u))

    # ---- baseline and polynomials
    add("constant", "常数", "a", "baseline", "y 与 x 无关（基准）", ["a"],
        lambda x, a: a * one(x), lambda x, y: [[float(np.mean(y))]])
    add("linear", "线性", "a + b·u", "polynomial", "常数斜率：Beta、久期近似、线性定价", ["a", "b"],
        lambda x, a, b: a + b * U(x),
        lambda x, y: [list(np.polyfit(U(x), y, 1)[::-1])], uses_u=True)
    add("quadratic", "二次", "a + b·u + c·u²", "polynomial", "U 形/倒 U 形：最优点、凸性", ["a", "b", "c"],
        lambda x, a, b, c: a + b * U(x) + c * U(x) ** 2,
        lambda x, y: [list(np.polyfit(U(x), y, 2)[::-1])], tags=("peak",), uses_u=True)
    add("cubic", "三次", "a + b·u + c·u² + d·u³", "polynomial", "带拐点的灵活曲线（外推危险）", ["a", "b", "c", "d"],
        lambda x, a, b, c, d: a + b * U(x) + c * U(x) ** 2 + d * U(x) ** 3,
        lambda x, y: [list(np.polyfit(U(x), y, 3)[::-1])], tags=("peak", "flexible"), uses_u=True)

    # ---- exponential family
    def init_exp(x, y):
        u, st = U(x), []
        if np.all(y > 0):
            b, la = np.polyfit(u, np.log(y), 1)
            st.append([np.exp(la), b])
        st += _grid(lambda b: ([np.exp(b * u)], lambda c: [c[0], b]), [-3.0, -1.0, 1.0, 3.0], y, top=2)
        return st

    add("exponential", "指数", "a·e^(b·u)", "growth_decay", "复利增长、指数衰减", ["a", "b"],
        lambda x, a, b: a * np.exp(b * U(x)), init_exp, ([-inf, -50], [inf, 50]), uses_u=True)

    add("exp_offset", "指数趋近", "c + a·e^(−κ·u)", "growth_decay",
        "均值回复、向长期水平收敛（波动率/利差期限结构）", ["c", "a", "κ"],
        lambda x, c, a, k: c + a * np.exp(-k * U(x)),
        lambda x, y: _grid(lambda k: ([one(x), np.exp(-k * U(x))], lambda c: [c[0], c[1], k]),
                           [1.0, 3.0, 10.0, 30.0, -2.0], y),
        ([-inf, -inf, -30], [inf, inf, 300]), tags=("saturating", "term_axis"), uses_u=True)

    add("double_exp", "双指数", "c + a₁·e^(−(κ₂+Δ)·u) + a₂·e^(−κ₂·u)", "growth_decay",
        "两因子均值回复：快、慢两个回复速度", ["c", "a1", "a2", "κ2", "Δ"],
        lambda x, c, a1, a2, k2, dk: c + a1 * np.exp(-(k2 + dk) * U(x)) + a2 * np.exp(-k2 * U(x)),
        lambda x, y: _grid(lambda t: ([one(x), np.exp(-(t[0] + t[1]) * U(x)), np.exp(-t[0] * U(x))],
                                      lambda c: [c[0], c[1], c[2], t[0], t[1]]),
                           [(k2, dk) for k2 in (0.5, 2.0, 6.0) for dk in (5.0, 20.0, 60.0)], y),
        ([-inf, -inf, -inf, 0, 0], [inf, inf, inf, 100, 500]), tags=("term_axis",), min_n=12, uses_u=True)

    # ---- power laws
    def init_power(x, y):
        st = []
        for sgn in (1, -1):
            if np.all(sgn * y > 0):
                b, la = np.polyfit(np.log(x), np.log(sgn * y), 1)
                st.append([sgn * np.exp(la), b])
        return st + _grid(lambda b: ([x ** b], lambda c: [c[0], b]), [0.5, 1.0, -0.5, 2.0], y, top=2)

    add("power", "幂律", "a·x^b", "power_law", "规模律：冲击成本平方根律、波动率时间标度、分布尾部", ["a", "b"],
        lambda x, a, b: a * np.power(x, b), init_power, ([-inf, -10], [inf, 10]), domain=("x_pos",))
    add("power_offset", "幂律+偏移", "a + b·x^c", "power_law", "带底数的幂律（固定成本 + 规模项）", ["a", "b", "c"],
        lambda x, a, b, c: a + b * np.power(x, c),
        lambda x, y: _grid(lambda c: ([one(x), x ** c], lambda k: [k[0], k[1], c]), [0.25, 0.5, 1.0, 2.0, -1.0], y),
        ([-inf, -inf, -5], [inf, inf, 5]), domain=("x_pos",))
    add("tempered_power", "截断幂律", "a·x^b·e^(−c·x/s)", "power_law", "先幂律增长、后指数截断", ["a", "b", "c"],
        lambda x, a, b, c: a * np.power(x, b) * np.exp(-c * np.asarray(x, float) / s),
        lambda x, y: _grid(lambda t: ([x ** t[0] * np.exp(-t[1] * x / s)], lambda k: [k[0], t[0], t[1]]),
                           [(b, c) for b in (0.5, 1.0, 2.0, -0.5) for c in (0.0, 1.0, 3.0)], y),
        ([-inf, -10, -20], [inf, 10, 50]), domain=("x_pos",), tags=("peak",))

    # ---- diminishing returns
    add("log", "对数", "a + b·ln x", "diminishing", "边际递减、学习曲线、对数收益", ["a", "b"],
        lambda x, a, b: a + b * np.log(x),
        lambda x, y: [list(np.polyfit(np.log(x), y, 1)[::-1])], domain=("x_pos",))
    add("log_shift", "平移对数", "a + b·ln(x + c)", "diminishing", "允许 x 含 0 的边际递减", ["a", "b", "c"],
        lambda x, a, b, c: a + b * np.log(np.asarray(x, float) + c),
        lambda x, y: _grid(lambda c: ([one(x), np.log(x + c)], lambda k: [k[0], k[1], c]),
                           [-x0 + s * f for f in (0.01, 0.1, 0.5, 2.0)], y),
        ([-inf, -inf, -x0 + 1e-3 * s], [inf, inf, 10 * s]))
    add("sqrt", "平方根", "a + b·√x", "diminishing", "扩散、√t 标度、冲击成本", ["a", "b"],
        lambda x, a, b: a + b * np.sqrt(x),
        lambda x, y: [list(np.polyfit(np.sqrt(x), y, 1)[::-1])], domain=("x_nonneg",))
    add("hyperbolic", "反比", "a + b/x", "diminishing", "单位成本摊薄、平均成本", ["a", "b"],
        lambda x, a, b: a + b / np.asarray(x, float),
        lambda x, y: [list(np.polyfit(1 / x, y, 1)[::-1])], domain=("x_same_sign",), tags=("saturating",))
    add("rational", "有理函数", "(a + b·u)/(1 + c·u)", "diminishing", "可饱和也可反比的灵活单调形状", ["a", "b", "c"],
        lambda x, a, b, c: (a + b * U(x)) / (1 + c * U(x)),
        lambda x, y: _grid(lambda c: ([1 / (1 + c * U(x)), U(x) / (1 + c * U(x))], lambda k: [k[0], k[1], c]),
                           [0.0, 0.5, 2.0, 10.0], y),
        ([-inf, -inf, -0.9], [inf, inf, 200]), tags=("saturating",), uses_u=True)

    # ---- saturation
    add("michaelis_menten", "饱和（MM）", "V·x/(K + x)", "saturation",
        "容量上限：渠道/策略容量、资金规模与收益", ["V", "K"],
        lambda x, V, K: V * np.asarray(x, float) / (K + np.asarray(x, float)),
        lambda x, y: _grid(lambda K: ([x / (K + x)], lambda c: [c[0], K]), [xmed / 2, xmed, 2 * xmed, s], y),
        ([-inf, 1e-6 * s], [inf, 100 * s]), domain=("x_nonneg",), tags=("saturating",))
    add("hill", "Hill 饱和", "d + V/(1 + (K/x)ⁿ)", "saturation", "带陡峭度的饱和 S 形", ["d", "V", "K", "n"],
        lambda x, d, V, K, n: d + V / (1 + np.power(K / np.maximum(np.asarray(x, float), 1e-300), n)),
        lambda x, y: _grid(lambda t: ([one(x), 1 / (1 + (t[0] / np.maximum(x, 1e-300)) ** t[1])],
                                      lambda c: [c[0], c[1], t[0], t[1]]),
                           [(K, n) for K in (xmed / 2, xmed, 2 * xmed) for n in (1.0, 2.0, 4.0)], y),
        ([-inf, -inf, 1e-6 * s, 0.2], [inf, inf, 100 * s, 12]), domain=("x_nonneg",),
        tags=("sigmoid", "saturating"), min_n=10)

    # ---- sigmoids
    sig_grid = [(m, r) for m in (0.2, 0.35, 0.5, 0.65, 0.8) for r in (5.0, 15.0, 40.0)]
    add("logistic", "逻辑斯蒂", "d + K/(1 + e^(−r·(u − m)))", "sigmoid",
        "S 形：违约率–评分、渗透率、状态切换", ["d", "K", "r", "m"],
        lambda x, d, K, r, m: d + K * expit(r * (U(x) - m)),
        lambda x, y: _grid(lambda t: ([one(x), expit(t[1] * (U(x) - t[0]))], lambda c: [c[0], c[1], t[1], t[0]]),
                           sig_grid, y),
        ([-inf, -inf, 0.1, -0.5], [inf, inf, 500, 1.5]), tags=("sigmoid",), min_n=8, uses_u=True)
    add("gompertz", "Gompertz", "d + K·exp(−e^(−r·(u − m)))", "sigmoid", "不对称 S 形：一侧拖得更长", ["d", "K", "r", "m"],
        lambda x, d, K, r, m: d + K * np.exp(-np.exp(-r * (U(x) - m))),
        lambda x, y: _grid(lambda t: ([one(x), np.exp(-np.exp(-t[1] * (U(x) - t[0])))],
                                      lambda c: [c[0], c[1], t[1], t[0]]), sig_grid, y),
        ([-inf, -inf, 0.1, -0.5], [inf, inf, 500, 1.5]), tags=("sigmoid",), min_n=8, uses_u=True)

    # ---- peaks
    def peak_init(kernel):
        def init(x, y):
            xs, ys = _smoothed(x, y)
            centres = {float(U(xs[np.argmax(ys)])), float(U(xs[np.argmin(ys)])), 0.5}
            grid = [(m, w) for m in centres for w in (0.05, 0.15, 0.4)]
            return _grid(lambda t: ([one(x), kernel((U(x) - t[0]) / t[1])], lambda c: [c[0], c[1], t[0], t[1]]),
                         grid, y)
        return init

    add("gaussian_peak", "高斯峰", "d + a·exp(−(u − m)²/(2w²))", "peak", "单峰：最优点附近的响应", ["d", "a", "m", "w"],
        lambda x, d, a, m, w: d + a * np.exp(-((U(x) - m) ** 2) / (2 * w * w)),
        peak_init(lambda z: np.exp(-z * z / 2)),
        ([-inf, -inf, -0.5, 0.005], [inf, inf, 1.5, 5]), tags=("peak",), min_n=8, uses_u=True)
    add("lorentzian", "洛伦兹峰", "d + a/(1 + ((u − m)/w)²)", "peak", "尖峰厚尾的单峰", ["d", "a", "m", "w"],
        lambda x, d, a, m, w: d + a / (1 + ((U(x) - m) / w) ** 2),
        peak_init(lambda z: 1 / (1 + z * z)),
        ([-inf, -inf, -0.5, 0.005], [inf, inf, 1.5, 5]), tags=("peak",), min_n=8, uses_u=True)

    # ---- piecewise
    add("hinge", "单拐点折线", "a + b·u + c·max(0, u − k)", "piecewise",
        "阈值效应：税档、杠杆/保证金约束、政策生效点", ["a", "b", "c", "k"],
        lambda x, a, b, c, k: a + b * U(x) + c * np.maximum(0, U(x) - k),
        lambda x, y: _grid(lambda k: ([one(x), U(x), np.maximum(0, U(x) - k)], lambda c: [c[0], c[1], c[2], k]),
                           [0.15, 0.3, 0.45, 0.6, 0.75, 0.85], y),
        ([-inf, -inf, -inf, 0.02], [inf, inf, inf, 0.98]), tags=("kink",), min_n=8, uses_u=True)
    add("softplus", "平滑曲棍球杆", "a + b·u + c·w·ln(1 + e^((u − k)/w))", "piecewise",
        "期权型 payoff、保底/止损条款", ["a", "b", "c", "k", "w"],
        lambda x, a, b, c, k, w: a + b * U(x) + c * w * np.logaddexp(0, (U(x) - k) / w),
        lambda x, y: _grid(lambda t: ([one(x), U(x), t[1] * np.logaddexp(0, (U(x) - t[0]) / t[1])],
                                      lambda c: [c[0], c[1], c[2], t[0], t[1]]),
                           [(k, w) for k in (0.2, 0.4, 0.6, 0.8) for w in (0.02, 0.1)], y),
        ([-inf, -inf, -inf, -0.2, 0.002], [inf, inf, inf, 1.2, 1.0]), tags=("kink",), min_n=10, uses_u=True)

    # ---- periodic
    def init_sine(x, y):
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

        def build(P):
            th = 2 * np.pi * (x - x0) / P
            return ([one(x), u, np.sin(th), np.cos(th)],
                    lambda c: [c[0], c[1], np.hypot(c[2], c[3]), P, np.arctan2(c[3], c[2])])
        return _grid(build, picks, y)

    add("sine_trend", "周期+趋势", "a + b·u + c·sin(2π·x/P + φ)", "periodic", "季节性、日内/周内效应",
        ["a", "b", "c", "P", "φ"],
        lambda x, a, b, c, P, ph: a + b * U(x) + c * np.sin(2 * np.pi * (np.asarray(x, float) - x0) / P + ph),
        init_sine, ([-inf, -inf, -inf, s / 400, -10], [inf, inf, inf, 2 * s, 10]),
        tags=("periodic",), min_n=20, uses_u=True)

    # ---- term structures
    tau_grid = s * np.array([0.01, 0.03, 0.07, 0.15, 0.3, 0.6])

    def f_ns(x, b0, b1, b2, tau):
        L, H = _ns_basis(x, tau)
        return b0 + b1 * L + b2 * H

    def init_ns(x, y):
        def build(tau):
            L, H = _ns_basis(x, tau)
            return [one(x), L, H], lambda c: [c[0], c[1], c[2], tau]
        return _grid(build, tau_grid, y)

    add("nelson_siegel", "Nelson–Siegel", "β₀ + β₁·L(x/τ) + β₂·[L(x/τ) − e^(−x/τ)]", "term_structure",
        "收益率/远期曲线：水平、斜率、曲率三因子", ["β0", "β1", "β2", "τ"], f_ns, init_ns,
        ([-inf, -inf, -inf, 0.002 * s], [inf, inf, inf, 5 * s]),
        domain=("x_nonneg",), tags=("saturating", "peak", "term_axis"), min_n=6)

    def f_sv(x, b0, b1, b2, b3, t1, t2):
        L1, H1 = _ns_basis(x, t1)
        _, H2 = _ns_basis(x, t2)
        return b0 + b1 * L1 + b2 * H1 + b3 * H2

    def init_sv(x, y):
        def build(t):
            L1, H1 = _ns_basis(x, t[0])
            _, H2 = _ns_basis(x, t[1])
            return [one(x), L1, H1, H2], lambda c: [c[0], c[1], c[2], c[3], t[0], t[1]]
        pairs = [(a, b) for i, a in enumerate(tau_grid) for b in tau_grid[i + 1:]]
        return _grid(build, pairs, y)

    add("svensson", "Svensson", "Nelson–Siegel + β₃·[L(x/τ₂) − e^(−x/τ₂)]", "term_structure",
        "收益率曲线：多一个驼峰，央行常用", ["β0", "β1", "β2", "β3", "τ1", "τ2"], f_sv, init_sv,
        ([-inf] * 4 + [0.002 * s] * 2, [inf] * 4 + [5 * s] * 2),
        domain=("x_nonneg",), tags=("peak", "term_axis"), min_n=12)

    def f_heston(x, v0, theta, kappa):
        L, _ = _ns_basis(x, 1.0 / kappa)
        return np.sqrt(np.maximum(theta + (v0 - theta) * L, 0))

    def init_heston(x, y):
        def build(kappa):
            L, _ = _ns_basis(x, 1.0 / kappa)
            return [1 - L, L], lambda c: [max(c[1], 1e-8), max(c[0], 1e-8), kappa]
        return _grid(build, np.array([0.5, 2.0, 8.0, 30.0]) / s, y ** 2)

    add("heston_vol_term", "Heston 波动率期限结构", "√(θ + (v₀ − θ)·(1 − e^(−κx))/(κx))", "term_structure",
        "平值隐含波动率期限结构：短端方差 v₀ 以速度 κ 回复到长期 θ", ["v0", "θ", "κ"], f_heston, init_heston,
        ([0, 0, 0.01 / s], [inf, inf, 1000 / s]), domain=("x_nonneg", "y_pos"),
        tags=("saturating", "term_axis"), min_n=5)

    # ---- volatility smile
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

    add("svi_smile", "SVI 微笑", "√(a + b·(ρ(x − m) + √((x − m)² + σ²)))", "volatility",
        "隐含波动率微笑/偏斜（x 为对数行权价）", ["a", "b", "ρ", "m", "σ"], f_svi, init_svi,
        ([-inf, 0, -0.999, x0 - s, 1e-4 * s], [inf, inf, 0.999, x0 + 2 * s, 10 * s]),
        domain=("y_pos",), tags=("peak",), min_n=8)

    # ---- fixed income
    def f_dcf(x, A, B, N):
        af, df = _annuity(x, N, unit)
        return A * af + B * df

    add("dcf_price", "现金流贴现价格", "A·(1 − e^(−rN))/r + B·e^(−rN)", "fixed_income",
        f"债券价格–收益率（凸性）：A 年息、B 面值、N 年限（r = x × {unit:g}）", ["A", "B", "N"], f_dcf,
        lambda x, y: _grid(lambda N: (list(_annuity(x, N, unit)), lambda c: [max(c[0], 0.0), max(c[1], 0.0), N]),
                           [0.5, 2.0, 5.0, 10.0, 20.0, 30.0], y),
        ([0, 0, 0.05], [inf, inf, 100]), domain=("y_pos",), min_n=5)

    # ---- credit
    add("hazard_cdf", "常数违约强度", "a·(1 − e^(−λx))", "credit", "累计违约率期限结构（强度恒定）", ["a", "λ"],
        lambda x, a, lam: a * -np.expm1(-lam * np.asarray(x, float)),
        lambda x, y: _grid(lambda lam: ([-np.expm1(-lam * x)], lambda c: [c[0], lam]),
                           np.array([0.3, 1.0, 3.0, 10.0]) / s, y),
        ([-inf, 1e-4 / s], [inf, 1e4 / s]), domain=("x_nonneg",), tags=("saturating", "term_axis"))
    add("weibull_cdf", "Weibull 违约曲线", "a·(1 − exp(−(x/c)^k))", "credit",
        "累计违约率（k > 1 强度随时间上升）", ["a", "c", "k"],
        lambda x, a, c, k: a * -np.expm1(-np.power(np.maximum(np.asarray(x, float), 0) / c, k)),
        lambda x, y: _grid(lambda t: ([-np.expm1(-(np.maximum(x, 0) / t[0]) ** t[1])], lambda q: [q[0], t[0], t[1]]),
                           [(c, k) for c in (xmed, s / 2, s) for k in (0.7, 1.5, 3.0)], y),
        ([-inf, 1e-3 * s, 0.2], [inf, 100 * s, 10]), domain=("x_nonneg",),
        tags=("sigmoid", "saturating", "term_axis"), min_n=6)

    # ---- options
    add("black_scholes", "Black–Scholes 期权形状", "a·C_BS(x; K, v) + b·(x − K)", "options",
        "期权价格–标的价格：b ≈ 0 为看涨，b ≈ −a 为看跌；v = σ√T", ["a", "b", "K", "v"],
        lambda x, a, b, K, v: a * _bs_call(x, K, v) + b * (np.asarray(x, float) - K),
        lambda x, y: _grid(lambda t: ([_bs_call(x, t[0], t[1]), x - t[0]], lambda c: [c[0], c[1], t[0], t[1]]),
                           [(K, v) for K in np.quantile(x, [0.25, 0.5, 0.75]) for v in (0.05, 0.15, 0.4)], y),
        ([-inf, -inf, max(x0, 1e-9) * 0.1, 1e-3], [inf, inf, (x0 + s) * 10, 5]),
        domain=("x_pos",), tags=("kink",), min_n=8)
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
            c, key=f"{family.link}_{key}", label=f"{family.link}-{c.label}", group="link",
            formula=f"{outer}({c.formula})",
            meaning=f"{'logit' if family.link == 'logit' else 'log'} 尺度上是{c.label}形状",
            func=(lambda f: lambda x, *p: family.inverse_link(f(x, *p)))(c.func),
            init=(lambda g: lambda x, y: g(x, family.working_response(y)))(c.init)))
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
        "key": c.key, "名称": c.label, "类别": GROUPS[c.group], "公式": c.formula,
        "参数个数": c.k, "金融含义": c.meaning,
        "定义域": ", ".join(sorted(c.domain)) or "-",
    } for c in lib])
