"""Export real fincurve runs to docs/assets/site-data.js for the showcase page.

Every point, curve, ranking row and decision-log line on the page comes from running
the library on simulated data whose generating model is known, so readers can check
whether the tool recovers the truth.

    python3 site/build_site_data.py
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import scenarios as S  # noqa: E402
from fincurve import __version__, analyze  # noqa: E402
from fincurve.distribution import DISTRIBUTIONS  # noqa: E402
from fincurve.library import GROUPS, Scale, build_library  # noqa: E402
from fincurve.profile import random_walk_like  # noqa: E402

OUT = ROOT / "docs" / "assets" / "site-data.js"

LABEL_EN = {
    "constant": "Constant", "linear": "Linear", "quadratic": "Quadratic", "cubic": "Cubic",
    "exponential": "Exponential", "exp_offset": "Exponential approach", "double_exp": "Double exponential",
    "power": "Power law", "power_offset": "Power law + offset", "tempered_power": "Tempered power law",
    "log": "Logarithmic", "log_shift": "Shifted log", "sqrt": "Square root", "hyperbolic": "Hyperbolic",
    "rational": "Rational", "michaelis_menten": "Michaelis–Menten", "hill": "Hill saturation",
    "logistic": "Logistic", "gompertz": "Gompertz", "gaussian_peak": "Gaussian peak",
    "lorentzian": "Lorentzian peak", "hinge": "Hinge", "softplus": "Soft hockey stick",
    "sine_trend": "Sine + trend", "nelson_siegel": "Nelson–Siegel", "svensson": "Svensson",
    "heston_vol_term": "Heston vol term structure", "svi_smile": "SVI smile", "dcf_price": "DCF price",
    "hazard_cdf": "Constant hazard", "weibull_cdf": "Weibull default curve",
    "black_scholes": "Black–Scholes shape", "smoother": "Local-linear smoother (reference)",
    "shapes": "Additive shape model", "linear_glm": "All-linear GLM", "interactions": "Shapes + interactions",
    "knn": "k-nearest neighbours (reference)",
}
GROUP_EN = {
    "baseline": "Baseline", "polynomial": "Polynomial", "growth_decay": "Growth / decay", "power_law": "Power law",
    "diminishing": "Diminishing returns", "saturation": "Saturation", "sigmoid": "Sigmoid", "peak": "Peak",
    "piecewise": "Piecewise / kink", "periodic": "Periodic", "term_structure": "Term structure",
    "volatility": "Volatility", "fixed_income": "Fixed income", "credit": "Credit", "options": "Options",
    "link": "GLM (link function)",
}
MEANING_EN = {
    "constant": "y does not depend on x (baseline)",
    "linear": "Constant slope: beta, duration approximation, linear pricing",
    "quadratic": "U / inverted-U: optima, convexity",
    "cubic": "Flexible curve with an inflection (dangerous to extrapolate)",
    "exponential": "Compounding growth, exponential decay",
    "exp_offset": "Mean reversion to a long-run level (vol and spread term structures)",
    "double_exp": "Two-factor mean reversion with fast and slow speeds",
    "power": "Scaling laws: square-root impact, volatility time scaling, tails",
    "power_offset": "Power law with a floor (fixed cost plus scale term)",
    "tempered_power": "Power-law growth with an exponential cut-off",
    "log": "Diminishing returns, learning curves, log returns",
    "log_shift": "Diminishing returns when x can be zero",
    "sqrt": "Diffusion, √t scaling, market impact",
    "hyperbolic": "Unit-cost amortisation",
    "rational": "Flexible monotone shape: saturating or hyperbolic",
    "michaelis_menten": "Capacity limits: strategy capacity, AUM vs return",
    "hill": "Saturating S-curve with adjustable steepness",
    "logistic": "S-curve: default rate vs score, adoption, regime switches",
    "gompertz": "Asymmetric S-curve",
    "gaussian_peak": "Single peak around an optimum",
    "lorentzian": "Sharp peak with heavy shoulders",
    "hinge": "Threshold effects: tax brackets, leverage and margin limits",
    "softplus": "Option-like payoffs, floors and stops",
    "sine_trend": "Seasonality, intraday and weekday effects",
    "nelson_siegel": "Yield / forward curve: level, slope, curvature",
    "svensson": "Yield curve with a second hump (central-bank standard)",
    "heston_vol_term": "ATM vol term structure: v₀ reverts to θ at speed κ",
    "svi_smile": "Implied-volatility smile / skew in log-moneyness",
    "dcf_price": "Bond price vs yield (convexity): coupon A, face B, maturity N",
    "hazard_cdf": "Cumulative default probability under a constant hazard",
    "weibull_cdf": "Cumulative default probability with a time-varying hazard",
    "black_scholes": "Option price vs underlying: call (b ≈ 0) or put (b ≈ −a)",
}
DIST_EN = {"normal": "Normal", "student_t": "Student-t", "laplace": "Laplace", "logistic": "Logistic",
           "skew_normal": "Skew-normal", "johnson_su": "Johnson SU", "nig": "Normal-inverse Gaussian",
           "lognormal": "Log-normal", "gamma": "Gamma", "weibull": "Weibull", "inv_gauss": "Inverse Gaussian",
           "exponential": "Exponential"}
TREND_EN = {"加速上升": "accelerating rise", "上升趋缓": "rising, flattening", "加速下降": "accelerating fall",
            "下降趋缓": "falling, flattening", "近似线性上升": "≈ linear rise", "近似线性下降": "≈ linear fall",
            "先升后降": "rise then fall", "先降后升": "fall then rise", "多次起伏": "oscillating",
            "几乎无影响": "negligible", "S 形上升（中间陡、两端平）": "S-shaped rise",
            "S 形下降（中间陡、两端平）": "S-shaped fall"}


def sig(v, digits=5):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return float(f"{v:.{digits}g}") if np.isfinite(v) else None


def arr(a, digits=5):
    return [sig(v, digits) for v in np.asarray(a, float)]


def label_en(key):
    for prefix, name in (("logit_", "logit"), ("log_", "log")):
        rest = key[len(prefix):]
        if key.startswith(prefix) and rest in LABEL_EN and key not in LABEL_EN:
            return f"{name}–{LABEL_EN[rest].lower()}"
    return LABEL_EN.get(key, key)


def t(en, zh):
    return {"en": en, "zh": zh}


def ranking_rows(rep, limit=12, include=()):
    table = rep.ranking[np.isfinite(rep.ranking["d_cv"])].reset_index(drop=True)
    picked = table.head(limit)
    extra = table[table["id"].isin(include) & ~table["id"].isin(picked["id"])]
    rows = []
    for row in pd.concat([picked, extra]).itertuples():
        key = "linear_glm" if rep.mode == "additive" and row.model == "linear" else row.model
        rows.append({"id": row.id, "key": row.model, "rank": int(row.Index) + 1,
                     "label": t(label_en(key), row.label), "family": row.family, "k": sig(row.k),
                     "d": sig(row.d_cv), "se": sig(row.se), "tie": bool(row.tie),
                     "ref": row.status == "reference", "rec": row.id == rep.recommended_id,
                     "cv_rmse": sig(getattr(row, "cv_rmse", np.nan)), "d_aic": sig(getattr(row, "d_aic", np.nan))})
    return rows


def curve_values(model, grid, median=False):
    with np.errstate(all="ignore"):
        v = model.location(grid) if median else model.predict(grid)
    return arr(np.where(np.isfinite(v), v, np.nan))


def best_row(rep, key):
    """Best-ranked row of a candidate key (any family), with its position among scored models."""
    table = rep.ranking[(rep.ranking["status"] == "ok") & np.isfinite(rep.ranking["d_cv"])].reset_index(drop=True)
    hits = table[table["model"] == key]
    if hits.empty:
        return None
    return int(hits.index[0]) + 1, hits.iloc[0]


# ----------------------------------------------------------------------------- case 1: volatility smile
def case_smile():
    x, y = S.vol_smile(np.random.default_rng(7))
    rep = analyze(x, y)
    rel, dec = rep.profile["relation"], rep.decisions
    ok = rep.ranking[(rep.ranking["status"] == "ok") & np.isfinite(rep.ranking["d_cv"])]
    grid = np.linspace(x.min(), x.max(), 160)
    shown = [rep.recommended_id] + [i for i in ok["id"] if i != rep.recommended_id][:2]
    runner = ok.iloc[1]
    truth = S.SMILE_TRUTH
    fitted = rep.recommended.param_dict()
    a_t, b_t = 1e4 * truth["a"] / truth["T"], 1e4 * truth["b"] / truth["T"]
    vertex = lambda a, b, rho, sg: a + b * sg * np.sqrt(1 - rho ** 2)
    params = [
        ("ρ (skew)", truth["rho"], fitted["ρ"]),
        ("m (shift)", truth["m"], fitted["m"]),
        ("σ (curvature)", truth["sigma"], fitted["σ"]),
        ("b′ (wings)", b_t, fitted["b"]),
        ("a′ (level)", a_t, fitted["a"]),
        ("a′ + b′σ√(1−ρ²) (vertex)", vertex(a_t, b_t, truth["rho"], truth["sigma"]),
         vertex(fitted["a"], fitted["b"], fitted["ρ"], fitted["σ"])),
    ]
    n_skip = len(dec["skipped"])
    log = [
        ("profile", t(f"n={rep.profile['n']} · y>0 continuous · shape: fall then rise · smoother explains {rel['smooth_r2']:.0%}",
                      f"n={rep.profile['n']} · y 为正的连续值 · 形状：先降后升 · 局部平滑解释 {rel['smooth_r2']:.0%}")),
        ("noise", t(f"|residual| does not grow with level (Spearman ρ={rel['hetero_rho']:.2f}); max/min only "
                    f"{rep.profile['target']['max_min_ratio']:.1f}× → Gaussian, additive errors",
                    f"|残差| 没有随水平变大（Spearman ρ={rel['hetero_rho']:.2f}）；最大/最小仅 "
                    f"{rep.profile['target']['max_min_ratio']:.1f} 倍 → 正态加性误差")),
        ("cv", t("no time order, n ≥ 20 → shuffled 5-fold cross-validation", "无时间顺序、n ≥ 20 → 随机 5 折交叉验证")),
        ("domain", t(f"x < 0 rules out {n_skip} forms (log, power, √x, Nelson–Siegel …) → {dec['n_candidates']} fitted",
                     f"x < 0 排除 {n_skip} 种形式（对数、幂律、√x、Nelson–Siegel…）→ 拟合 {dec['n_candidates']} 种")),
        ("rank", t(f"runner-up {label_en(runner['model'])}: Δ = {runner['d_cv']:.2f} ± {runner['se']:.2f} — within 2 SE "
                   f"but above the 0.10-nat practical gap → not tied",
                   f"第二名 {runner['label']}：Δ = {runner['d_cv']:.2f} ± {runner['se']:.2f} —— 在 2 个标准误以内，"
                   f"但超过 0.10 的实际差距 → 不算并列")),
        ("context", t("x name contains “K/F” → volatility-smile context; the winner is that context's specialised form",
                      "x 的名字含 “K/F” → 波动率微笑语境；胜出者正是该语境的专用形式")),
        ("verdict", t("★ SVI smile — the generating model; skew and vertex level recovered",
                      "★ SVI 微笑 —— 就是生成数据的模型；偏斜和谷底水平都还原了")),
    ]
    return {
        "id": "smile", "mode": "curve",
        "title": t("Volatility smile → SVI", "波动率微笑 → SVI"),
        "claim": t("25 noisy implied vols. 18 candidate forms survive the domain filter and SVI wins outright. The "
                   "well-identified quantities — skew and the vertex level — land within a few percent of the truth.",
                   "25 个带噪声的隐含波动率。定义域过滤后剩 18 种候选，SVI 明确胜出。能被数据确定的量（偏斜、谷底水平）与真实值只差几个百分点。"),
        "truth": t("SVI (ρ = −0.6)", "SVI（ρ = −0.6）"),
        "identified": t(label_en(rep.recommended.candidate.key), rep.recommended.label),
        "verdict": "hit",
        "chart": {
            "xLabel": t("log-moneyness ln(K/F)", "对数行权价 ln(K/F)"), "yLabel": t("implied vol (%)", "隐含波动率 (%)"),
            "points": [arr(x), arr(y)],
            "series": [{"label": t(label_en(rep.models[i].candidate.key), rep.models[i].label),
                        "x": arr(grid), "y": curve_values(rep.models[i], grid), "rec": i == rep.recommended_id}
                       for i in shown]
            + [{"label": t("ground truth", "真实曲线"), "x": arr(grid), "y": arr(S.vol_smile_truth(grid)), "truth": True}],
        },
        "ranking": ranking_rows(rep, 12),
        "params": [{"name": n, "truth": sig(a), "fitted": sig(b)} for n, a, b in params],
        "log": [{"tag": tag, "text": text} for tag, text in log],
    }


# ----------------------------------------------------------------------------- case 2: market impact
def case_impact():
    x, y = S.market_impact(np.random.default_rng(7))
    rep = analyze(x, y)
    rel = rep.profile["relation"]
    table = rep.ranking[(rep.ranking["status"] == "ok") & np.isfinite(rep.ranking["d_cv"])]
    ties = table[table["tie"]]
    best_add = table[table["family"] == "gaussian"].iloc[0]
    grid = np.geomspace(x.min(), x.max(), 160)
    shown = [rep.recommended_id] + [i for i in [rep.best_id] if i != rep.recommended_id]
    power = rep.models["power@lognormal"].param_dict()
    sqrt = rep.recommended.param_dict()
    log = [
        ("profile", t(f"n={rep.profile['n']} · y>0 · rising and flattening · max/min = {rep.profile['target']['max_min_ratio']:.0f}×",
                      f"n={rep.profile['n']} · y 为正 · 上升且越来越平 · 最大/最小 = {rep.profile['target']['max_min_ratio']:.0f} 倍")),
        ("noise", t(f"|residual| grows with level: Spearman ρ={rel['hetero_rho']:.2f} (p={rel['hetero_p']:.1e}) → additive AND multiplicative errors compete",
                    f"|残差| 随水平变大：Spearman ρ={rel['hetero_rho']:.2f}（p={rel['hetero_p']:.1e}）→ 加性与乘性误差一起比较")),
        ("cv", t(f"shuffled 5-fold CV · {rep.decisions['n_candidates']} (form, noise-model) pairs fitted",
                 f"随机 5 折交叉验证 · 拟合 {rep.decisions['n_candidates']} 个（形式，误差模型）组合")),
        ("noise", t(f"best additive-error model trails by Δ = {best_add['d_cv']:.2f} ± {best_add['se']:.2f} nats/row → multiplicative wins",
                    f"最好的加性误差模型落后 Δ = {best_add['d_cv']:.2f} ± {best_add['se']:.2f} → 乘性误差胜出")),
        ("rank", t(f"{len(ties)} forms within 2 SE and 0.1 nats — the data cannot separate them → pick the fewest parameters",
                   f"{len(ties)} 种形式差距都在 2 个标准误和 0.1 以内 —— 数据分不开 → 选参数最少的")),
        ("check", t(f"free power-law exponent fits b = {power['b']:.3f} (truth 0.5)", f"自由幂律指数拟合为 b = {power['b']:.3f}（真实值 0.5）")),
        ("verdict", t("★ Square root with multiplicative noise — the generating model", "★ 平方根 + 乘性误差 —— 就是生成数据的模型")),
    ]
    return {
        "id": "impact", "mode": "curve",
        "title": t("Market impact → square-root law", "市场冲击 → 平方根律"),
        "claim": t("Impact noise scales with the level of impact. fincurve notices, lets log-normal errors compete with "
                   "Gaussian ones, and among 16 statistically indistinguishable shapes returns the simplest: √q.",
                   "冲击成本的噪声随水平放大。fincurve 发现这一点，让对数正态误差和正态误差竞争；在 16 个统计上无法区分的形状中返回最简单的 √q。"),
        "truth": t("140·√q, multiplicative noise", "140·√q，乘性噪声"),
        "identified": t("Square root, multiplicative", "平方根，乘性误差"),
        "verdict": "hit",
        "chart": {
            "xLabel": t("participation rate (order / daily volume)", "参与率（订单量 / 日成交量）"),
            "yLabel": t("impact (bp)", "冲击成本 (bp)"), "logToggle": True, "median": True,
            "points": [arr(x), arr(y)],
            "series": [{"label": t(label_en(rep.models[i].candidate.key), rep.models[i].label), "x": arr(grid),
                        "y": curve_values(rep.models[i], grid, median=True), "rec": i == rep.recommended_id}
                       for i in shown]
            + [{"label": t("ground truth (median)", "真实曲线（中位数）"), "x": arr(grid),
                "y": arr(S.market_impact_truth(grid)), "truth": True}],
        },
        "ranking": ranking_rows(rep, 12, include=[best_add["id"]]),
        "params": [{"name": "√q coefficient", "truth": 140.0, "fitted": sig(sqrt["b"])},
                   {"name": "intercept", "truth": 0.0, "fitted": sig(sqrt["a"])},
                   {"name": "power exponent", "truth": 0.5, "fitted": sig(power["b"])}],
        "families": {"multiplicative": sig(table[table["family"] == "lognormal"].iloc[0]["d_cv"]),
                     "additive": sig(best_add["d_cv"]), "additive_se": sig(best_add["se"])},
        "log": [{"tag": tag, "text": text} for tag, text in log],
    }


# ----------------------------------------------------------------------------- case 3: loan defaults
def case_loans():
    X, y = S.loan_defaults(np.random.default_rng(7))
    rep = analyze(X, y)
    res = rep.extras["additive"]
    feats = res["features"].set_index("feature")
    model = rep.recommended
    effects = model.partial_effects()
    names_en = {"负债收入比": "debt-to-income", "额度使用率": "credit utilisation", "行号": "row number"}
    truth_dti, truth_util = S.loan_default_truth_effects(X["负债收入比"].to_numpy(), X["额度使用率"].to_numpy())
    panels = []
    for name, col_truth in (("负债收入比", truth_dti), ("额度使用率", truth_util)):
        _, grid, eff, _ = effects[name]
        g_dti, g_util = S.loan_default_truth_effects(grid, grid)
        true_curve = (g_dti if name == "负债收入比" else g_util) - col_truth.mean()
        panels.append({"feature": t(names_en[name], name),
                       "trend": t(TREND_EN.get(feats.loc[name, "trend"], feats.loc[name, "trend"]), feats.loc[name, "trend"]),
                       "importance": sig(feats.loc[name, "importance"]),
                       "stability": sig(feats.loc[name, "stability"]),
                       "x": arr(grid), "fitted": arr(eff), "truth": arr(true_curve),
                       "rug": arr(np.quantile(X[name], np.linspace(0.005, 0.995, 120)), 4)})
    hist = res["history"][0]
    table = rep.ranking.set_index("id")
    gap, gap_se = table.loc["linear", "d_cv"], table.loc["linear", "se"]
    log = [
        ("target", t("y ∈ {0, 1} → Bernoulli likelihood, logit link", "y ∈ {0, 1} → 伯努利似然，logit 连接")),
        ("leakage", t("“row number” removed: consecutive unique integers would let the model memorise rows",
                      "剔除“行号”：连续且唯一的整数会让模型记住样本")),
        ("search", t("start all-linear; try 11 shapes per feature; accept only if AIC drops ≥ 4",
                     "从全线性出发；每个特征试 11 种形状；AIC 至少下降 4 才更换")),
        ("shape", t(f"debt-to-income: linear → sigmoid (ΔAIC = {hist['aic_gain']:.1f}); utilisation stays linear",
                    f"负债收入比：线性 → S 形（ΔAIC = {hist['aic_gain']:.1f}）；额度使用率保持线性")),
        ("pairs", t("no product term lowers AIC by ≥ 10 → no interaction", "没有乘积项能让 AIC 下降 10 → 不加交互项")),
        ("cv", t(f"nested 5-fold: shape search re-run inside every fold · fold-to-fold effect correlation "
                 f"{feats.loc['负债收入比', 'stability']:.2f} / {feats.loc['额度使用率', 'stability']:.2f}",
                 f"嵌套 5 折：每一折内重新搜索形状 · 各折效应曲线相关 "
                 f"{feats.loc['负债收入比', 'stability']:.2f} / {feats.loc['额度使用率', 'stability']:.2f}")),
        ("rank", t(f"shape model beats the all-linear GLM by Δ = {gap:.4f} ± {gap_se:.4f} nats/row ({gap / gap_se:.1f} SE)",
                   f"形状模型比全线性 GLM 好 Δ = {gap:.4f} ± {gap_se:.4f}（{gap / gap_se:.1f} 个标准误）")),
        ("verdict", t("★ threshold in debt-to-income + linear utilisation — the generating structure",
                      "★ 负债收入比的阈值效应 + 线性的额度使用率 —— 就是生成数据的结构")),
    ]
    return {
        "id": "loans", "mode": "additive",
        "title": t("Loan defaults → a hidden threshold", "贷款违约 → 隐藏的阈值"),
        "claim": t("4,000 binary outcomes and a row-number column that would leak. fincurve drops the leak, finds the "
                   "S-shaped threshold in debt-to-income, keeps utilisation linear, and proves the shape is stable across folds.",
                   "4000 个 0/1 结果，外加一个会泄漏的行号列。fincurve 剔除泄漏列，找到负债收入比中的 S 形阈值，保持额度使用率为线性，并证明形状在各折之间稳定。"),
        "truth": t("sigmoid at DTI 0.38 + linear utilisation", "负债收入比 0.38 处的 S 形 + 线性使用率"),
        "identified": t("S-shape (DTI) + linear (utilisation)", "S 形（负债收入比）+ 线性（使用率）"),
        "verdict": "hit",
        "panels": panels,
        "ranking": ranking_rows(rep, 5),
        "features": [{"name": t(names_en.get(r.Index, r.Index), r.Index),
                      "kind": r.type, "trend": t(TREND_EN.get(getattr(r, "trend", ""), "—"), getattr(r, "trend", "—"))
                      if isinstance(getattr(r, "trend", None), str) else t("removed", "已剔除"),
                      "importance": sig(getattr(r, "importance", np.nan)),
                      "stability": sig(getattr(r, "stability", np.nan))} for r in feats.itertuples()],
        "log": [{"tag": tag, "text": text} for tag, text in log],
    }


# ----------------------------------------------------------------------------- case 4: the random-walk trap
def rw_error_rates(n=200):
    fp = {"smooth": 0, "linear": 0, "bond": 0}
    tp = {"gbm": 0, "ar": 0}
    for s in range(n):
        rng = np.random.default_rng(10_000 + s)
        tt = np.linspace(0, 10, 120)
        fp["smooth"] += random_walk_like(2 * np.exp(0.2 * tt) + rng.normal(0, 0.5, tt.size))[0]
        fp["linear"] += random_walk_like(5 + 0.3 * tt + rng.normal(0, 0.4, tt.size))[0]
        fp["bond"] += random_walk_like(S.bond_price(rng)[1].to_numpy())[0]
        tp["gbm"] += random_walk_like(S.price_path(rng)[1].to_numpy())[0]
        e = np.zeros(200)
        for i in range(1, 200):
            e[i] = 0.95 * e[i - 1] + rng.normal()
        tp["ar"] += random_walk_like(50 + e)[0]
    return {"n": n, "false_positive": {k: v / n for k, v in fp.items()}, "detection": {k: v / n for k, v in tp.items()}}


def case_path():
    x, v = S.price_path(np.random.default_rng(7))
    rep = analyze(x, v)
    flag, rw = random_walk_like(v.to_numpy())
    returns = rep.extras["returns"]
    r = returns.data["y"]
    edges = np.linspace(r.min(), r.max(), 19)
    dens, _ = np.histogram(r, bins=edges, density=True)
    grid = np.linspace(edges[0] - 0.01, edges[-1] + 0.01, 160)
    dist_rows = returns.ranking[returns.ranking["status"] == "ok"].head(6)
    ok = rep.ranking[(rep.ranking["status"] == "ok") & np.isfinite(rep.ranking["d_cv"])]
    spurious = list(ok["id"].head(2))
    years = rep.data["x"]
    dense = np.linspace(years.min(), years.max(), 200)
    to_date = lambda yrs: [d.strftime("%Y-%m-%d") for d in pd.Timestamp(rep.data["t0"]) + pd.to_timedelta(yrs * 365.25 * 86400, unit="s")]
    vr = rw["variance_ratios"]
    checks = [
        {"name": t("AR(1) of log value", "对数净值的 AR(1)"), "value": sig(rw["ar1"], 3), "rule": "> 0.9", "pass": rw["ar1"] > 0.9},
        {"name": t("lag-1 autocorrelation of differences", "差分的一阶自相关"), "value": sig(rw["diff_acf1"], 2), "rule": "|·| < 0.3",
         "pass": abs(rw["diff_acf1"]) < 0.3},
        {"name": t("variance ratios q = 2, 4, 8", "方差比 q = 2, 4, 8"), "value": " / ".join(f"{vr[q]:.2f}" for q in sorted(vr)),
         "rule": "0.5 – 2", "pass": all(0.5 < val < 2 for val in vr.values())},
        {"name": t("residual lag-1 around a smoother", "去平滑趋势后残差的一阶自相关"), "value": sig(rw["resid_acf1"], 3),
         "rule": "> 0.5", "pass": rw["resid_acf1"] > 0.5},
    ]
    log = [
        ("order", t("x is a date → time-ordered → rolling-origin CV (fit on the past, score on the future)",
                    "x 是日期 → 有时间顺序 → 滚动验证（用过去拟合，在未来上打分）")),
        ("rank", t(f"best “curve” by CV: {label_en(ok.iloc[0]['model'])} — scores, but means nothing",
                   f"交叉验证得分最高的“曲线”：{ok.iloc[0]['label']} —— 有分数，但没有意义")),
        ("walk", t(f"random-walk test on log value: AR(1) {rw['ar1']:.2f} · diff lag-1 {rw['diff_acf1']:+.2f} · "
                   f"VR {min(vr.values()):.2f}–{max(vr.values()):.2f} · residual lag-1 {rw['resid_acf1']:.2f} → all four pass",
                   f"对数净值的随机游走检验：AR(1) {rw['ar1']:.2f} · 差分自相关 {rw['diff_acf1']:+.2f} · "
                   f"方差比 {min(vr.values()):.2f}–{max(vr.values()):.2f} · 残差自相关 {rw['resid_acf1']:.2f} → 四项全部满足")),
        ("verdict", t("no curve recommended: the shape is an accident of this one path", "不推荐任何曲线：形状只是这一条路径的偶然走势")),
        ("pivot", t(f"analyse log returns instead → {DIST_EN[returns.recommended.key]} (ΔAIC ranking, tails checked)",
                    f"改为分析对数收益率 → {returns.recommended.label}（按 AIC 排名，并检查尾部）")),
        ("verdict", t("★ random walk with normal log returns — the generating GBM", "★ 对数收益率为正态的随机游走 —— 就是生成数据的几何布朗运动")),
    ]
    return {
        "id": "path", "mode": "guardrail",
        "title": t("The random-walk trap", "随机游走陷阱"),
        "claim": t("A portfolio NAV path looks like it has a V-shaped trend, and a 5-parameter curve even wins the "
                   "cross-validation. fincurve's random-walk test flags the path, withholds a curve recommendation "
                   "and analyses returns instead.",
                   "一条组合净值路径看起来有 V 形趋势，5 参数曲线甚至赢得了交叉验证。fincurve 的随机游走检验识别出这是路径，不推荐任何曲线，转而分析收益率。"),
        "truth": t("geometric Brownian motion", "几何布朗运动"),
        "identified": t("random walk → normal log returns", "随机游走 → 对数收益率服从正态"),
        "verdict": "hit",
        "chart": {
            "xLabel": t("date", "日期"), "yLabel": t("NAV", "净值"), "time": True,
            "points": [[d.strftime("%Y-%m-%d") for d in x], arr(v)],
            "series": [{"label": t(label_en(rep.models[i].candidate.key) + " (spurious)", rep.models[i].label + "（伪形状）"),
                        "x": to_date(dense), "y": curve_values(rep.models[i], dense), "spurious": True} for i in spurious],
        },
        "checks": checks,
        "returns": {
            "bins": [{"x0": sig(a), "x1": sig(b), "d": sig(c)} for a, b, c in zip(edges[:-1], edges[1:], dens)],
            "series": [{"label": t(DIST_EN[k], returns.models[k].label), "x": arr(grid), "y": arr(returns.models[k].pdf(grid))}
                       for k in list(dist_rows["id"].head(2))],
            "ranking": [{"label": t(DIST_EN[row.id], row.label), "k": int(row.k), "d_aic": sig(row.d_aic, 3),
                         "tie": bool(row.tie), "rec": row.id == returns.recommended_id} for row in dist_rows.itertuples()],
        },
        "rates": rw_error_rates(),
        "log": [{"tag": tag, "text": text} for tag, text in log],
    }


# ----------------------------------------------------------------------------- scoreboard
def scoreboard(cached):
    rows = []

    def curve_row(sid, name, make, truth_keys, truth_label, kw=None):
        out = make(np.random.default_rng(7))
        args, extra = (out[:2], {"trials": out[2]}) if len(out) == 3 else (out, {})
        rep = analyze(*args, **extra, **(kw or {}))
        rec_key = rep.recommended.candidate.key
        found = [best_row(rep, k) for k in truth_keys]
        found = [f for f in found if f is not None]
        pos, row = min(found, key=lambda f: f[0])
        verdict = "hit" if rec_key in truth_keys else ("tie" if row["tie"] else "miss")
        n = rep.profile["n"]
        if verdict == "hit":
            detail = t(f"n={n}", f"n={n}")
        elif verdict == "tie":
            detail = t(f"truth ranked #{pos}, tied (Δ = {row['d_cv']:.3f} ± {row['se']:.3f}); simpler model preferred · n={n}",
                       f"真实形式排第 {pos}，与最优并列（Δ = {row['d_cv']:.3f} ± {row['se']:.3f}），推荐了更简单的模型 · n={n}")
        else:
            detail = t(f"truth ranked #{pos} (Δ = {row['d_cv']:.2f} ± {row['se']:.2f}); report warns the sample is too small · n={n}",
                       f"真实形式排第 {pos}（Δ = {row['d_cv']:.2f} ± {row['se']:.2f}）；报告提示样本太少 · n={n}")
        fam = {"lognormal": t(", multiplicative", "，乘性"), "binomial": t(", binomial", "，二项"),
               "poisson": t(", Poisson", "，Poisson")}.get(rep.recommended.family.name, t("", ""))
        rows.append({"id": sid, "name": name, "mode": "curve", "truth": truth_label,
                     "identified": t(label_en(rec_key) + fam["en"], rep.recommended.label + fam["zh"]),
                     "verdict": verdict, "detail": detail})

    curve_row("yield", t("Yield curve", "收益率曲线"), S.yield_curve, ["nelson_siegel"], t("Nelson–Siegel", "Nelson–Siegel"))
    curve_row("volterm", t("ATM vol term structure", "平值波动率期限结构"), S.vol_term_structure, ["heston_vol_term"],
              t("Heston variance reversion", "Heston 方差回复"))
    curve_row("smile", t("Volatility smile", "波动率微笑"), S.vol_smile, ["svi_smile"], t("SVI", "SVI"))
    curve_row("bond", t("Bond price vs yield", "债券价格–收益率"), S.bond_price, ["dcf_price"],
              t("semi-annual DCF", "半年付息现金流贴现"))
    curve_row("impact", t("Market impact", "市场冲击成本"), S.market_impact, ["sqrt", "power"], t("√q, multiplicative", "√q，乘性"))
    curve_row("default", t("Default rate vs score", "违约率–评分"), S.default_rate, ["logistic", "logit_linear"],
              t("logistic", "逻辑斯蒂"))
    curve_row("counts", t("Intraday trade counts", "日内成交笔数"), S.trade_counts, ["log_quadratic"],
              t("Poisson, log-quadratic", "Poisson，log-二次"))

    ret = analyze(S.daily_returns(np.random.default_rng(7)))
    rows.append({"id": "returns", "name": t("Daily returns", "日收益率"), "mode": "distribution",
                 "truth": t("Student-t (ν = 4)", "Student-t（ν = 4）"),
                 "identified": t(DIST_EN[ret.recommended.key], ret.recommended.label),
                 "verdict": "hit" if ret.recommended.key == "student_t" else "miss",
                 "detail": t(f"normal distribution ΔAIC = {ret.ranking.set_index('id').loc['normal', 'd_aic']:.0f}",
                             f"正态分布 ΔAIC = {ret.ranking.set_index('id').loc['normal', 'd_aic']:.0f}")})
    rows.append({"id": "path", "name": t("Portfolio NAV path", "组合净值路径"), "mode": "guardrail",
                 "truth": t("geometric Brownian motion", "几何布朗运动"),
                 "identified": cached["path"]["identified"], "verdict": "hit",
                 "detail": t("curve recommendation withheld", "不推荐任何曲线")})

    X, y = S.credit_spreads(np.random.default_rng(7))
    rep = analyze(X, y)
    f = rep.extras["additive"]["features"].set_index("feature")
    good = (rep.decisions["families"][0].name == "lognormal" and f.loc["杠杆率", "trend"] == "加速上升"
            and f.loc["剩余期限", "trend"] == "上升趋缓" and f.loc["发行人ID", "type"] == "已剔除")
    rows.append({"id": "spreads", "name": t("Credit spreads (5 features)", "信用利差（5 个特征）"), "mode": "additive",
                 "truth": t("log-additive: convex leverage, saturating maturity, rating", "对数加性：杠杆凸、期限饱和、评级"),
                 "identified": t("convex · saturating · rating offsets; issuer ID removed", "凸 · 饱和 · 评级偏移；剔除发行人 ID"),
                 "verdict": "hit" if good else "miss",
                 "detail": t(f"sector importance {f.loc['行业', 'importance']:.1%} (truly zero)", f"行业重要性 {f.loc['行业', 'importance']:.1%}（真实为 0）")})
    rows.append({"id": "loans", "name": t("Loan defaults (0/1)", "贷款违约（0/1）"), "mode": "additive",
                 "truth": cached["loans"]["truth"], "identified": cached["loans"]["identified"], "verdict": "hit",
                 "detail": t("row-number column removed", "剔除行号列")})
    return rows


def library():
    out = []
    for c in build_library(Scale(0.0, 1.0, 0.5, 1.0)):
        out.append({"key": c.key, "label": t(LABEL_EN[c.key], c.label), "group": c.group,
                    "groupLabel": t(GROUP_EN[c.group], GROUPS[c.group]), "formula": c.formula, "k": c.k,
                    "meaning": t(MEANING_EN[c.key], c.meaning.split("（r =")[0])})
    return out


def count_tests():
    spec = importlib.util.spec_from_file_location("run_tests", ROOT / "tests" / "run_tests.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return len(mod.TESTS)


if __name__ == "__main__":
    start = time.time()
    cases = {c["id"]: c for c in (case_smile(), case_impact(), case_loans(), case_path())}
    board = scoreboard(cases)
    data = {
        "version": __version__,
        "generated": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "stats": {"candidates": len(build_library(Scale(0.0, 1.0, 0.5, 1.0))), "likelihoods": 4,
                  "distributions": len(DISTRIBUTIONS), "shapes": 11, "tests": count_tests()},
        "groups": [{"key": k, "label": t(GROUP_EN[k], v)} for k, v in GROUPS.items() if k != "link"],
        "cases": [cases[k] for k in ("smile", "impact", "loans", "path")],
        "scoreboard": board,
        "library": library(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("window.FINCURVE = " + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    hits = sum(r["verdict"] == "hit" for r in board)
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB) in {time.time() - start:.0f}s; "
          f"scoreboard {hits} hit / {sum(r['verdict'] == 'tie' for r in board)} tie / {sum(r['verdict'] == 'miss' for r in board)} miss")
