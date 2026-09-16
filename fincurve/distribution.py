"""Sample-only mode: which distribution describes y, with emphasis on the tails."""
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .profile import acf1, ljung_box

DISTRIBUTIONS = [
    ("normal", "正态", stats.norm, "real", "基准；金融收益率的尾部通常比它厚"),
    ("student_t", "Student-t", stats.t, "real", "对称厚尾：日收益率最常用"),
    ("laplace", "Laplace", stats.laplace, "real", "尖峰、指数型尾部"),
    ("logistic", "Logistic", stats.logistic, "real", "比正态略厚的对称尾部"),
    ("skew_normal", "偏正态", stats.skewnorm, "real", "有偏但尾部不厚"),
    ("johnson_su", "Johnson SU", stats.johnsonsu, "real", "偏度、峰度都能调"),
    ("nig", "正态逆高斯 NIG", stats.norminvgauss, "real", "厚尾且有偏：Lévy 收益率模型"),
    ("lognormal", "对数正态", stats.lognorm, "positive", "价格、规模、乘性过程"),
    ("gamma", "Gamma", stats.gamma, "positive", "损失金额、等待时间"),
    ("weibull", "Weibull", stats.weibull_min, "positive", "违约时间、寿命"),
    ("inv_gauss", "逆高斯", stats.invgauss, "positive", "首次触及时间"),
    ("exponential", "指数", stats.expon, "positive", "无记忆的等待时间"),
]
TAIL_QUANTILES = (0.01, 0.05, 0.95, 0.99)


@dataclass
class FittedDistribution:
    key: str
    label: str
    meaning: str
    dist: object
    params: tuple
    loc0: float
    s0: float

    def _z(self, v):
        return (np.asarray(v, float) - self.loc0) / self.s0

    def logpdf(self, v):
        return self.dist.logpdf(self._z(v), *self.params) - np.log(self.s0)

    def pdf(self, v):
        return np.exp(self.logpdf(v))

    def cdf(self, v):
        return self.dist.cdf(self._z(v), *self.params)

    def ppf(self, q):
        return self.loc0 + self.s0 * self.dist.ppf(q, *self.params)


def _fit(key, label, dist, support, meaning, y):
    """Fit on a standardised copy of y so the optimiser works on O(1) numbers."""
    if support == "real":
        loc0, s0 = float(np.median(y)), float(np.std(y)) or 1.0
        params = dist.fit((y - loc0) / s0)
        k = len(params)
    else:
        loc0, s0 = 0.0, float(np.median(y)) or 1.0
        params = dist.fit(y / s0, floc=0)
        k = len(params) - 1
    return FittedDistribution(key, label, meaning, dist, tuple(params), loc0, s0), k


def run_distribution(y, include=None, exclude=None):
    y = np.asarray(y, float)
    n = y.size
    positive = bool(np.all(y > 0))
    med, rows, models = np.median(y), [], {}
    emp_q = np.quantile(y, TAIL_QUANTILES)
    for key, label, dist, support, meaning in DISTRIBUTIONS:
        if (include and key not in include) or (exclude and key in exclude):
            continue
        if support == "positive" and not positive:
            continue
        row = {"id": key, "model": key, "label": label, "meaning": meaning, "status": "ok"}
        rows.append(row)
        try:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                fd, k = _fit(key, label, dist, support, meaning, y)
                ll = fd.logpdf(y)
                if not np.all(np.isfinite(ll)):
                    raise FloatingPointError
                ks = stats.kstest(y, fd.cdf).statistic
                mod_q = fd.ppf(np.array(TAIL_QUANTILES))
        except Exception:
            row["status"] = "拟合失败"
            continue
        nll = -float(np.sum(ll))
        row.update(k=k, nll=nll, aic=2 * k + 2 * nll, bic=k * np.log(n) + 2 * nll, ks=float(ks))
        for q, e, m in zip(TAIL_QUANTILES, emp_q, mod_q):
            row[f"q{round(q * 100):02d}_err"] = float(abs(m - med) / (abs(e - med) or 1.0) - 1)
        models[key] = fd

    table = pd.DataFrame(rows)
    table = table.sort_values("aic", na_position="last", kind="stable").reset_index(drop=True)
    table["d_aic"] = table["aic"] - table["aic"].min()
    table["tie"] = table["d_aic"] < 2
    ties = table[table["tie"]].sort_values(["k", "aic"], kind="stable")
    best = table["id"].iloc[0] if models else None
    recommended = ties["id"].iloc[0] if not ties.empty else best
    return {"table": table, "models": models, "best": best, "recommended": recommended}


def sample_checks(y, ordered):
    y = np.asarray(y, float)
    out = {"n": int(y.size), "mean": float(y.mean()), "std": float(y.std()),
           "skew": float(stats.skew(y)), "excess_kurtosis": float(stats.kurtosis(y)),
           "positive": bool(np.all(y > 0)), "ordered": bool(ordered)}
    if ordered and y.size >= 30:
        out["acf1"] = acf1(y)
        out["lb_p"] = ljung_box(y)[1]
        out["lb_sq_p"] = ljung_box((y - y.mean()) ** 2)[1]
        out["looks_like_levels"] = bool(out["acf1"] > 0.9)
    return out
