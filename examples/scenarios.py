"""Synthetic financial datasets with known ground truth, shared by the demo and the tests."""
import numpy as np
import pandas as pd
from scipy import stats


def yield_curve(rng):
    T = np.array([1, 2, 3, 4, 6, 12, 24, 36, 60, 84, 120, 240, 360]) / 12
    tau = 1.8
    f1 = (1 - np.exp(-T / tau)) / (T / tau)
    y = 4.2 + 0.9 * f1 - 1.2 * (f1 - np.exp(-T / tau))
    return pd.Series(T, name="tenor (years)"), pd.Series(y + rng.normal(0, 0.03, T.size), name="yield (%)")


def vol_term_structure(rng):
    T = np.array([1 / 52, 2 / 52, 1 / 12, 2 / 12, 3 / 12, 6 / 12, 9 / 12, 1, 1.5, 2])
    v0, theta, kappa = 0.32**2, 0.18**2, 3.0
    L = (1 - np.exp(-kappa * T)) / (kappa * T)
    y = 100 * np.sqrt(theta + (v0 - theta) * L)
    return pd.Series(T, name="maturity (years)"), pd.Series(y + rng.normal(0, 0.3, T.size), name="ATM implied vol (%)")


def vol_smile(rng):
    k = np.linspace(-0.3, 0.3, 25)
    a, b, rho, m, s, T = 0.004, 0.08, -0.6, 0.02, 0.08, 0.25
    w = a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s**2))
    return pd.Series(k, name="log-moneyness ln(K/F)"), pd.Series(100 * np.sqrt(w / T) + rng.normal(0, 0.25, k.size), name="implied vol (%)")


def bond_price(rng):
    yl = np.linspace(1, 12, 30)
    t = np.arange(1, 21) / 2
    y = np.array([np.sum(2.5 / (1 + r / 200) ** (2 * t)) + 100 / (1 + r / 200) ** 20 for r in yl])
    return pd.Series(yl, name="yield to maturity (%)"), pd.Series(y + rng.normal(0, 0.15, yl.size), name="bond price")


def market_impact(rng):
    q = np.exp(rng.uniform(np.log(0.002), np.log(0.25), 90))
    return pd.Series(q, name="participation rate"), pd.Series(140 * np.sqrt(q) * np.exp(rng.normal(0, 0.3, q.size)), name="impact (bp)")


def default_rate(rng):
    score = np.arange(450, 826, 25).astype(float)
    pd_true = 0.35 / (1 + np.exp((score - 600) / 40))
    return pd.Series(score, name="credit score"), pd.Series(rng.binomial(600, pd_true) / 600, name="default rate"), np.full(score.size, 600)


def daily_returns(rng):
    r = stats.t.rvs(df=4, size=2520, random_state=rng) * 1.2 / np.sqrt(2)
    return pd.Series(r, name="daily return (%)")


def price_path(rng):
    dates = pd.date_range("2016-01-01", periods=121, freq="MS")
    dt = 1 / 12
    logret = (0.08 - 0.5 * 0.16**2) * dt + 0.16 * np.sqrt(dt) * rng.normal(size=120)
    value = 100 * np.exp(np.concatenate([[0], np.cumsum(logret)]))
    return pd.Series(dates, name="date"), pd.Series(value, name="portfolio NAV")


def credit_spreads(rng, n=1500):
    """Multi-feature: spread depends on leverage (convex), rating (categorical), maturity (saturating)."""
    rating = rng.choice(["AA", "A", "BBB", "BB"], size=n, p=[0.15, 0.35, 0.35, 0.15])
    leverage = rng.uniform(0.1, 0.8, n)
    maturity = rng.uniform(0.5, 30, n)
    sector = rng.choice(["financials", "industrials", "utilities"], size=n)
    issuer_id = np.array([f"ISS{i:05d}" for i in range(n)])
    base = {"AA": 40, "A": 70, "BBB": 130, "BB": 280}
    log_spread = (np.log([base[r] for r in rating]) + 1.8 * leverage ** 2
                  + 0.35 * (1 - np.exp(-maturity / 5)) + rng.normal(0, 0.18, n))
    X = pd.DataFrame({"leverage": leverage, "maturity": maturity, "rating": rating, "sector": sector, "issuer_id": issuer_id})
    return X, pd.Series(np.exp(log_spread), name="credit spread (bp)")


def loan_defaults(rng, n=4000):
    """Binary target with a threshold effect in debt-to-income and a linear effect in utilisation."""
    dti = rng.uniform(0, 0.6, n)
    util = rng.uniform(0, 1, n)
    eta = -3.2 + 2.5 / (1 + np.exp(-(dti - 0.38) / 0.03)) + 1.2 * util
    X = pd.DataFrame({"debt_to_income": dti, "utilization": util, "row_number": np.arange(n)})
    return X, pd.Series(rng.binomial(1, 1 / (1 + np.exp(-eta))), name="default")


def trade_counts(rng, n=400):
    """Poisson counts with intraday U-shape in minutes since the open."""
    minute = rng.uniform(0, 390, n)
    lam = np.exp(2.0 + 1.2 * ((minute - 195) / 195) ** 2)
    return pd.Series(minute, name="minutes since open"), pd.Series(rng.poisson(lam), name="trades")


# ----------------------------------------------------------------------------- ground truth (for the showcase site)
SMILE_TRUTH = {"a": 0.004, "b": 0.08, "rho": -0.6, "m": 0.02, "sigma": 0.08, "T": 0.25}


def vol_smile_truth(k):
    p = SMILE_TRUTH
    w = p["a"] + p["b"] * (p["rho"] * (k - p["m"]) + np.sqrt((k - p["m"]) ** 2 + p["sigma"] ** 2))
    return 100 * np.sqrt(w / p["T"])


def market_impact_truth(q):
    """Median impact: 140·√q bp (the noise is multiplicative with zero log-mean)."""
    return 140 * np.sqrt(q)


def loan_default_truth_effects(dti, util):
    """True contributions to logit(p), each centred on its sample mean."""
    f_dti = 2.5 / (1 + np.exp(-(dti - 0.38) / 0.03))
    f_util = 1.2 * util
    return f_dti, f_util
