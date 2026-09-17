"""Regression tests for fincurve. Run with:  python tests/run_tests.py [name filter ...]

Each test generates data with a known ground truth and checks that the characteriser
recovers it (or, where the data cannot identify the truth, that it says so). Numerical
tests check that the compiled kernels and the variable-projection fits agree with the
NumPy and general least-squares reference implementations.

Environment: FINCURVE_REQUIRE_C=1 fails if the C extension is missing;
FINCURVE_NO_C=1 forces the NumPy fallback.
"""
import os
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import scenarios as S  # noqa: E402
from fincurve import analyze, kernels, list_candidates  # noqa: E402
from fincurve.families import get_family  # noqa: E402
from fincurve.fitting import fit_params, fit_separable, jitter  # noqa: E402
from fincurve.library import Scale, build_library  # noqa: E402
from fincurve.profile import random_walk_like  # noqa: E402

TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def tied_models(report):
    t = report.ranking
    return set(t.loc[t["tie"], "model"])


# ----------------------------------------------------------------------------- numerical kernels
@test
def c_extension_is_available_when_required():
    if os.environ.get("FINCURVE_REQUIRE_C"):
        assert kernels.HAVE_C, "the C extension was required but did not import"


@test
def local_linear_kernel_matches_numpy_reference():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        n = int(rng.integers(6, 300))
        x = np.round(rng.uniform(0, 10, n), int(rng.integers(0, 3)))    # includes tied x values
        y = np.sin(x) + rng.normal(0, 0.3, n)
        frac = float(rng.choice([0.1, 0.2, 0.3, 0.5, 0.8]))
        x_eval = rng.uniform(-1, 11, int(rng.integers(1, 40)))
        for kw in ({"loo": False}, {"loo": True}, {"x_eval": x_eval}):
            fast = kernels.local_linear(x, y, frac=frac, **kw)
            ref = kernels.local_linear(x, y, frac=frac, force_numpy=True, **kw)
            assert np.array_equal(np.isnan(fast), np.isnan(ref))
            ok = ~np.isnan(ref)
            worst = max(worst, float(np.max(np.abs(fast[ok] - ref[ok]), initial=0.0)))
    assert worst < 1e-9, worst


# ----------------------------------------------------------------------------- library and fitting engines
@test
def every_candidate_refits_its_own_curve():
    x = np.linspace(0.5, 30, 80)
    fam = get_family("gaussian")
    rng = np.random.default_rng(0)
    for c in build_library(Scale.from_x(x)):
        base = 3 + np.log(x) + 0.3 * np.sin(x / 3)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            p_true, _ = fit_params(c.func, x, base, np.ones_like(x), fam, c.init(x, base), c.bounds)
            y_true = c.func(x, *p_true)
            y = y_true + rng.normal(0, 0.01 * np.ptp(y_true) + 1e-9, x.size)
            p, _ = fit_params(c.func, x, y, np.ones_like(x), fam, c.init(x, y), c.bounds)
            r2 = 1 - np.sum((y - c.func(x, *p)) ** 2) / np.sum((y - y.mean()) ** 2)
        assert c.key == "constant" or r2 > 0.99, f"{c.key}: R² = {r2:.3f}"


@test
def separable_basis_reproduces_the_formula():
    x = np.linspace(0.5, 30, 60)
    rng = np.random.default_rng(1)
    for c in build_library(Scale.from_x(x)):
        if not c.separable:
            continue
        lb, ub = (np.asarray(b, float) for b in c.nl_bounds)
        for _ in range(10):
            lo = np.where(np.isfinite(lb), lb, -3.0)
            hi = np.where(np.isfinite(ub), np.minimum(ub, lo + 5.0), lo + 5.0)
            theta = tuple(rng.uniform(lo, hi))
            cols = [np.broadcast_to(col, x.shape) for col in c.basis(x, theta)]
            beta = np.abs(rng.normal(size=len(cols)))
            with np.errstate(all="ignore"):
                via_basis = np.column_stack(cols) @ beta
                via_formula = c.func(x, *c.assemble(beta, theta))
            ok = np.isfinite(via_basis) & np.isfinite(via_formula)
            assert np.allclose(via_basis[ok], via_formula[ok], rtol=1e-9, atol=1e-9), c.key


@test
def variable_projection_matches_general_least_squares():
    x = np.linspace(0.5, 30, 80)
    fam = get_family("gaussian")
    w = np.ones_like(x)
    rng = np.random.default_rng(2)
    for c in build_library(Scale.from_x(x)):
        if not c.separable:
            continue
        base = 3 + np.log(x) + 0.3 * np.sin(x / 3)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            p_true, _ = fit_params(c.func, x, base, w, fam, c.init(x, base), c.bounds)
            y = c.func(x, *p_true) + rng.normal(0, 0.05, x.size)
            _, general = fit_params(c.func, x, y, w, fam, jitter(c.init(x, y), 2, rng), c.bounds)
            _, _, projected = fit_separable(c, x, y, w, fam)
        assert projected <= general * 1.01 + 1e-9, (c.key, projected, general)


@test
def library_table_lists_all_candidates():
    t = list_candidates()
    assert len(t) >= 30 and t["key"].is_unique


# ----------------------------------------------------------------------------- curve mode
@test
def recovers_svi_smile():
    x, y = S.vol_smile(np.random.default_rng(7))
    assert analyze(x, y).recommended.candidate.key == "svi_smile"


@test
def recovers_logistic_default_curve_with_trials():
    x, y, trials = S.default_rate(np.random.default_rng(7))
    rep = analyze(x, y, trials=trials)
    assert rep.decisions["families"][0].name == "binomial"
    assert rep.recommended.candidate.key in ("logistic", "logit_linear"), rep.recommended.candidate.key


@test
def market_impact_prefers_multiplicative_square_root():
    for seed in range(3):
        x, y = S.market_impact(np.random.default_rng(seed))
        rep = analyze(x, y)
        assert "lognormal" in {f.name for f in rep.decisions["families"]}
        assert {"sqrt", "power"} & tied_models(rep), seed


@test
def counts_use_poisson_and_find_log_quadratic():
    x, y = S.trade_counts(np.random.default_rng(7))
    rep = analyze(x, y)
    assert rep.decisions["families"][0].name == "poisson"
    assert "log_quadratic" in tied_models(rep)


@test
def price_path_is_flagged_as_random_walk_and_uses_time_cv():
    x, y = S.price_path(np.random.default_rng(7))
    rep = analyze(x, y)
    assert rep.decisions["cv"] == "time"
    assert "returns" in rep.extras and rep.extras["returns"].recommended is not None
    assert any("random walk" in w for w in rep.warnings)


@test
def random_walk_test_error_rates():
    false_pos = detected = 0
    for s in range(40):
        rng = np.random.default_rng(100 + s)
        t = np.linspace(0, 10, 120)
        false_pos += random_walk_like(2 * np.exp(0.2 * t) + rng.normal(0, 0.5, t.size))[0]
        false_pos += random_walk_like(S.bond_price(rng)[1].to_numpy())[0]
        detected += random_walk_like(S.price_path(rng)[1].to_numpy())[0]
    assert false_pos == 0, false_pos
    assert detected >= 34, detected


@test
def term_names_are_not_mistaken_for_time():
    for make in (S.yield_curve, S.vol_term_structure):
        x, y = make(np.random.default_rng(7))
        assert analyze(x, y).decisions["ordered"] is False, make.__name__


@test
def small_samples_do_not_recommend_a_poor_fit():
    for make in (S.yield_curve, S.vol_term_structure):
        x, y = make(np.random.default_rng(7))
        rep = analyze(x, y)
        r2 = rep.ranking.set_index("id").loc[rep.recommended_id, "r2"]
        assert r2 > 0.98, (make.__name__, rep.recommended.candidate.key, r2)
        assert any("the ranking is unstable" in w for w in rep.warnings)


@test
def term_structure_context_note_and_axis():
    x, y = S.yield_curve(np.random.default_rng(7))
    rep = analyze(x, y)
    assert rep.data["term_axis"]
    assert any("term structure" in n for n in rep.notes)


@test
def datetime_x_round_trips_through_predict():
    x, y = S.price_path(np.random.default_rng(7))
    rep = analyze(x, y)
    pred = rep.predict(x.iloc[:5])
    assert np.allclose(pred, rep.recommended.predict(rep.data["x"][:5]))


# ----------------------------------------------------------------------------- distribution mode
@test
def fat_tailed_returns_pick_a_heavy_tailed_distribution():
    rep = analyze(S.daily_returns(np.random.default_rng(7)))
    assert rep.recommended.key in ("student_t", "johnson_su", "nig")
    normal = rep.ranking.set_index("id").loc["normal"]
    assert normal["d_aic"] > 50 and normal["q01_err"] < 0


@test
def volatility_clustering_is_detected():
    rng = np.random.default_rng(3)
    n = 3000
    sig2, r = np.empty(n), np.empty(n)
    sig2[0] = 1.0
    for t in range(n):
        if t:
            sig2[t] = 0.05 + 0.1 * r[t - 1] ** 2 + 0.85 * sig2[t - 1]
        r[t] = np.sqrt(sig2[t]) * rng.standard_normal()
    rep = analyze(pd.Series(r, name="GARCH returns"), time_ordered=True)
    assert any("volatility clusters" in w for w in rep.warnings)


# ----------------------------------------------------------------------------- additive mode
@test
def credit_spreads_shapes_and_leakage_guard():
    X, y = S.credit_spreads(np.random.default_rng(7))
    rep = analyze(X, y)
    feats = rep.extras["additive"]["features"].set_index("feature")
    assert rep.decisions["families"][0].name == "lognormal"
    assert feats.loc["issuer_id", "type"] == "excluded"
    assert feats.loc["leverage", "trend"] == "accelerating rise"
    assert feats.loc["maturity", "trend"] == "rising, flattening"
    assert feats.loc["sector", "importance"] < 0.01
    assert rep.recommended_id == "shapes"
    pred = rep.predict(X.head(10))
    assert np.all(np.isfinite(pred)) and np.all(pred > 0)


@test
def loan_defaults_find_threshold_and_drop_row_number():
    X, y = S.loan_defaults(np.random.default_rng(7))
    rep = analyze(X, y)
    feats = rep.extras["additive"]["features"].set_index("feature")
    assert feats.loc["row_number", "type"] == "excluded"
    assert feats.loc["debt_to_income", "trend"] == "S-shaped rise"
    assert feats.loc["utilization", "trend"] == "roughly linear rise"


@test
def interaction_is_detected_when_present():
    rng = np.random.default_rng(11)
    n = 1500
    X = pd.DataFrame({"momentum": rng.normal(size=n), "volatility": rng.normal(size=n), "noise": rng.normal(size=n)})
    y = 0.5 * X["momentum"] + 0.3 * X["volatility"] + 0.8 * X["momentum"] * X["volatility"] + rng.normal(0, 0.5, n)
    rep = analyze(X, pd.Series(y, name="return"))
    pairs = {tuple(sorted(i["pair"])) for i in rep.extras["additive"]["interactions"]}
    assert ("momentum", "volatility") in pairs, pairs
    assert rep.recommended_id == "interactions"


if __name__ == "__main__":
    only = sys.argv[1:]
    failed = 0
    start = time.time()
    print(f"C extension: {'loaded' if kernels.HAVE_C else 'not available (NumPy fallback)'}")
    for fn in TESTS:
        if only and not any(o in fn.__name__ for o in only):
            continue
        t = time.time()
        try:
            fn()
            print(f"PASS  {fn.__name__}  ({time.time() - t:.1f}s)")
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}  ({time.time() - t:.1f}s)")
            traceback.print_exc()
    print(f"\n{'all tests passed' if not failed else f'{failed} failed'} ({time.time() - start:.0f}s)")
    sys.exit(1 if failed else 0)
