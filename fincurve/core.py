"""Entry point: profile the data, choose the error model and validation scheme, run the right mode."""
import numpy as np
import pandas as pd

from .additive import SHAPE_LABELS, fit_linear, fit_mean, prepare_features, run_additive
from .distribution import run_distribution, sample_checks
from .families import get_family
from .profile import describe_relation, describe_target, random_walk_like
from .report import FAMILY_SHORT, Report
from .scoring import make_splits
from .univariate import extrapolation_spread, residual_checks, run_race

YEAR_SECONDS = 365.25 * 86400


def analyze(X, y=None, *, family="auto", cv="auto", k_folds=5, groups=None, weights=None, trials=None,
            time_ordered=None, include=None, exclude=None, n_starts=2, nested=True, interactions=True,
            random_state=0, n_jobs=1):
    """Characterise how y depends on X, or which distribution X follows when y is omitted.

    X            1-D array/Series (one x), DataFrame (one or more features), or the sample itself if y is None
    family       "auto" | "gaussian" | "lognormal" | "binomial" | "poisson"
    cv           "auto" | "kfold" | "time" | "loo" | "group"
    groups       group labels for grouped cross-validation (e.g. issuer, date)
    weights      observation weights; trials = number of trials behind each observed proportion
    time_ordered force (True) or forbid (False) time-series treatment; None lets the profile decide
    include / exclude   candidate keys or group names, e.g. include=["term_structure", "polynomial"]
    nested       repeat the multi-feature shape search inside every fold (honest but slower)
    n_jobs       worker processes: 1 runs in-process, -1 uses every core. On macOS and Windows call it
                 from code guarded by `if __name__ == "__main__":`
    """
    if y is None:
        return analyze_distribution(X, ordered=bool(time_ordered), include=include, exclude=exclude)

    y_name = str(getattr(y, "name", None) or "y")
    y = np.asarray(y, float).ravel()
    if isinstance(X, pd.DataFrame):
        df = X.reset_index(drop=True)
    elif isinstance(X, pd.Series):
        df = X.reset_index(drop=True).to_frame(name=X.name if X.name is not None else "x")
    else:
        arr = np.asarray(X)
        df = pd.DataFrame({"x": arr}) if arr.ndim == 1 else pd.DataFrame(arr, columns=[f"x{i + 1}" for i in range(arr.shape[1])])
    if len(df) != y.size:
        raise ValueError(f"X has {len(df)} rows but y has {y.size} values")

    extra = {k: None if v is None else np.asarray(v).ravel() for k, v in
             {"weights": weights, "trials": trials, "groups": groups}.items()}
    keep = ~df.isna().any(axis=1).to_numpy() & np.isfinite(y)
    for k in ("weights", "trials"):
        if extra[k] is not None:
            keep &= np.isfinite(extra[k].astype(float))
    dropped = int((~keep).sum())
    df, y = df[keep].reset_index(drop=True), y[keep]
    extra = {k: None if v is None else v[keep] for k, v in extra.items()}

    w = np.ones(y.size) if extra["weights"] is None else extra["weights"].astype(float)
    if extra["trials"] is not None:
        t = extra["trials"].astype(float)
        if np.any(y > 1):
            y = y / t
        w = w * t
    opts = dict(family=family, cv=cv, k_folds=k_folds, groups=extra["groups"], trials=extra["trials"],
                time_ordered=time_ordered, include=include, exclude=exclude, n_starts=n_starts,
                random_state=random_state, dropped=dropped, y_name=y_name, n_jobs=n_jobs)

    single = df.shape[1] == 1 and (pd.api.types.is_numeric_dtype(df.iloc[:, 0])
                                   or pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]))
    if single:
        return _analyze_curve(df.iloc[:, 0], y, w, **opts)
    return _analyze_additive(df, y, w, nested=nested, interactions=interactions, **opts)


# ----------------------------------------------------------------------------- decisions
def _choose_families(family, target, y, mu_hint):
    if family != "auto":
        return [get_family(family)], "set by the caller"
    kind = target["kind"]
    gaussian, lognormal = get_family("gaussian"), get_family("lognormal")
    if kind == "binary":
        return [get_family("binomial")], "y takes only the values 0 and 1: Bernoulli likelihood"
    if kind == "proportion":
        return [get_family("binomial")], "y lies between 0 and 1: treated as a proportion (binomial likelihood)"
    if kind == "count":
        mu = np.clip(mu_hint, 1e-6, None)
        disp = float(np.mean((y - mu) ** 2 / mu))
        if 0.3 < disp < 3:
            return [get_family("poisson")], f"non-negative integer counts with dispersion {disp:.2f} (close to 1): Poisson likelihood"
        fams = [gaussian] + ([lognormal] if y.min() > 0 else [])
        return fams, f"integer counts, but dispersion is {disp:.1f} (Poisson needs about 1): continuous error models instead"
    return None, None


def _decide_order(time_ordered, is_datetime, rel, name, y_sorted):
    if time_ordered is not None:
        return bool(time_ordered), "set by the caller"
    if is_datetime:
        return True, "x is a datetime"
    if rel.get("time_like_name") and not rel.get("term_like_name"):
        return True, f"the name of x ({name!r}) looks like time"
    if rel.get("equally_spaced") and rel.get("n", 0) >= 30:
        flag, rw = random_walk_like(y_sorted)
        if flag:
            return True, (f"x is equally spaced and y moves like a random walk along it "
                          f"(AR(1) {rw['ar1']:.2f}, near-independent differences)")
    return False, "x does not look like time"


def _choose_cv(cv, n, ordered, groups):
    if cv != "auto":
        return cv, "set by the caller"
    if groups is not None:
        return "group", "group labels were provided"
    if ordered:
        return "time", "the data are ordered in time: shuffling would use the future to predict the past"
    if n < 20:
        return "loo", f"only {n} observations: leave-one-out"
    return "kfold", "no time ordering"


def _group_of(models, key):
    for m in models.values():
        if m.candidate.key == key:
            return m.candidate.group
    return None


# ----------------------------------------------------------------------------- curve mode
def _analyze_curve(xs, y, w, *, family, cv, k_folds, groups, trials, time_ordered, include, exclude,
                   n_starts, random_state, dropped, y_name, n_jobs):
    x_name = str(xs.name) if xs.name is not None else "x"
    is_datetime = pd.api.types.is_datetime64_any_dtype(xs)
    t0 = xs.min() if is_datetime else None
    x = ((xs - t0).dt.total_seconds().to_numpy() / YEAR_SECONDS) if is_datetime else xs.to_numpy(float)
    n = y.size
    warnings_, notes = [], []
    if dropped:
        notes.append(f"dropped {dropped} rows with missing values")

    target = describe_target(y, trials)
    rel = describe_relation(x, y, x_name)
    order = np.argsort(x, kind="stable")
    mu_hint = np.empty(n)
    mu_hint[order] = rel["_smooth"][1]
    families, fam_reason = _choose_families(family, target, y, mu_hint)
    if families is None:
        if target["kind"] == "positive":
            hetero = rel["hetero_p"] < 0.05 and rel["hetero_rho"] > 0.15
            wide = target["max_min_ratio"] > 3
            if hetero or wide:
                why = (f"noise grows with the level (Spearman ρ = {rel['hetero_rho']:.2f})" if hetero
                       else f"the max/min ratio is {target['max_min_ratio']:.1f}")
                families = [get_family("gaussian"), get_family("lognormal")]
                fam_reason = f"y is positive and {why}: additive and multiplicative errors compete"
            else:
                families = [get_family("gaussian")]
                fam_reason = "y is positive with a narrow range and constant noise: additive errors"
        else:
            families, fam_reason = [get_family("gaussian")], "y can take either sign: additive (Gaussian) errors"

    ordered, order_reason = _decide_order(time_ordered, is_datetime, rel, x_name, y[order])
    strategy, cv_reason = _choose_cv(cv, n, ordered, groups)
    splits = make_splits(n, strategy, k_folds, order=order, groups=groups, seed=random_state)
    race = run_race(x, y, w, families, splits, n_starts, include, exclude, random_state, frac=rel["frac"], n_jobs=n_jobs)
    table, models = race["table"], race["models"]
    if race["recommended"] is None:
        raise RuntimeError("every candidate failed to fit")
    rec, best = models[race["recommended"]], models[race["best"]]
    ranked = table[table["status"] == "ok"]

    # --- time-series paths
    extras = {}
    if ordered:
        flag, rw = random_walk_like(y[order])
        if flag:
            base = np.log(y[order]) if rw["log_scale"] else y[order]
            name = "log return" if rw["log_scale"] else "first difference"
            extras["returns"] = analyze_distribution(pd.Series(np.diff(base), name=name), ordered=True)
            warnings_.append(
                f"y looks like a price or NAV path, i.e. a random walk (AR(1) {rw['ar1']:.2f}, differences nearly "
                f"uncorrelated {rw['diff_acf1']:+.2f}): any curve shape is an accident of this one path and must not be "
                f"extrapolated. Study the distribution of the {name}s instead — attached as report.extras['returns']")
        if order_reason != "set by the caller":
            notes.append(f"treated as a time series because {order_reason}")
        notes.append("rolling-origin validation measures forward prediction, which is stricter than shuffled K-fold")

    # --- ties, extrapolation, context, reference
    ties = ranked[ranked["tie"]]
    multi = len(families) > 1
    names = [f"{row.label} [{FAMILY_SHORT[row.family]}]" if multi else row.label for row in ties.itertuples()]
    if len(ties) > 1:
        notes.append(f"{len(ties)} models are practically tied with the best: " + ", ".join(names[:8]))
        if ties["group"].nunique() > 1:
            warnings_.append("the tied models belong to different families (" + ", ".join(ties["group"].unique()[:6]) +
                             "): the data cannot tell these shapes apart — choose on economic grounds or collect "
                             "a wider x range")
        spread = extrapolation_spread([models[i] for i in ties["id"].head(6)], x, y)
        if spread > 0.5:
            size = f"{spread:.0f}×" if spread >= 10 else f"{spread:.0%}"
            warnings_.append(f"the tied models disagree strongly outside the data (within 25% beyond the observed "
                             f"range the largest gap is {size} of y's range): do not extrapolate")
    context = rel.get("context")
    if context:
        label, context_groups = context
        themed = ranked[ranked["model"].map(lambda k: _group_of(models, k) in context_groups)]
        if not themed.empty and rec.candidate.group not in context_groups:
            row = themed.iloc[0]
            pos = int(ranked.index.get_loc(themed.index[0])) + 1
            status = ("practically tied with the best" if row["tie"]
                      else f"{row['d_cv']:.3g} ± {row['se']:.2g} behind the best")
            notes.append(f"the name of x suggests a {label}: the specialised form '{row['label']}' ranks #{pos} "
                         f"({status}). Its parameters have a direct economic meaning, so prefer it when the gap is small")
        elif rec.candidate.group in context_groups:
            notes.append(f"the name of x suggests a {label}, and the recommended model is that context's specialised form")
    for rid in race["reference_ids"]:
        ref = table.set_index("id").loc[rid]
        if np.isfinite(ref["d_cv"]) and ref["d_cv"] < 0 and abs(ref["d_cv"]) > 2 * (ref["se"] or 0):
            warnings_.append(f"the non-parametric smoother beats every candidate (ΔCV-NLL = {ref['d_cv']:.3g}): the "
                             "library may be missing the true shape — inspect the residual plot")
    shape = rel["shape"]
    for label, m in (("the recommended model", rec), ("the best-scoring model", best)):
        c = m.candidate
        if "peak" in c.tags and c.group not in ("polynomial", "term_structure") and shape in ("increasing", "decreasing"):
            warnings_.append(f"{label} ('{c.label}') is a peak shape, but the data are monotone over the observed range: "
                             "it is probably using half a peak to mimic a sigmoid or saturation, so the peak location "
                             "is unsupported")
            break

    # --- residuals and data quality
    checks = residual_checks(rec, x, y)
    if not ordered and n >= 20 and checks["resid_lag1"] > 0.5:
        warnings_.append(f"the recommended model's residuals are strongly autocorrelated along x (lag-1 = "
                         f"{checks['resid_lag1']:.2f}): either a shape is missing or the data are a time series "
                         "(then pass time_ordered=True, because shuffled K-fold is optimistic)")
    elif not ordered and n >= 20 and checks["resid_lag1"] > 0.3:
        warnings_.append(f"the recommended model's residuals still show structure along x (lag-1 = "
                         f"{checks['resid_lag1']:.2f}): a shape may be missing")
    if (rec.family.name == "gaussian" and checks.get("hetero_p", 1) < 0.01 and checks.get("hetero_rho", 0) > 0.2
            and len(families) == 1 and y.min() > 0):
        notes.append("residuals grow with the fitted value: try family='lognormal' (multiplicative errors)")
    if rel["n_outliers"] or rel["resid_excess_kurtosis"] > 3:
        warnings_.append(f"heavy-tailed residuals or outliers ({rel['n_outliers']} beyond 5 MADs): least squares can be "
                         "pulled by a few points — check them")
    if rel["max_gap_share"] > 0.3:
        notes.append(f"x has a large gap (the widest spans {rel['max_gap_share']:.0%} of its range): the shape inside "
                     "the gap is unsupported")
    if n < 15:
        warnings_.append(f"only {n} observations: the ranking is unstable, treat it as indicative")
    if target["kind"] == "proportion" and trials is None:
        notes.append("no trial counts were given for the proportions, so all points get equal weight; pass trials= "
                     "for a proper likelihood")
    if race["skipped"]:
        notes.append("candidates skipped because of their domain or the sample size: " + ", ".join(race["skipped"]))

    term_axis = bool(x.min() >= 0 and (rel["term_like_name"] or "term_axis" in rec.candidate.tags))
    profile = {"n": n, "target": target, "relation": {k: v for k, v in rel.items() if not k.startswith("_")},
               "residual_checks": checks}
    decisions = {"families": families, "family_reason": fam_reason, "cv": strategy, "cv_reason": cv_reason,
                 "n_splits": len(splits), "ordered": ordered, "order_reason": order_reason,
                 "n_candidates": int((table["status"] != "reference").sum()), "skipped": race["skipped"]}
    data = {"x": x, "y": y, "w": w, "x_name": x_name, "y_name": y_name, "t0": t0, "term_axis": term_axis}
    return Report("curve", data, profile, decisions, table, models, race["best"], race["recommended"],
                  warnings_, notes, extras)


# ----------------------------------------------------------------------------- additive mode
def _analyze_additive(df, y, w, *, family, cv, k_folds, groups, trials, time_ordered, include, exclude,
                      n_starts, random_state, dropped, y_name, nested, interactions, n_jobs):
    n = y.size
    warnings_, notes = [], []
    if dropped:
        notes.append(f"dropped {dropped} rows with missing values")
    feats = prepare_features(df)
    usable = [f for f in feats if f.kind != "excluded"]
    if not usable:
        raise ValueError("no usable features: " + "; ".join(f"{f.name} ({f.note})" for f in feats))
    target = describe_target(y, trials)
    idx = np.arange(n)

    families, fam_reason = None, None
    if family == "auto" and target["kind"] == "count":
        fit, _ = fit_linear(usable, get_family("poisson"), idx, y, w)
        families, fam_reason = _choose_families(family, target, y, get_family("poisson").clip(fit_mean(fit, idx)))
    elif family != "auto" or target["kind"] in ("binary", "proportion"):
        families, fam_reason = _choose_families(family, target, y, None)
    if families is None or len(families) > 1:
        if y.min() > 0:
            aics = {name: fit_linear(usable, get_family(name), idx, y, w)[1] for name in ("gaussian", "lognormal")}
            pick = min(aics, key=aics.get)
            families = [get_family(pick)]
            gap = abs(aics["gaussian"] - aics["lognormal"])
            fam_reason = ((fam_reason + "; ") if fam_reason else "") + \
                f"y is positive: with all-linear terms, {families[0].label} has an AIC {gap:.1f} lower, so it is used"
        else:
            families, fam_reason = [get_family("gaussian")], "y can take either sign: additive (Gaussian) errors"
    fam = families[0]

    time_feats = [f for f in usable if f.is_time]
    ordered = bool(time_ordered) if time_ordered is not None else bool(time_feats)
    if time_ordered is not None:
        order_reason = "set by the caller"
    else:
        order_reason = f"feature {time_feats[0].name!r} is a datetime" if time_feats else ""
    strategy, cv_reason = _choose_cv(cv, n, ordered, groups)
    order = np.argsort(time_feats[0].raw, kind="stable") if time_feats else None
    splits = make_splits(n, strategy, k_folds, order=order, groups=groups, seed=random_state)

    res = run_additive(feats, fam, y, w, splits, nested=nested, interactions=interactions, n_jobs=n_jobs)
    table = res["table"]

    for f in feats:
        if f.kind == "excluded":
            warnings_.append(f"feature {f.name!r} was removed: {f.note}")
    numeric = [f for f in usable if f.kind == "numeric"]
    if len(numeric) >= 2:
        C = np.corrcoef(np.column_stack([f.z for f in numeric]), rowvar=False)
        for i in range(len(numeric)):
            for j in range(i + 1, len(numeric)):
                if abs(C[i, j]) > 0.9:
                    warnings_.append(f"{numeric[i].name!r} and {numeric[j].name!r} are highly correlated "
                                     f"(r = {C[i, j]:.2f}): how shape and importance are split between them is unstable")
    feat_table = res["features"]
    if "stability" in feat_table:
        for row in feat_table.itertuples():
            stab = getattr(row, "stability", np.nan)
            full_shape = res["full_shapes"].get(row.feature)
            if full_shape and full_shape != "linear" and np.isfinite(stab) and stab < 0.9:
                warnings_.append(f"the nonlinear shape of {row.feature!r} is not consistent across folds (mean "
                                 f"effect-curve correlation {stab:.2f}): its details may be noise")
    ref = table.set_index("id")
    if "knn" in ref.index and np.isfinite(ref.loc["knn", "d_cv"]) and ref.loc["knn", "d_cv"] < 0 \
            and abs(ref.loc["knn", "d_cv"]) > 2 * (ref.loc["knn", "se"] or 0):
        warnings_.append("the k-nearest-neighbour reference is clearly better: there may be interactions or local "
                         "structure the additive model cannot express")
    rec_row = ref.loc[res["recommended"]]
    if rec_row["k"] > n / 10:
        warnings_.append(f"the recommended model has {int(rec_row['k'])} parameters for {n} observations: "
                         "risk of overfitting")
    for h in res["history"]:
        notes.append(f"{h['feature']!r}: {SHAPE_LABELS[h['from']]} → {SHAPE_LABELS[h['to']]} "
                     f"(full-sample AIC −{h['aic_gain']:.1f})")
    for it in res["interactions"]:
        notes.append(f"added interaction {it['pair'][0]} × {it['pair'][1]} (AIC −{it['aic_gain']:.1f})")
    if ordered and order_reason:
        notes.append(f"treated as a time series because {order_reason}")

    kinds = {"numeric": sum(f.kind == "numeric" for f in feats), "binary": sum(f.kind == "binary" for f in feats),
             "categorical": sum(f.kind == "categorical" for f in feats),
             "removed": sum(f.kind == "excluded" for f in feats)}
    profile = {"n": n, "target": target, "feature_kinds": kinds}
    decisions = {"families": families, "family_reason": fam_reason, "cv": strategy, "cv_reason": cv_reason,
                 "n_splits": len(splits), "ordered": ordered}
    data = {"X": df, "y": y, "w": w, "columns": list(df.columns), "y_name": y_name}
    return Report("additive", data, profile, decisions, table, res["models"], res["best"], res["recommended"],
                  warnings_, notes, {"additive": res})


# ----------------------------------------------------------------------------- distribution mode
def analyze_distribution(sample, ordered=False, include=None, exclude=None):
    name = str(getattr(sample, "name", None) or "y")
    y = np.asarray(sample, float).ravel()
    dropped = int((~np.isfinite(y)).sum())
    y = y[np.isfinite(y)]
    checks = sample_checks(y, ordered)
    res = run_distribution(y, include, exclude)
    table = res["table"]
    warnings_, notes = [], []
    if dropped:
        notes.append(f"dropped {dropped} missing or infinite values")
    if checks.get("looks_like_levels"):
        warnings_.append(f"neighbouring values are highly correlated (lag-1 = {checks['acf1']:.2f}): these look like "
                         "price levels, not returns — fit distributions to returns, e.g. np.diff(np.log(price))")
    if checks.get("lb_sq_p", 1) < 0.01:
        warnings_.append(f"squared values are significantly autocorrelated (Ljung–Box p = {checks['lb_sq_p']:.1e}): "
                         "volatility clusters, so the draws are not independent. This is the unconditional distribution; "
                         "for conditional risk such as tomorrow's VaR use a GARCH-type model")
    if checks.get("lb_p", 1) < 0.01 and not checks.get("looks_like_levels"):
        notes.append(f"the series itself is autocorrelated (Ljung–Box p = {checks['lb_p']:.1e})")
    rec_row = table.set_index("id").loc[res["recommended"]]
    worst = max(abs(rec_row["q01_err"]), abs(rec_row["q99_err"]))
    if worst > 0.15:
        thin = min(rec_row["q01_err"], rec_row["q99_err"]) < -0.15
        warnings_.append(f"the recommended distribution misses the tails (left {rec_row['q01_err']:+.0%}, right "
                         f"{rec_row['q99_err']:+.0%}): "
                         + ("tails too thin, so VaR-type risk will be understated" if thin
                            else "tails too fat, so tail risk will be overstated"))
    t = table.set_index("id")
    if "normal" in t.index and checks["excess_kurtosis"] > 1 and np.isfinite(t.loc["normal", "d_aic"]):
        notes.append(f"excess kurtosis {checks['excess_kurtosis']:.1f}: the normal distribution is "
                     f"ΔAIC = {t.loc['normal', 'd_aic']:.0f} behind the best — the tails are clearly heavier")
    if y.size < 200:
        notes.append("fewer than 200 observations: the 1% quantiles rest on a handful of points, so tail errors are "
                     "indicative only")
    profile = {"n": int(y.size), **checks}
    data = {"y": y, "y_name": name}
    return Report("distribution", data, profile, {"criterion": "AIC"}, table, res["models"], res["best"],
                  res["recommended"], warnings_, notes, {})
