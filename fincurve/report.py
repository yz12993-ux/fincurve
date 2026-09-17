"""Report object: ranking, recommended model, plain-language summary, plots, predictions."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .profile import PATTERNS, SHAPES, TARGET_KINDS
from .scoring import CV_LABELS

FAMILY_SHORT = {"gaussian": "additive", "lognormal": "multiplicative", "binomial": "binomial", "poisson": "Poisson"}


def _num(v, digits=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "-"
    return f"{v:.{digits}g}"


@dataclass
class Report:
    mode: str                      # "curve" | "additive" | "distribution"
    data: dict
    profile: dict
    decisions: dict
    ranking: pd.DataFrame
    models: dict
    best_id: object
    recommended_id: object
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)

    @property
    def recommended(self):
        return self.models.get(self.recommended_id)

    def predict(self, X, model=None):
        m = self.models[model or self.recommended_id]
        if self.mode == "additive":
            return m.predict(X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.data["columns"]))
        if self.mode == "curve":
            x = X
            if self.data.get("t0") is not None:
                x = (pd.to_datetime(pd.Series(X)) - self.data["t0"]).dt.total_seconds().to_numpy() / (365.25 * 86400)
            return m.predict(np.asarray(x, float))
        raise ValueError("distribution mode has no predict(); use report.recommended.pdf / cdf / ppf")

    # ------------------------------------------------------------------ text
    def summary(self, top=10, show=True):
        text = {"curve": _curve_summary, "additive": _additive_summary,
                "distribution": _distribution_summary}[self.mode](self, top)
        if show:
            print(text)
        return text

    def plot(self, path=None, top=3):
        from . import plotting
        fig = {"curve": plotting.plot_curve, "additive": plotting.plot_additive,
               "distribution": plotting.plot_distribution}[self.mode](self, top)
        if path:
            fig.savefig(path, dpi=150)
        return fig

    def __repr__(self):
        return f"<fincurve.Report mode={self.mode} n={self.profile.get('n')} recommended={self.recommended_id!r}>"


def _section(title, items):
    return [f"[{title}]"] + [f"  - {s}" for s in items] if items else []


def _ranking_block(r, top):
    t = r.ranking.head(top)
    multi = t["family"].nunique() > 1 if "family" in t else False
    lines = ["[Ranking]  ΔCV-NLL = held-out negative log-likelihood per row relative to the best model;",
             "           * recommended, = practically tied with the best (within 2 paired SE and 0.1 nats)"]
    for i, row in enumerate(t.itertuples(), 1):
        tag = "*" if row.id == r.recommended_id else ("=" if getattr(row, "tie", False) else " ")
        name = row.label + (f" [{FAMILY_SHORT.get(row.family, row.family)}]" if multi else "")
        d = "reference" if row.status == "reference" else _num(row.d_cv)
        parts = [f"{tag}{i:>2}. {name:<42}", f"k={_num(row.k, 2):>2}", f"ΔCV-NLL={d:>9} ±{_num(row.se, 2):<7}",
                 f"CV-RMSE={_num(getattr(row, 'cv_rmse', np.nan)):>8}"]
        if "d_aic" in t and np.isfinite(getattr(row, "d_aic", np.nan)):
            parts.append(f"ΔAIC={getattr(row, 'd_aic'):>7.1f}")
        if row.status not in ("ok", "reference"):
            parts.append(f"({row.status})")
        lines.append("  ".join(parts))
    return lines


def _curve_summary(r, top):
    p, d = r.profile, r.decisions
    rel, tgt = p["relation"], p["target"]
    shape = SHAPES[rel["shape"]] + (f", {PATTERNS[rel['pattern']]}" if rel.get("pattern") else "")
    hetero = rel["hetero_p"] < 0.05 and rel["hetero_rho"] > 0.15
    lines = [f"== fincurve report: curve mode ({r.data['y_name']} vs {r.data['x_name']}) =="]
    lines += _section("Data profile", [
        f"{p['n']} observations",
        f"y: {TARGET_KINDS[tgt['kind']]}, range [{_num(tgt['min'], 4)}, {_num(tgt['max'], 4)}]",
        f"shape: {shape}; a local smoother explains {rel['smooth_r2']:.0%} of the variation",
        "noise: " + (f"grows with the level (Spearman ρ = {rel['hetero_rho']:.2f})" if hetero else "roughly constant")
        + ("; x is equally spaced" if rel["equally_spaced"] else ""),
    ])
    skipped = f", {len(d['skipped'])} skipped (domain or sample size)" if d["skipped"] else ""
    lines += _section("Automatic choices", [
        f"error model: {' + '.join(f.label for f in d['families'])} — {d['family_reason']}",
        f"validation: {CV_LABELS[d['cv']]}, {d['n_splits']} splits — {d['cv_reason']}",
        f"candidates: {d['n_candidates']} fitted{skipped}",
    ])
    lines += _ranking_block(r, top)
    rec = r.recommended
    returns = r.extras.get("returns")
    if returns is not None:
        ret = returns.recommended
        lines += _section("Recommendation", [
            "no curve is recommended: the data are a random-walk path (see Warnings), so the ranking above only "
            "reflects the accidental shape of this one path",
            f"distribution of the {returns.data['y_name']}s: {ret.label} ({ret.meaning}); "
            "see report.extras['returns'].summary()",
        ])
        rec = None
    if rec is not None:
        c = rec.candidate
        params = ", ".join(f"{k} = {_num(v, 4)}" for k, v in rec.param_dict().items())
        row = r.ranking.set_index("id").loc[r.recommended_id]
        ties = r.ranking[r.ranking["tie"]]
        if len(ties) > 1 and r.recommended_id != r.best_id:
            why = f"fewest parameters among the {len(ties)} models practically tied with the best"
        elif len(ties) > 1:
            why = f"best cross-validated score, and none of the {len(ties) - 1} tied models has fewer parameters"
        else:
            why = "best cross-validated score"
        lines += _section("Recommendation", [
            f"{c.label} [{rec.family.label}]: {c.formula}",
            f"meaning: {c.meaning}",
            f"parameters: {params}" + (" (u = (x − min) / range)" if c.uses_u else ""),
            f"why: {why}; in-sample R² = {_num(row['r2'])}",
        ])
    lines += _section("Warnings", r.warnings)
    lines += _section("Notes", r.notes)
    return "\n".join(lines)


def _additive_summary(r, top):
    p, d, res = r.profile, r.decisions, r.extras["additive"]
    kinds = p["feature_kinds"]
    lines = [f"== fincurve report: multi-feature mode ({r.data['y_name']} vs {len(r.data['columns'])} features) =="]
    lines += _section("Data profile", [
        f"{p['n']} observations; y: {TARGET_KINDS[p['target']['kind']]}",
        "features: " + ", ".join(f"{v} {k}" for k, v in kinds.items() if v),
    ])
    lines += _section("Automatic choices", [
        f"error model: {d['families'][0].label} — {d['family_reason']}",
        f"validation: {CV_LABELS[d['cv']]}, {d['n_splits']} splits — {d['cv_reason']}",
        "shape search: each numeric feature chooses among linear, quadratic, cubic, log, square root, reciprocal, "
        "power, exponential, sigmoid, peak and hinge; a shape is accepted only if AIC drops by at least 4, and an "
        "interaction only if AIC drops by at least 10",
        "the shape search is repeated inside every fold (nested), so the score is not flattered by it" if res["nested"]
        else "shapes were chosen once on the full sample and only refitted per fold: the score is slightly optimistic",
    ])
    lines += _ranking_block(r, top)
    body = []
    for row in res["features"].itertuples():
        if row.type == "excluded":
            body.append(f"{row.feature}: removed — {row.note}")
            continue
        stab = getattr(row, "stability", np.nan)
        trend = getattr(row, "trend", np.nan)
        shape = f"{trend} ({row.shape})" if isinstance(trend, str) else row.shape
        extra = f", fold-to-fold curve correlation {stab:.2f}" if isinstance(stab, float) and np.isfinite(stab) else ""
        note = f" ({row.note})" if isinstance(row.note, str) and row.note else ""
        body.append(f"{row.feature} [{row.type}]: {shape}, importance {row.importance:.0%}{extra}{note}")
    lines += _section(f"Recommended model: {r.models[r.recommended_id].label}", body)
    lines += _section("Warnings", r.warnings)
    lines += _section("Notes", r.notes)
    return "\n".join(lines)


def _distribution_summary(r, top):
    p = r.profile
    lines = [f"== fincurve report: distribution mode ({r.data['y_name']}) =="]
    lines += _section("Data profile", [
        f"{p['n']} observations; mean {_num(p['mean'])}, standard deviation {_num(p['std'])}",
        f"skewness {p['skew']:.2f}, excess kurtosis {p['excess_kurtosis']:.2f} (0 for a normal distribution)",
    ])
    rows = ["[Ranking]  by AIC; * recommended, = within 2 of the best; tail error < 0 means the model's tail is too thin"]
    for i, row in enumerate(r.ranking.head(top).itertuples(), 1):
        tag = "*" if row.id == r.recommended_id else ("=" if row.tie else " ")
        if row.status != "ok":
            rows.append(f"{tag}{i:>2}. {row.label:<24} ({row.status})")
            continue
        rows.append(f"{tag}{i:>2}. {row.label:<24} k={int(row.k)}  ΔAIC={row.d_aic:>8.1f}  KS={row.ks:.3f}  "
                    f"left tail (1%) {row.q01_err:+.0%}  right tail (99%) {row.q99_err:+.0%}")
    lines += rows
    rec = r.recommended
    if rec is not None:
        lines += _section("Recommendation", [
            f"{rec.label}: {rec.meaning}",
            f"1% / 99% quantiles: {_num(rec.ppf(0.01), 4)} / {_num(rec.ppf(0.99), 4)} "
            f"(empirical {_num(np.quantile(r.data['y'], 0.01), 4)} / {_num(np.quantile(r.data['y'], 0.99), 4)})",
            "tail error = distance of the model quantile from the median ÷ distance of the empirical quantile − 1",
        ])
    lines += _section("Warnings", r.warnings)
    lines += _section("Notes", r.notes)
    return "\n".join(lines)
