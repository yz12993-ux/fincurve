"""Entry point: profile the data, choose the error model and validation scheme, run the right mode."""
import numpy as np
import pandas as pd

from .additive import SHAPE_LABELS, fit_linear, prepare_features, run_additive
from .distribution import run_distribution, sample_checks
from .families import get_family
from .profile import describe_relation, describe_target, random_walk_like
from .report import FAMILY_SHORT, Report
from .scoring import make_splits
from .univariate import extrapolation_spread, residual_checks, run_race

YEAR_SECONDS = 365.25 * 86400


def analyze(X, y=None, *, family="auto", cv="auto", k_folds=5, groups=None, weights=None, trials=None,
            time_ordered=None, include=None, exclude=None, n_starts=2, nested=True, interactions=True,
            random_state=0):
    """Characterise how y depends on X, or which distribution X follows when y is omitted.

    X            1-D array/Series (one x), DataFrame (one or more features), or the sample itself if y is None
    family       "auto" | "gaussian" | "lognormal" | "binomial" | "poisson"
    cv           "auto" | "kfold" | "time" | "loo" | "group"
    groups       group labels for grouped cross-validation (e.g. issuer, date)
    weights      observation weights; trials = number of trials behind each observed proportion
    time_ordered force (True) or forbid (False) time-series treatment; None lets the profile decide
    include / exclude   candidate keys or group names, e.g. include=["term_structure", "polynomial"]
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
        raise ValueError(f"X 有 {len(df)} 行，y 有 {y.size} 个值")

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
                random_state=random_state, dropped=dropped, y_name=y_name)

    single = df.shape[1] == 1 and (pd.api.types.is_numeric_dtype(df.iloc[:, 0])
                                   or pd.api.types.is_datetime64_any_dtype(df.iloc[:, 0]))
    if single:
        return _analyze_curve(df.iloc[:, 0], y, w, **opts)
    return _analyze_additive(df, y, w, nested=nested, interactions=interactions, **opts)


# ----------------------------------------------------------------------------- decisions
def _choose_families(family, target, y, mu_hint):
    if family != "auto":
        return [get_family(family)], "用户指定"
    kind = target["kind"]
    gaussian, lognormal = get_family("gaussian"), get_family("lognormal")
    if kind == "binary":
        return [get_family("binomial")], "y 只有 0 和 1，用二项（伯努利）似然"
    if kind == "proportion":
        return [get_family("binomial")], "y 在 0 到 1 之间，按比例/概率处理（二项似然）"
    if kind == "count":
        mu = np.clip(mu_hint, 1e-6, None)
        disp = float(np.mean((y - mu) ** 2 / mu))
        if 0.3 < disp < 3:
            return [get_family("poisson")], f"非负整数计数，离散度 {disp:.2f} 接近 1，用 Poisson 似然"
        fams = [gaussian] + ([lognormal] if y.min() > 0 else [])
        return fams, f"虽是整数计数，但离散度为 {disp:.1f}（Poisson 要求约为 1），改用连续误差模型"
    return None, None


def _decide_order(time_ordered, is_datetime, rel, name, y_sorted):
    if time_ordered is not None:
        return bool(time_ordered), "用户指定"
    if is_datetime:
        return True, "x 是日期时间"
    if rel.get("time_like_name"):
        return True, f"x 的名字“{name}”看起来是时间"
    if rel.get("equally_spaced") and rel.get("n", 0) >= 30:
        flag, rw = random_walk_like(y_sorted)
        if flag:
            return True, f"x 等间距，且 y 沿 x 像随机游走（相邻值相关 {rw['ar1']:.2f}，差分几乎不相关）"
    return False, "x 不像时间变量"


def _choose_cv(cv, n, ordered, groups):
    if cv != "auto":
        return cv, "用户指定"
    if groups is not None:
        return "group", "提供了分组标签"
    if ordered:
        return "time", "数据有时间顺序：随机打乱会用未来预测过去，分数会过于乐观"
    if n < 20:
        return "loo", f"样本只有 {n} 个，用留一法"
    return "kfold", "数据没有时间顺序"


def models_group(models, ranked, key):
    """Library group of a candidate key (the ranking stores Chinese group labels, models store keys)."""
    for m in models.values():
        if m.candidate.key == key:
            return m.candidate.group
    return None


# ----------------------------------------------------------------------------- curve mode
def _analyze_curve(xs, y, w, *, family, cv, k_folds, groups, trials, time_ordered, include, exclude,
                   n_starts, random_state, dropped, y_name):
    x_name = str(xs.name) if xs.name is not None else "x"
    is_datetime = pd.api.types.is_datetime64_any_dtype(xs)
    t0 = xs.min() if is_datetime else None
    x = ((xs - t0).dt.total_seconds().to_numpy() / YEAR_SECONDS) if is_datetime else xs.to_numpy(float)
    n = y.size
    warnings_, notes = [], []
    if dropped:
        notes.append(f"删除了 {dropped} 行含缺失值的数据")

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
                why = f"噪声随水平放大（ρ = {rel['hetero_rho']:.2f}）" if hetero else \
                    f"最大/最小值之比为 {target['max_min_ratio']:.1f}"
                families = [get_family("gaussian"), get_family("lognormal")]
                fam_reason = f"y 为正且{why}，同时比较加性误差和乘性误差"
            else:
                families, fam_reason = [get_family("gaussian")], "y 为正但跨度不大、噪声大致恒定，用加性误差"
        else:
            families, fam_reason = [get_family("gaussian")], "y 可正可负，用加性误差（正态）"

    ordered, order_reason = _decide_order(time_ordered, is_datetime, rel, x_name, y[order])
    strategy, cv_reason = _choose_cv(cv, n, ordered, groups)
    splits = make_splits(n, strategy, k_folds, order=order, groups=groups, seed=random_state)
    race = run_race(x, y, w, families, splits, n_starts, include, exclude, random_state, frac=rel["frac"])
    table, models = race["table"], race["models"]
    if race["recommended"] is None:
        raise RuntimeError("所有候选都拟合失败")
    rec, best = models[race["recommended"]], models[race["best"]]
    ranked = table[table["status"] == "ok"]

    # --- time-series paths
    extras = {}
    if ordered:
        flag, rw = random_walk_like(y[order])
        if flag:
            base = np.log(y[order]) if rw["log_scale"] else y[order]
            extras["returns"] = analyze_distribution(pd.Series(np.diff(base), name="对数收益率" if rw["log_scale"] else "一阶差分"),
                                                     ordered=True)
            warnings_.append(
                f"y 看起来像价格/净值这类随机游走路径（相邻值相关 {rw['ar1']:.2f}，差分几乎不相关 {rw['diff_acf1']:+.2f}）："
                "曲线形状只是这一条路径的偶然走势，外推没有意义。更合适的是研究"
                + ("对数收益率" if rw["log_scale"] else "一阶差分") + "的分布，已自动附在 report.extras['returns']")
        if order_reason != "用户指定":
            notes.append(f"按时间处理的原因：{order_reason}")
        notes.append("按时间滚动验证评估的是向前外推能力，比随机 K 折更严格，分数通常更差")

    # --- ties, extrapolation, reference
    ties = ranked[ranked["tie"]]
    multi = len(families) > 1
    names = [f"{row.label}［{FAMILY_SHORT[row.family]}］" if multi else row.label for row in ties.itertuples()]
    if len(ties) > 1:
        notes.append(f"与最优模型没有实质差距的共有 {len(ties)} 个：" + "、".join(names[:8]))
        if ties["group"].nunique() > 1:
            warnings_.append("并列模型分属不同类型（" + "、".join(ties["group"].unique()[:6]) +
                             "）：数据本身区分不了这些含义不同的形状，要结合经济含义选择，或补充 x 范围更宽的数据")
        spread = extrapolation_spread([models[i] for i in ties["id"].head(6)], x, y)
        if spread > 0.5:
            size = f"{spread:.0f} 倍" if spread >= 10 else f"{spread:.0%}"
            warnings_.append(f"这些并列模型在数据范围之外分歧很大（向外延伸 25% 的区间里，最大差距达到 y 全距的 {size}）：不要外推")
    for rid in race["reference_ids"]:
        ref = table.set_index("id").loc[rid]
        if np.isfinite(ref["d_cv"]) and ref["d_cv"] < 0 and abs(ref["d_cv"]) > 2 * (ref["se"] or 0):
            warnings_.append(f"非参数平滑比所有候选都好（ΔCV-NLL = {ref['d_cv']:.3g}）：候选库里可能缺少真实形状，请看残差图找规律")
    context = rel.get("context")
    if context:
        label, groups_ = context
        themed = ranked[ranked["model"].map(lambda k: models_group(models, ranked, k) in groups_)]
        if not themed.empty and rec.candidate.group not in groups_:
            row = themed.iloc[0]
            pos = int(ranked.index.get_loc(themed.index[0])) + 1
            status = "与最优没有实质差距" if row["tie"] else f"比最优差 {row['d_cv']:.3g} ± {row['se']:.2g}"
            notes.append(f"x 的名字提示这是{label}：该语境的专用形式“{row['label']}”排第 {pos}（{status}）。"
                         "专用形式的参数有明确经济含义，差距不大时可以优先使用")
        elif rec.candidate.group in groups_:
            notes.append(f"x 的名字提示这是{label}，推荐模型正好是该语境的专用形式")
    shape = rel["shape"]
    for label, m in (("推荐模型", rec), ("得分最高的模型", best)):
        c = m.candidate
        if "peak" in c.tags and c.group not in ("polynomial", "term_structure") and shape in ("increasing", "decreasing"):
            warnings_.append(f"{label}“{c.label}”是峰形函数，但数据在观测范围内是单调的：它可能只是在用峰的半边近似 S 形或饱和形状，"
                             "峰的位置没有数据支撑")
            break

    # --- residuals and data quality
    checks = residual_checks(rec, x, y)
    if not ordered and n >= 20 and checks["resid_lag1"] > 0.5:
        warnings_.append(f"推荐模型的残差沿 x 高度自相关（lag-1 = {checks['resid_lag1']:.2f}）：要么漏掉了形状，要么数据其实是"
                         "时间序列（随机 K 折会偏乐观，此时请传 time_ordered=True）")
    elif not ordered and n >= 20 and checks["resid_lag1"] > 0.3:
        warnings_.append(f"推荐模型的残差沿 x 仍有结构（lag-1 = {checks['resid_lag1']:.2f}）：可能漏掉了某种形状")
    if (rec.family.name == "gaussian" and checks.get("hetero_p", 1) < 0.01 and checks.get("hetero_rho", 0) > 0.2
            and len(families) == 1 and y.min() > 0):
        notes.append("残差随预测值增大而增大：可以试 family='lognormal'（乘性误差）")
    if rel["n_outliers"] or rel["resid_excess_kurtosis"] > 3:
        warnings_.append(f"残差有厚尾或离群点（{rel['n_outliers']} 个超过 5 倍 MAD）：最小二乘会被少数点拉动，建议检查这些点")
    if rel["max_gap_share"] > 0.3:
        notes.append(f"x 有大段空白（最大间隔占全距 {rel['max_gap_share']:.0%}）：空白区间里的形状没有数据支撑")
    if n < 15:
        warnings_.append(f"样本只有 {n} 个：排名很不稳定，只作参考")
    if target["kind"] == "proportion" and trials is None:
        notes.append("没有提供每个比例背后的样本数（trials），各点等权；提供后似然更准确")
    if race["skipped"]:
        notes.append("因定义域或样本量跳过的候选：" + "、".join(race["skipped"]))

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
                      n_starts, random_state, dropped, y_name, nested, interactions):
    n = y.size
    warnings_, notes = [], []
    if dropped:
        notes.append(f"删除了 {dropped} 行含缺失值的数据")
    feats = prepare_features(df)
    usable = [f for f in feats if f.kind != "excluded"]
    if not usable:
        raise ValueError("没有可用的特征：" + "；".join(f"{f.name}（{f.note}）" for f in feats))
    target = describe_target(y, trials)
    idx = np.arange(n)

    families, fam_reason = None, None
    if family == "auto" and target["kind"] == "count":
        design, theta, _ = fit_linear(usable, get_family("poisson"), idx, y, w)
        mu_hint = get_family("poisson").clip(design.mean_fn(idx, *theta))
        families, fam_reason = _choose_families(family, target, y, mu_hint)
    elif family != "auto" or target["kind"] in ("binary", "proportion"):
        families, fam_reason = _choose_families(family, target, y, None)
    if families is None or len(families) > 1:
        if y.min() > 0:
            aics = {name: fit_linear(usable, get_family(name), idx, y, w)[2] for name in ("gaussian", "lognormal")}
            pick = min(aics, key=aics.get)
            families = [get_family(pick)]
            gap = abs(aics["gaussian"] - aics["lognormal"])
            fam_reason = (fam_reason + "；" if fam_reason else "") + \
                f"y 为正：线性基准下{families[0].label}的 AIC 低 {gap:.1f}，因此选用它"
        else:
            families, fam_reason = [get_family("gaussian")], "y 可正可负，用加性误差（正态）"
    fam = families[0]

    time_feats = [f for f in usable if f.is_time]
    ordered = bool(time_ordered) if time_ordered is not None else bool(time_feats)
    order_reason = "用户指定" if time_ordered is not None else (f"特征“{time_feats[0].name}”是日期时间" if time_feats else "")
    strategy, cv_reason = _choose_cv(cv, n, ordered, groups)
    order = np.argsort(time_feats[0].raw, kind="stable") if time_feats else None
    splits = make_splits(n, strategy, k_folds, order=order, groups=groups, seed=random_state)

    res = run_additive(feats, fam, y, w, splits, nested=nested, interactions=interactions)
    table, models = res["table"], res["models"]

    for f in feats:
        if f.kind == "excluded":
            warnings_.append(f"特征“{f.name}”已剔除：{f.note}")
    numeric = [f for f in usable if f.kind == "numeric"]
    if len(numeric) >= 2:
        Z = np.column_stack([f.z for f in numeric])
        C = np.corrcoef(Z, rowvar=False)
        for i in range(len(numeric)):
            for j in range(i + 1, len(numeric)):
                if abs(C[i, j]) > 0.9:
                    warnings_.append(f"“{numeric[i].name}”和“{numeric[j].name}”高度相关（r = {C[i, j]:.2f}）："
                                     "两者的形状和重要性怎么分配并不稳定")
    feat_table = res["features"]
    if "stability" in feat_table:
        for row in feat_table.itertuples():
            stab = getattr(row, "stability", np.nan)
            full_shape = res["full_shapes"].get(row.feature)
            if full_shape and full_shape != "linear" and np.isfinite(stab) and stab < 0.9:
                warnings_.append(f"“{row.feature}”的非线性形状在各折之间不一致（效应曲线平均相关 {stab:.2f}）："
                                 "这条曲线的细节可能是噪声造成的")
    ref = table.set_index("id")
    if "knn" in ref.index and np.isfinite(ref.loc["knn", "d_cv"]) and ref.loc["knn", "d_cv"] < 0 \
            and abs(ref.loc["knn", "d_cv"]) > 2 * (ref.loc["knn", "se"] or 0):
        warnings_.append("K 近邻参照明显更好：可能存在加性结构表达不了的交互作用或局部模式")
    rec_row = ref.loc[res["recommended"]]
    if rec_row["k"] > n / 10:
        warnings_.append(f"推荐模型有 {int(rec_row['k'])} 个参数，而样本只有 {n} 个：容易过拟合")
    for h in res["history"]:
        notes.append(f"“{h['feature']}”：{SHAPE_LABELS[h['from']]} → {SHAPE_LABELS[h['to']]}（全样本 AIC 下降 {h['aic_gain']:.1f}）")
    for it in res["interactions"]:
        notes.append(f"加入交互项 {it['pair'][0]} × {it['pair'][1]}（AIC 下降 {it['aic_gain']:.1f}）")
    if ordered and order_reason:
        notes.append(f"按时间处理的原因：{order_reason}")

    kinds = {"数值特征": sum(f.kind == "numeric" for f in feats), "二元特征": sum(f.kind == "binary" for f in feats),
             "类别特征": sum(f.kind == "categorical" for f in feats), "剔除的特征": sum(f.kind == "excluded" for f in feats)}
    profile = {"n": n, "target": target, "feature_kinds": kinds}
    decisions = {"families": families, "family_reason": fam_reason, "cv": strategy, "cv_reason": cv_reason,
                 "n_splits": len(splits), "ordered": ordered}
    data = {"X": df, "y": y, "w": w, "columns": list(df.columns), "y_name": y_name}
    return Report("additive", data, profile, decisions, table, models, res["best"], res["recommended"],
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
        notes.append(f"删除了 {dropped} 个缺失/无穷值")
    if checks.get("looks_like_levels"):
        warnings_.append(f"相邻值高度相关（lag-1 = {checks['acf1']:.2f}），像价格水平而不是收益率："
                         "分布拟合通常应针对收益率，例如 np.diff(np.log(price))")
    if checks.get("lb_sq_p", 1) < 0.01:
        warnings_.append(f"平方序列显著自相关（Ljung–Box p = {checks['lb_sq_p']:.1e}）：存在波动聚集，独立同分布不成立。"
                         "这里拟合的是无条件分布；明天的 VaR 这类条件风险应使用 GARCH 类模型")
    if checks.get("lb_p", 1) < 0.01 and not checks.get("looks_like_levels"):
        notes.append(f"序列本身有自相关（Ljung–Box p = {checks['lb_p']:.1e}）")
    rec_row = table.set_index("id").loc[res["recommended"]]
    worst = max(abs(rec_row["q01_err"]), abs(rec_row["q99_err"]))
    if worst > 0.15:
        thin = min(rec_row["q01_err"], rec_row["q99_err"]) < -0.15
        warnings_.append(f"推荐分布的尾部偏差较大（左尾 {rec_row['q01_err']:+.0%}，右尾 {rec_row['q99_err']:+.0%}）："
                         + ("尾部偏薄，会低估 VaR 这类尾部风险" if thin else "尾部偏厚，尾部风险估计偏保守"))
    t = table.set_index("id")
    if "normal" in t.index and checks["excess_kurtosis"] > 1 and np.isfinite(t.loc["normal", "d_aic"]):
        notes.append(f"超额峰度 {checks['excess_kurtosis']:.1f}：正态分布比最优分布差 ΔAIC = {t.loc['normal', 'd_aic']:.0f}，尾部明显更厚")
    if y.size < 200:
        notes.append("样本少于 200 个：1% 尾部分位数主要由少数几个点决定，尾部误差仅供参考")
    profile = {"n": int(y.size), **checks}
    data = {"y": y, "y_name": name}
    decisions = {"criterion": "AIC"}
    return Report("distribution", data, profile, decisions, table, res["models"], res["best"], res["recommended"],
                  warnings_, notes, {})
