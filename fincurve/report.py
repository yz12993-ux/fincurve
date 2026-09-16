"""Report object: ranking, recommended model, plain-language summary, plots, predictions."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .profile import PATTERNS, SHAPES, TARGET_KINDS
from .scoring import CV_LABELS

FAMILY_SHORT = {"gaussian": "加性", "lognormal": "乘性", "binomial": "二项", "poisson": "Poisson"}


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
        raise ValueError("分布模式没有 predict：请用 report.recommended.pdf / cdf / ppf")

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
        rec = self.recommended_id
        return f"<fincurve.Report mode={self.mode} n={self.profile.get('n')} recommended={rec!r}>"


def _bullets(title, items):
    return [f"【{title}】"] + [f"· {s}" for s in items] if items else []


def _ranking_block(r, top):
    t = r.ranking.head(top)
    multi = t["family"].nunique() > 1 if "family" in t else False
    lines = []
    for i, row in enumerate(t.itertuples(), 1):
        tag = "★" if row.id == r.recommended_id else ("=" if getattr(row, "tie", False) else " ")
        name = f"{row.label}" + (f"［{FAMILY_SHORT.get(row.family, row.family)}］" if multi else "")
        d = "参照" if row.status == "reference" else _num(row.d_cv)
        se = _num(row.se, 2)
        parts = [f"{tag}{i:>2}. {name:<22}", f"k={_num(row.k, 2):>2}", f"ΔCV-NLL={d:>8} ±{se:<6}",
                 f"CV-RMSE={_num(getattr(row, 'cv_rmse', np.nan)):>8}"]
        if "d_aic" in t and np.isfinite(getattr(row, "d_aic", np.nan)):
            parts.append(f"ΔAIC={getattr(row, 'd_aic'):>7.1f}")
        if row.status not in ("ok", "reference"):
            parts.append(f"（{row.status}）")
        lines.append("  ".join(parts))
    return (["【排名】（ΔCV-NLL = 与最优模型相比、每个观测的留出负对数似然之差；★ 推荐；= 与最优的差距在 2 个标准误以内）"]
            + lines)


def _curve_summary(r, top):
    p, d = r.profile, r.decisions
    rel, tgt = p["relation"], p["target"]
    shape = SHAPES[rel["shape"]] + (f"，{PATTERNS[rel['pattern']]}" if rel.get("pattern") else "")
    hetero = rel["hetero_p"] < 0.05 and rel["hetero_rho"] > 0.15
    lines = [f"══ fincurve 刻画报告：曲线模式（{r.data['y_name']} 对 {r.data['x_name']}）══"]
    lines += _bullets("数据画像", [
        f"样本 {p['n']} 个",
        f"y：{TARGET_KINDS[tgt['kind']]}，范围 [{_num(tgt['min'], 4)}, {_num(tgt['max'], 4)}]",
        f"形状：{shape}；局部平滑能解释 {rel['smooth_r2']:.0%} 的变化",
        "噪声：" + (f"随水平放大（Spearman ρ = {rel['hetero_rho']:.2f}）" if hetero else "大致恒定")
        + ("；x 等间距" if rel["equally_spaced"] else ""),
    ])
    lines += _bullets("自动选择", [
        f"误差模型：{' + '.join(f.label for f in d['families'])}——{d['family_reason']}",
        f"验证方式：{CV_LABELS[d['cv']]}，{d['n_splits']} 次切分——{d['cv_reason']}",
        f"候选：比较了 {d['n_candidates']} 个函数" + (f"，跳过 {len(d['skipped'])} 个（定义域或样本量不满足）"
                                                if d["skipped"] else ""),
    ])
    lines += _ranking_block(r, top)
    rec = r.recommended
    returns = r.extras.get("returns")
    if returns is not None:
        ret = returns.recommended
        lines += _bullets("推荐", [
            "不建议用任何曲线描述这组数据：它是随机游走路径（见“需要注意”），上面的排名只反映这一条路径的偶然走势",
            f"{returns.data['y_name']}的分布：推荐 {ret.label}（{ret.meaning}），完整结果见 report.extras['returns'].summary()",
        ])
        rec = None
    if rec is not None:
        c = rec.candidate
        params = "，".join(f"{k} = {_num(v, 4)}" for k, v in rec.param_dict().items())
        row = r.ranking.set_index("id").loc[r.recommended_id]
        ties = r.ranking[r.ranking["tie"]]
        why = (f"与最优模型差距在 2 个标准误以内的 {len(ties)} 个模型里参数最少" if len(ties) > 1 and r.recommended_id != r.best_id
               else "交叉验证分数最优" + (f"，且与其并列的 {len(ties) - 1} 个模型参数都不比它少" if len(ties) > 1 else ""))
        lines += _bullets("推荐", [
            f"{c.label}［{rec.family.label}］：{c.formula}",
            f"含义：{c.meaning}",
            f"参数：{params}" + ("（u = (x − 最小值)/全距）" if c.uses_u else ""),
            f"理由：{why}；样本内 R² = {_num(row['r2'])}",
        ])
    lines += _bullets("需要注意", r.warnings)
    lines += _bullets("备注", r.notes)
    return "\n".join(lines)


def _additive_summary(r, top):
    p, d, res = r.profile, r.decisions, r.extras["additive"]
    kinds = p["feature_kinds"]
    lines = [f"══ fincurve 刻画报告：多特征模式（{r.data['y_name']} 对 {len(r.data['columns'])} 个特征）══"]
    lines += _bullets("数据画像", [
        f"样本 {p['n']} 个；y：{TARGET_KINDS[p['target']['kind']]}",
        "特征：" + "，".join(f"{v} 个{k}" for k, v in kinds.items() if v),
    ])
    lines += _bullets("自动选择", [
        f"误差模型：{d['families'][0].label}——{d['family_reason']}",
        f"验证方式：{CV_LABELS[d['cv']]}，{d['n_splits']} 次切分——{d['cv_reason']}",
        "形状搜索：每个数值特征从线性、二次、三次、对数、平方根、反比、幂律、指数、S 形、单峰、折线中选择，"
        "AIC 至少下降 4 才换形状；交互项 AIC 至少下降 10 才加入",
        "交叉验证在每一折里重新选择形状（嵌套），分数没有被形状搜索美化" if res["nested"]
        else "形状只在全样本上选择一次，交叉验证只重估参数：分数略偏乐观",
    ])
    lines += _ranking_block(r, top)
    feat = res["features"].copy()
    body = []
    for row in feat.itertuples():
        if row.type == "已剔除":
            body.append(f"{row.feature}：已剔除——{row.note}")
            continue
        stab = getattr(row, "stability", np.nan)
        trend = getattr(row, "trend", np.nan)
        shape = f"{trend}（{row.shape}）" if isinstance(trend, str) else row.shape
        extra = f"，各折效应曲线相关 {stab:.2f}" if isinstance(stab, float) and np.isfinite(stab) else ""
        note = f"（{row.note}）" if isinstance(row.note, str) and row.note else ""
        body.append(f"{row.feature}［{row.type}］：{shape}，重要性 {row.importance:.0%}{extra}{note}")
    lines += _bullets(f"推荐模型：{r.models[r.recommended_id].label}", body)
    lines += _bullets("需要注意", r.warnings)
    lines += _bullets("备注", r.notes)
    return "\n".join(lines)


def _distribution_summary(r, top):
    p = r.profile
    lines = [f"══ fincurve 刻画报告：分布模式（{r.data['y_name']}）══"]
    lines += _bullets("数据画像", [
        f"样本 {p['n']} 个；均值 {_num(p['mean'])}，标准差 {_num(p['std'])}",
        f"偏度 {p['skew']:.2f}，超额峰度 {p['excess_kurtosis']:.2f}（正态为 0）",
    ])
    t = r.ranking.head(top)
    rows = ["【排名】（按 AIC；★ 推荐，= 与最优相差不到 2；尾部误差：负数 = 模型尾部比实际薄，正数 = 偏厚）"]
    for i, row in enumerate(t.itertuples(), 1):
        tag = "★" if row.id == r.recommended_id else ("=" if row.tie else " ")
        if row.status != "ok":
            rows.append(f"{tag}{i:>2}. {row.label:<14} （{row.status}）")
            continue
        rows.append(f"{tag}{i:>2}. {row.label:<14} k={int(row.k)}  ΔAIC={row.d_aic:>8.1f}  KS={row.ks:.3f}  "
                    f"左尾(1%) {row.q01_err:+.0%}  右尾(99%) {row.q99_err:+.0%}")
    lines += rows
    rec = r.recommended
    if rec is not None:
        lines += _bullets("推荐", [
            f"{rec.label}：{rec.meaning}",
            f"1% / 99% 分位数：{_num(rec.ppf(0.01), 4)} / {_num(rec.ppf(0.99), 4)}"
            f"（经验值 {_num(np.quantile(r.data['y'], 0.01), 4)} / {_num(np.quantile(r.data['y'], 0.99), 4)}）",
            "尾部误差 = 模型分位数到中位数的距离 ÷ 经验分位数到中位数的距离 − 1",
        ])
    lines += _bullets("需要注意", r.warnings)
    lines += _bullets("备注", r.notes)
    return "\n".join(lines)
