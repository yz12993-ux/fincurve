"""Figures for the three report modes."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

from .profile import local_linear

SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#b5b4ae"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
LINESTYLES = ["-", "--", ":"]
DATA = "#8a8984"
FAMILY_SHORT = {"gaussian": "加性", "lognormal": "乘性", "binomial": "二项", "poisson": "Poisson"}
EFFECT_AXIS = {"gaussian": "对 y 的贡献", "lognormal": "对 ln y 的贡献（乘性）",
               "binomial": "对 logit(p) 的贡献", "poisson": "对 ln(均值) 的贡献"}


def _style():
    available = {f.name for f in font_manager.fontManager.ttflist}
    cjk = [f for f in ("Hiragino Sans GB", "PingFang SC", "Heiti SC", "Microsoft YaHei", "Noto Sans CJK SC",
                       "SimHei", "Arial Unicode MS") if f in available]
    return {
        "font.family": cjk + ["DejaVu Sans"], "axes.unicode_minus": False, "font.size": 9,
        "mathtext.fontset": "dejavusans",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "legend.frameon": False, "axes.titlelocation": "left", "axes.titlesize": 11, "axes.titleweight": "bold",
    }


def _valid_grid(model, grid, x):
    dom = model.candidate.domain
    keep = np.ones_like(grid, bool)
    if "x_pos" in dom:
        keep &= grid > 0
    if "x_nonneg" in dom:
        keep &= grid >= 0
    if "x_same_sign" in dom:
        keep &= np.sign(grid) == np.sign(x.min())
    return grid[keep]


def _ranking_panel(ax, r, value="d_cv", err="se", xlabel="与最优模型的差距（每个观测的 CV 负对数似然）", limit=15):
    t = r.ranking
    t = t[np.isfinite(t[value])].head(limit)
    ypos = np.arange(len(t))[::-1]
    labels = []
    multi = "family" in t and t["family"].nunique() > 1
    for yy, row in zip(ypos, t.itertuples()):
        is_ref = getattr(row, "status", "") == "reference"
        color = SERIES[0] if row.id == r.recommended_id else (INK2 if getattr(row, "tie", False) else MUTED)
        e = getattr(row, err, np.nan) if err else np.nan
        ax.errorbar(max(getattr(row, value), 0), yy, xerr=e if np.isfinite(e) else None,
                    fmt="D" if is_ref else "o", ms=6, capsize=2, color=color,
                    mfc=SURFACE if is_ref else color, mew=1.5, lw=1.2)
        name = row.label + (f"［{FAMILY_SHORT.get(row.family, row.family)}］" if multi else "")
        labels.append(("★ " if row.id == r.recommended_id else "") + name)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xscale("symlog", linthresh=0.01 if value == "d_cv" else 1.0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlim(left=0)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


def _axis_values(r, v):
    """Map internal x (years since the first date, for datetime inputs) back to what the user passed."""
    t0 = r.data.get("t0")
    if t0 is None:
        return v
    return pd.Timestamp(t0) + pd.to_timedelta(np.asarray(v) * 365.25 * 86400, unit="s")


def plot_curve(r, top=3):
    x, y = r.data["x"], r.data["y"]
    X = _axis_values(r, x)
    table = r.ranking
    ranked = [i for i in table["id"] if i in r.models]
    shown = ([r.recommended_id] if r.recommended_id else []) + [i for i in ranked if i != r.recommended_id]
    shown = shown[:top]
    multi = table["family"].nunique() > 1
    with plt.rc_context(_style()):
        fig = plt.figure(figsize=(15, 8.5))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.65, 1], height_ratios=[2.2, 1], hspace=0.38, wspace=0.32)
        ax, axr, axk = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])

        if r.profile["target"]["kind"] == "binary":
            jitter = (np.random.default_rng(0).random(y.size) - 0.5) * 0.04
            ax.scatter(X, y + jitter, s=6, color=DATA, alpha=0.3, edgecolor="none", label="观测（0/1，加了抖动）")
            edges = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
            ids = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
            xb, pb, eb = [], [], []
            for b in np.unique(ids):
                sel = ids == b
                p = y[sel].mean()
                xb.append(x[sel].mean())
                pb.append(p)
                eb.append(1.96 * np.sqrt(p * (1 - p) / sel.sum()))
            ax.errorbar(_axis_values(r, np.array(xb)), pb, yerr=eb, fmt="o", color=INK2, ms=5, capsize=2,
                        label="分箱事件率 ±95%")
        else:
            big = x.size > 500
            ax.scatter(X, y, s=8 if big else 18, color=DATA, alpha=0.35 if big else 0.75, edgecolor="none", label="数据")

        grid = np.linspace(0.0 if r.data["term_axis"] else x.min(), x.max(), 400)
        for i, mid in enumerate(shown):
            m = r.models[mid]
            g = _valid_grid(m, grid, x)
            row = table.set_index("id").loc[mid]
            rank = ranked.index(mid) + 1
            tag = "推荐" if mid == r.recommended_id else f"第 {rank} 名"
            fam = f"［{FAMILY_SHORT[m.family.name]}］" if multi else ""
            d = row["d_cv"]
            score = f"ΔCV-NLL {d:.3g}" if np.isfinite(d) else "CV 失败"
            ax.plot(_axis_values(r, g), m.predict(g), color=SERIES[i], ls=LINESTYLES[i], lw=2,
                    label=f"{tag}：{m.label}{fam}（{score}）")
        if r.data["term_axis"]:
            ax.set_xlim(left=0)
        ax.set_title(f"{r.data['y_name']} 对 {r.data['x_name']}：数据与排名靠前的曲线")
        ax.set_xlabel(r.data["x_name"])
        ax.set_ylabel(r.data["y_name"])
        ax.legend(fontsize=8.5, loc="best")

        rec = r.recommended
        if rec is not None:
            res = y - rec.predict(x)
            axr.scatter(X, res, s=8 if x.size > 500 else 12, color=DATA, alpha=0.6, edgecolor="none")
            axr.axhline(0, color=INK2, lw=1)
            if x.size >= 20:
                o = np.argsort(x, kind="stable")
                sub = o if o.size <= 1500 else o[np.linspace(0, o.size - 1, 1500).astype(int)]
                axr.plot(_axis_values(r, x[sub]), local_linear(x[sub], res[sub], frac=0.3), color=SERIES[0], lw=1.5,
                         label="残差平滑线")
                axr.legend(fontsize=8, loc="best")
            if r.data["term_axis"]:
                axr.set_xlim(left=0)
            axr.set_title(f"推荐模型的残差（{rec.label}）")
            axr.set_xlabel(r.data["x_name"])
            axr.set_ylabel("残差")

        _ranking_panel(axk, r)
        axk.set_title("候选排名（◇ = 非参数参照，误差线 = 1 个标准误）")
        title = "fincurve 曲线刻画"
        if "returns" in r.extras:
            title += "（随机游走路径：曲线只反映这条路径的偶然走势，不要据此外推）"
        fig.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold")
        fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.08)
    return fig


def plot_distribution(r, top=3):
    y = r.data["y"]
    ok = r.ranking[r.ranking["status"] == "ok"]
    shown = [r.recommended_id] + [i for i in ok["id"] if i != r.recommended_id]
    shown = [i for i in shown if i in r.models][:top]
    with plt.rc_context(_style()):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        (ah, al), (aq, ak) = axes
        lo, hi = np.quantile(y, [0.001, 0.999])
        pad = 0.1 * (hi - lo)
        grid = np.linspace(lo - pad, hi + pad, 500)
        if r.profile["positive"]:
            grid = grid[grid > 0]
        bins = min(80, max(15, int(np.sqrt(y.size))))
        ah.hist(y, bins=bins, range=(lo - pad, hi + pad), density=True, color=MUTED, edgecolor=SURFACE, label="数据")
        counts, edges = np.histogram(y, bins=bins, range=(lo - pad, hi + pad), density=True)
        centres = (edges[:-1] + edges[1:]) / 2
        al.scatter(centres[counts > 0], counts[counts > 0], s=14, color=DATA, label="数据（直方图密度）")
        for i, key in enumerate(shown):
            m = r.models[key]
            tag = "推荐" if key == r.recommended_id else "对比"
            with np.errstate(all="ignore"):
                dens = m.pdf(grid)
            for axis in (ah, al):
                axis.plot(grid, dens, color=SERIES[i], ls=LINESTYLES[i], lw=2, label=f"{tag}：{m.label}")
        ah.set_title("密度")
        al.set_yscale("log")
        al.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        al.set_ylim(bottom=max(counts[counts > 0].min() / 5, 1e-6))
        al.set_title("对数密度（看尾部）")
        for axis in (ah, al):
            axis.set_xlabel(r.data["y_name"])
            axis.legend(fontsize=8.5)

        rec = r.recommended
        if rec is not None:
            q = (np.arange(1, y.size + 1) - 0.5) / y.size
            ys = np.sort(y)
            with np.errstate(all="ignore"):
                theo = rec.ppf(q)
            aq.scatter(theo, ys, s=8, color=DATA, alpha=0.6, edgecolor="none")
            lims = [np.nanmin([theo.min(), ys.min()]), np.nanmax([theo.max(), ys.max()])]
            aq.plot(lims, lims, color=SERIES[0], lw=1.5)
            aq.set_title(f"QQ 图（{rec.label}）：尾部偏离对角线 = 尾部拟合不好")
            aq.set_xlabel("模型分位数")
            aq.set_ylabel("经验分位数")

        _ranking_panel(ak, r, value="d_aic", err=None, xlabel="ΔAIC（越小越好）")
        ak.set_title("分布排名")
        fig.suptitle(f"fincurve 分布刻画：{r.data['y_name']}", x=0.01, ha="left", fontsize=14, fontweight="bold")
        fig.subplots_adjust(left=0.06, right=0.98, top=0.9, bottom=0.07, hspace=0.35, wspace=0.28)
    return fig


def plot_additive(r, top=3):
    model = r.recommended
    effects = model.partial_effects()
    feats = r.extras["additive"]["features"].set_index("feature")
    names = list(effects)[:11]
    panels = len(names) + 1
    ncols = 3
    nrows = int(np.ceil(panels / ncols))
    fam = model.design.family.name
    height = 4.6 * nrows + 1.0
    global_span = max((np.ptp(eff) if kind == "numeric" else np.max(np.abs(eff)))
                      for kind, _, eff, _ in (effects[n] for n in names)) or 1.0
    with plt.rc_context(_style()):
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, height), squeeze=False)
        flat = axes.ravel()
        for ax, name in zip(flat, names):
            kind, xs, eff, raw = effects[name]
            info = feats.loc[name]
            if kind == "numeric":
                ax.plot(xs, eff, color=SERIES[0], lw=2)
                sample = raw if raw.size <= 400 else np.random.default_rng(0).choice(raw, 400, replace=False)
                lo, span = np.nanmin(eff), (np.ptp(eff) or 1.0)
                ax.plot(sample, np.full(sample.size, lo - 0.12 * span), "|", color=DATA, ms=8, alpha=0.5)
                ax.set_ylim(lo - 0.2 * span, np.nanmax(eff) + 0.08 * span)
                ax.set_xlabel(f"{name}（下方短线 = 数据分布）")
            else:
                order = np.argsort(eff)
                eff, xs = np.asarray(eff)[order], [xs[i] for i in order]
                ypos = np.arange(len(xs))
                ax.hlines(ypos, 0, eff, color=MUTED, lw=2)
                ax.plot(eff, ypos, "o", color=SERIES[0], ms=7)
                ax.axvline(0, color=INK2, lw=1)
                ax.set_yticks(ypos)
                ax.set_yticklabels([str(v) for v in xs])
                ax.grid(axis="y", visible=False)
                # same scale as the largest effect, so a negligible feature does not look important
                lo, hi = min(eff.min(), 0.0), max(eff.max(), 0.0)
                pad = max(0.0, 0.5 * global_span - (hi - lo)) / 2 + 0.05 * global_span
                ax.set_xlim(lo - pad, hi + pad)
            trend = info.get("trend")
            desc = trend if isinstance(trend, str) else info["shape"]
            ax.set_title(f"{name}：{desc}（重要性 {info['importance']:.0%}）", fontsize=10)
            if kind == "numeric":
                ax.set_ylabel(EFFECT_AXIS[fam])
            else:
                ax.set_xlabel(EFFECT_AXIS[fam] + "（相对基准类别）")
        ak = flat[len(names)]
        _ranking_panel(ak, r)
        ak.set_title("模型比较（◇ = K 近邻参照）", fontsize=10)
        for ax in flat[len(names) + 1:]:
            ax.axis("off")
        fig.suptitle(f"fincurve 多特征刻画：{r.data['y_name']}（推荐：{model.label}）",
                     x=0.01, ha="left", fontsize=14, fontweight="bold")
        fig.subplots_adjust(left=0.08, right=0.98, top=1 - 0.9 / height, bottom=0.75 / height,
                            hspace=0.5, wspace=0.4)
    return fig
