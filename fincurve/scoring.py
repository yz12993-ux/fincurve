"""Cross-validation splits and ranking with a paired one-standard-error rule."""
import numpy as np
import pandas as pd

CV_LABELS = {
    "kfold": "随机 K 折",
    "time": "按时间滚动（只用过去预测未来）",
    "loo": "留一法",
    "group": "按组 K 折（同一组不会同时出现在训练和测试里）",
}


def make_splits(n, strategy, k=5, order=None, groups=None, seed=0):
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    if strategy == "loo":
        return [(np.delete(idx, i), np.array([i])) for i in range(n)]
    if strategy == "kfold":
        return [(np.setdiff1d(idx, f), np.sort(f)) for f in np.array_split(rng.permutation(n), k)]
    if strategy == "time":
        o = idx if order is None else np.asarray(order)
        start = n // 2
        blocks = [b for b in np.array_split(o[start:], k) if len(b)]
        splits, used = [], start
        for b in blocks:
            splits.append((np.sort(o[:used]), np.sort(b)))
            used += len(b)
        return splits
    if strategy == "group":
        g = np.asarray(groups)
        uniq = rng.permutation(np.unique(g))
        parts = np.array_split(uniq, min(k, uniq.size))
        return [(idx[~np.isin(g, p)], idx[np.isin(g, p)]) for p in parts]
    raise ValueError(f"unknown cv strategy {strategy!r}")


PRACTICAL_GAP = 0.1   # nats per data row; for Gaussian errors this is roughly a 10% larger RMSE


def cv_score(test_nll, w=None):
    """Mean held-out NLL per data row (a row with n trials contributes the NLL of all its trials)."""
    m = np.isfinite(test_nll)
    return float(np.mean(test_nll[m])) if m.any() else np.inf


def paired_difference(a, b, w=None):
    """Mean and standard error of the per-row held-out NLL difference a − b."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return np.nan, np.nan
    d = a[m] - b[m]
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(m.sum()))


def rank_table(rows, test_nll, w, reference_ids=()):
    """Sort by CV NLL; a model ties with the best when its gap is within 2 paired standard errors
    and also below PRACTICAL_GAP (small samples otherwise produce huge, uninformative tie sets).
    The recommendation is the tied model with the fewest parameters."""
    df = pd.DataFrame(rows)
    df["cv_nll"] = [cv_score(test_nll[i], w) if i in test_nll else np.inf for i in df["id"]]
    df = df.sort_values("cv_nll", kind="stable").reset_index(drop=True)
    eligible = df[np.isfinite(df["cv_nll"]) & ~df["id"].isin(reference_ids)]
    df["d_cv"], df["se"], df["tie"] = np.nan, np.nan, False
    if eligible.empty:
        return df, None, None
    best = eligible["id"].iloc[0]
    for i, row in df.iterrows():
        if not np.isfinite(row["cv_nll"]):
            continue
        d, se = (0.0, 0.0) if row["id"] == best else paired_difference(test_nll[row["id"]], test_nll[best], w)
        df.loc[i, ["d_cv", "se"]] = d, se
        df.loc[i, "tie"] = bool(row["id"] == best or (np.isfinite(d) and d <= 2 * se and d <= PRACTICAL_GAP))
    df.loc[df["id"].isin(reference_ids), "tie"] = False
    ties = df[df["tie"]].sort_values(["k", "cv_nll"], kind="stable")
    return df, best, ties["id"].iloc[0]
