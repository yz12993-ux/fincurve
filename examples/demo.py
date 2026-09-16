"""Run fincurve on synthetic financial datasets and save summaries and figures to examples/output/."""
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import scenarios as S  # noqa: E402
from fincurve import analyze  # noqa: E402

OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

CASES = [
    ("01_yield_curve", S.yield_curve),
    ("02_vol_term_structure", S.vol_term_structure),
    ("03_vol_smile", S.vol_smile),
    ("04_bond_price", S.bond_price),
    ("05_market_impact", S.market_impact),
    ("06_default_rate", S.default_rate),
    ("07_daily_returns", S.daily_returns),
    ("08_price_path", S.price_path),
    ("09_trade_counts", S.trade_counts),
    ("10_credit_spreads", S.credit_spreads),
    ("11_loan_defaults", S.loan_defaults),
]

if __name__ == "__main__":
    only = set(sys.argv[1:])
    for name, make in CASES:
        if only and not any(o in name for o in only):
            continue
        rng = np.random.default_rng(7)
        out = make(rng)
        if not isinstance(out, tuple):
            args, kw = (out,), {}
        elif len(out) == 3:
            args, kw = out[:2], {"trials": out[2]}
        else:
            args, kw = out, {}
        start = time.time()
        report = analyze(*args, **kw)
        elapsed = time.time() - start
        text = report.summary(show=False)
        (OUT / f"{name}.txt").write_text(text + f"\n\n（耗时 {elapsed:.1f} 秒）\n", encoding="utf-8")
        report.plot(OUT / f"{name}.png")
        if "returns" in report.extras:
            report.extras["returns"].plot(OUT / f"{name}_returns.png")
        if "returns" in report.extras:
            verdict = f"随机游走路径 → 收益率分布：{report.extras['returns'].recommended.label}"
        else:
            verdict = f"推荐：{report.recommended.label}"
        print(f"{name:28s} {elapsed:5.1f}s  {verdict}")
