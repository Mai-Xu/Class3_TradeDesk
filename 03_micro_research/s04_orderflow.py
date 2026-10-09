# -*- coding: utf-8 -*-
"""Order-flow imbalance (OFI) and short-horizon return predictability."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib import ROOT, SNP_PATH, savefig, savetab, name

panel = pd.read_parquet(os.path.join(ROOT, "out", "snap_panel.parquet"))
panel = panel.sort_values(["Symbol", "sec"]).reset_index(drop=True)


def ofi1(g):
    bp = g.BidPrice1.values
    ap = g.AskPrice1.values
    bv = g.TotalBidVolume.values.astype(float) if "TotalBidVolume" in g else g.BidPrice1.values * 0
    # use level-1 volumes if available
    return g


# Level-1 OFI (Cont, Kukanov, Stoikov 2014) needs level-1 volumes; reload from snp
import pyarrow.parquet as pq
snp = pd.read_parquet(os.path.join(ROOT, "out", "snap_panel.parquet"),
                      columns=["Symbol", "sec", "BidPrice1", "AskPrice1"])
raw = pd.read_parquet(SNP_PATH,
                      columns=["Symbol", "Time", "BidPrice1", "AskPrice1", "BidVolume1", "AskVolume1"])

from lib import to_sec
raw["sec"] = to_sec(raw.Time)
raw = raw.sort_values(["Symbol", "sec"]).reset_index(drop=True)
v = raw[(raw.BidPrice1 > 0) & (raw.AskPrice1 > 0)].copy()

out = []
for s, g in v.groupby("Symbol"):
    g = g.sort_values("sec").reset_index(drop=True)
    bp, ap = g.BidPrice1.values, g.AskPrice1.values
    bv, av = g.BidVolume1.values.astype(float), g.AskVolume1.values.astype(float)
    n = len(g)
    ofi = np.zeros(n)
    # OFI_t from t-1 -> t
    for t in range(1, n):
        d_bid = (bp[t] >= bp[t - 1]) * bv[t] - (bp[t] <= bp[t - 1]) * bv[t - 1]
        d_ask = (ap[t] <= ap[t - 1]) * av[t] - (ap[t] >= ap[t - 1]) * av[t - 1]
        ofi[t] = d_bid - d_ask
    out.append(pd.DataFrame(dict(Symbol=s, sec=g.sec.values, bid=bp, ask=ap,
                                 bvol=bv, avol=av, ofi=ofi, mid=(bp + ap) / 2.0)))
O = pd.concat(out, ignore_index=True)
O = O.sort_values(["Symbol", "sec"]).reset_index(drop=True)

# forward mid returns over snapshot horizons
for h in [1, 3, 10, 20]:
    O[f"fret{h}"] = O.groupby("Symbol").mid.transform(lambda x: np.log(x).shift(-h) - np.log(x))
# contemporaneous return (previous snapshot -> current)
O["cret"] = O.groupby("Symbol").mid.transform(lambda x: np.log(x) - np.log(x).shift(1))

# signed trade flow (aggressor sided) per snapshot interval
from lib import load_exe
exe = load_exe(["Symbol", "Time", "Volume", "BSFlag", "Price"])
exe["sec"] = to_sec(exe.Time)
exe = exe[exe.BSFlag != 0]
exe["sv"] = np.where(exe.BSFlag == 66, exe.Volume, -exe.Volume)
exe["samt"] = np.where(exe.BSFlag == 66, exe.Volume * exe.Price, -exe.Volume * exe.Price)

# assign each trade to the next snapshot boundary within its symbol
O["key"] = O.Symbol.astype("int64") * 1_000_000 + (O.sec * 1000).astype("int64")
snap_idx = O[["Symbol", "sec"]].copy()
snap_idx["sid"] = snap_idx.groupby("Symbol").cumcount()
exe2 = exe.sort_values("sec")
merged = []
for s, g in exe2.groupby("Symbol"):
    q = snap_idx[snap_idx.Symbol == s][["sec", "sid"]].sort_values("sec")
    m = pd.merge_asof(g.sort_values("sec"), q, on="sec", direction="forward")
    merged.append(m)
mflow = pd.concat(merged, ignore_index=True)
flow = mflow.groupby(["Symbol", "sid"]).agg(sv=("sv", "sum"), samt=("samt", "sum"),
                                            ntr=("sv", "size")).reset_index()
O = O.merge(snap_idx.assign(sid=snap_idx.groupby("Symbol").cumcount()), on=["Symbol", "sec"], how="left")
O = O.merge(flow, on=["Symbol", "sid"], how="left")
O[["sv", "samt", "ntr"]] = O[["sv", "samt", "ntr"]].fillna(0)

# normalize within symbol
def z(x):
    sd = x.std()
    return (x - x.mean()) / sd if sd and sd > 0 else x * 0

for c in ["ofi", "sv", "samt"]:
    O[c + "_z"] = O.groupby("Symbol")[c].transform(z)
O = O.dropna(subset=["ofi_z"])

# ---- regressions: predict forward return on OFI (and signed flow) ----
from numpy.linalg import lstsq


def ols(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    if len(x) < 30:
        return np.nan, np.nan, 0
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - (resid ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-30)
    return beta[1], r2, len(x)


def per_symbol_r2(feat, target):
    betas, r2s, ns = [], [], 0
    for s, g in O.groupby("Symbol"):
        b, r2, n = ols(g[feat].values, g[target].values)
        if n:
            betas.append(b); r2s.append(r2); ns += n
    return dict(beta_mean=np.mean(betas), r2_mean=np.mean(r2s),
                r2_median=np.median(r2s), N=ns, n_sym=len(betas))


res = []
for tag, feat in [("OFI(t)", "ofi_z"), ("主动净额(t)", "samt_z")]:
    # contemporaneous (price impact)
    d = per_symbol_r2(feat, "cret"); d.update(feature=tag, kind="同 期(价 格 冲 击)", horizon_s=0); res.append(d)
    for h in [1, 3, 10, 20]:
        d = per_symbol_r2(feat, f"fret{h}"); d.update(feature=tag, kind="预 测", horizon_s=round(h * 3)); res.append(d)
R = pd.DataFrame(res)[["feature", "kind", "horizon_s", "beta_mean", "r2_mean", "r2_median", "N", "n_sym"]]
savetab(R, "04_ofi_regression.csv")
print("=== 逐股回归(均值)· OFI / 主动净额 -> mid 收益 ===")
print(R.to_string(index=False))

# ---- figure: OFI predictive curve + contemporaneous price impact ----
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

# panel 1: R2 by horizon for OFI vs signed amount
for feat, col in [("OFI(t)", "#2c7fb8"), ("主动净额(t)", "#de2d26")]:
    d = R[(R.feature == feat) & (R.kind == "预 测")]
    ax[0].plot(d.horizon_s, d.r2_mean * 100, "-o", color=col, label=feat + " 预测")
    dd = R[(R.feature == feat) & (R.kind == "同 期(价 格 冲 击)")]
    ax[0].axhline(dd.r2_mean.iloc[0] * 100, color=col, ls="--", lw=1,
                  label=feat + " 同期(冲击)")
ax[0].set_xscale("log")
ax[0].set_xlabel("预测视野 (秒, log)")
ax[0].set_ylabel("R² (%)")
ax[0].set_ylim(0, None)
ax[0].set_title("OFI 对中期收益的预测力")
ax[0].legend()
ax[0].grid(alpha=0.25)

# panel 2: contemporaneous impact vs liquidity (cross-section)
full = O.dropna(subset=["ofi_z", "fret1"])
imp = full.groupby("Symbol").apply(
    lambda g: pd.Series(dict(beta=ols(g.ofi_z.values, g.cret.values)[0],
                             r2=ols(g.ofi_z.values, g.cret.values)[1])), include_groups=False).reset_index()
liq = pd.read_csv(os.path.join(ROOT, "out", "tables", "02_liquidity.csv"))
imp = imp.merge(liq[["Symbol", "RelSpread_bp", "DepthAmt_wan"]], on="Symbol")
ax[1].scatter(imp.RelSpread_bp, imp.beta * 1e4, s=24, c="#756bb1")
ax[1].set_xlabel("相对价差 (bp)")
ax[1].set_ylabel("OFI 冲击系数 (bp / z)")
ax[1].set_title("冲击强度 vs 流动性")
ax[1].grid(alpha=0.25)

# panel 3: OFI autocorrelation (mean across symbols)
lags = range(1, 11)
ac = []
for s, g in O.groupby("Symbol"):
    x = g.ofi_z.values
    x = x[np.isfinite(x)]
    for L in lags:
        if len(x) > L:
            ac.append(np.corrcoef(x[:-L], x[L:])[0, 1])
acm = pd.DataFrame(ac, columns=["ac"]).assign(lag=lambda d: np.tile(list(lags), len(O.Symbol.unique())))
acmean = acm.groupby("lag").ac.mean()
ax[2].bar(acmean.index, acmean.values, color="#238b45")
ax[2].set_xlabel("滞后 (快照, 3s)")
ax[2].set_ylabel("OFI 自相关")
ax[2].set_title("OFI 的持续性")
ax[2].grid(alpha=0.25)

fig.suptitle("图3  订单流不平衡(OFI)与短期收益预测", fontsize=13)
savefig(fig, "04_ofi.png")

imp.to_csv(os.path.join(ROOT, "out", "tables", "04_impact_crosssection.csv"), index=False, encoding="utf-8-sig")
print("\nsaved 04_ofi.png + tables")
