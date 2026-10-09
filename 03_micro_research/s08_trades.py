# -*- coding: utf-8 -*-
"""Trade characteristics: size distribution, VPIN, 1s/10s/60s flow predictability."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib import ROOT, load_exe, to_sec, savefig, savetab, name

exe = load_exe(["Symbol", "Time", "Price", "Volume", "BSFlag"])
exe["sec"] = to_sec(exe.Time)
exe = exe[(exe.BSFlag != 0) & (exe.sec >= 9.5 * 3600) & (exe.sec <= 14 * 3600 + 57 * 60)].copy()
exe["sign"] = np.where(exe.BSFlag == 66, 1.0, -1.0)
exe["sv"] = exe.sign * exe.Volume
exe["amt"] = exe.Price * exe.Volume

# trade size distribution + concentration
rows = []
for s, g in exe.groupby("Symbol"):
    v = g.Volume.sort_values()
    tot = g.Volume.sum()
    top1 = v.iloc[int(len(v) * 0.99):].sum() / tot
    rows.append(dict(Symbol=s, Name=name(s), Trades=len(g),
                     MedianVol=v.median(), MeanVol=g.Volume.mean(),
                     Top1VolShare=top1,
                     ActiveBuyShare=(g.BSFlag == 66).mean(),
                     Turnover=g.amt.sum()))
SZ = pd.DataFrame(rows).sort_values("Turnover", ascending=False)
savetab(SZ, "08_trade_sizes.csv")
print("=== 成交规模与集中度 ===")
print(SZ[["MedianVol", "MeanVol", "Top1VolShare", "ActiveBuyShare"]].describe().to_string())

# VPIN (equal-volume buckets)
vpin_ts = []
for s, g in exe.groupby("Symbol"):
    g = g.sort_values("sec")
    V = g.Volume.sum()
    nb = 50
    bsz = V / nb
    cum = g.Volume.cumsum().values
    bid = np.floor(cum / bsz).astype(int)
    g = g.assign(bucket=bid)
    grp = g.groupby("bucket").agg(buy=("Volume", lambda x: 0)).reset_index()
    buys = g[g.sign > 0].groupby("bucket").Volume.sum()
    sells = g[g.sign < 0].groupby("bucket").Volume.sum()
    b = pd.concat([buys, sells], axis=1).fillna(0)
    b.columns = ["buy", "sell"]
    b = b[b.index < nb]  # drop final partial bucket
    b["vpin"] = (b.buy - b.sell).abs() / (b.buy + b.sell)
    b["sec"] = g.groupby("bucket").sec.median()
    b["Symbol"] = s
    vpin_ts.append(b.reset_index())
V = pd.concat(vpin_ts, ignore_index=True)
vp = V.groupby("Symbol").vpin.mean().rename("VPIN").reset_index()
SZ = SZ.merge(vp, on="Symbol", how="left")
savetab(SZ, "08_trade_sizes.csv")
print("\nVPIN (平均, 50 桶) 描述:")
print(SZ.VPIN.describe().to_string())
V.to_parquet(os.path.join(ROOT, "out", "vpin.parquet"))

# ---- 1s bars -> flow predictability at 1s / 10s / 60s ----
pred = []
for s, g in exe.groupby("Symbol"):
    t0, t1 = 9.5 * 3600, 14 * 3600 + 57 * 60
    b = np.arange(0, t1 - t0).astype(int)
    sec = (g.sec.values - t0).astype(int)
    ok = (sec >= 0) & (sec < len(b))
    d = pd.DataFrame(dict(b=sec[ok], px=g.Price.values[ok], vo=g.Volume.values[ok],
                          sv=g.sv.values[ok], am=g.amt.values[ok]))
    bar = d.groupby("b").agg(px=("px", "last"), sv=("sv", "sum"), vo=("vo", "sum"),
                             am=("am", "sum")).reindex(b)
    bar["px"] = bar.px.ffill().bfill()
    bar["sv"] = bar.sv.fillna(0)
    r = np.log(bar.px)
    for h in [1, 10, 60]:
        fr = r.shift(-h) - r
        m = np.isfinite(bar.sv.values) & np.isfinite(fr.values)
        x, y = bar.sv.values[m], fr.values[m]
        if len(x) > 100:
            sd = x.std()
            if sd > 0:
                b1 = np.polyfit(x, y, 1)[0]
                rho = np.corrcoef(x, y)[0, 1]
                ss = 1 - np.var(y - (b1 * x + (y.mean() - b1 * x.mean()))) / np.var(y)
                pred.append(dict(Symbol=s, h=h, beta=b1, ic=rho, r2=ss, n=len(x)))
PR = pd.DataFrame(pred)
agg = PR.groupby("h").agg(ic=("ic", "mean"), r2=("r2", "mean"), r2_med=("r2", "median"),
                          beta=("beta", "mean")).reset_index()
savetab(agg, "08_flow_predict_1s.csv")
print("\n=== 1 秒净主动量 -> 未来收益(逐股平均) ===")
print(agg.to_string(index=False))

# figure
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ax[0].hist(np.log10(exe.Volume.clip(lower=1)), bins=80, color="#2c7fb8")
ax[0].set_xlabel("log10(成交量/股)")
ax[0].set_ylabel("成交笔数")
ax[0].set_title("成交规模分布")

vv = V.pivot_table(index="bucket", columns="Symbol", values="vpin")
for s in vv.columns[:0]:
    pass
mm = vv.mean(axis=1)
ax[1].plot(mm.index, mm.values, color="#c51b8a")
ax[1].set_xlabel("等量桶 (共 50)")
ax[1].set_ylabel("VPIN")
ax[1].set_title("VPIN 日内路径(横截面均值)")
ax[1].grid(alpha=0.25)

ax[2].bar(range(len(agg)), agg.r2 * 100, color="#238b45")
ax[2].set_xticks(range(len(agg)))
ax[2].set_xticklabels([f"{h}s" for h in agg.h])
ax[2].set_ylabel("R² (%)")
ax[2].set_title("净主动量对收益的解释力")
for i, (r2, ic) in enumerate(zip(agg.r2, agg.ic)):
    ax[2].text(i, r2 * 100 + 0.05, f"IC={ic:.3f}", ha="center")
fig.suptitle("图7  成交特征、VPIN 与订单流预测力", fontsize=13)
savefig(fig, "08_trades.png")
print("\nsaved 08_trades.png + tables")
