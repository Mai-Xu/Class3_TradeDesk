# -*- coding: utf-8 -*-
"""Intraday rhythm: U-shape of spread, volatility, depth, cancel, flow."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib import ROOT, FIG, savefig, savetab

panel = pd.read_parquet(os.path.join(ROOT, "out", "snap_panel.parquet"))

SESS_AM = (9 * 3600 + 30 * 60, 11 * 3600 + 30 * 60)
SESS_PM = (13 * 3600, 14 * 3600 + 57 * 60)


def smin(sec):
    """map seconds-of-day to contiguous session minute 0..240."""
    out = np.full_like(sec, -1, dtype=float)
    am = (sec >= SESS_AM[0]) & (sec <= SESS_AM[1])
    pm = (sec >= SESS_PM[0]) & (sec <= SESS_PM[1])
    out[am] = (sec[am] - SESS_AM[0]) / 60.0
    out[pm] = 120 + (sec[pm] - SESS_PM[0]) / 60.0
    return out


panel["smin"] = smin(panel.sec.values)
p = panel[panel.smin >= 0].copy()
p["bucket"] = (p.smin // 5).astype(int)  # 5-minute buckets

buckets = []
for s, g in p.groupby("Symbol"):
    for b, gb in g.groupby("bucket"):
        rv = np.sqrt(np.nansum(gb.ret.values ** 2))
        buckets.append(dict(Symbol=s, bucket=b, rel_spread=gb.rel_spread.mean(),
                            rv=rv, depth=gb.depth_tot.median(), imbal_std=gb.imbal10.std(),
                            obase=gb.mid.mean(), n=len(gb)))
B = pd.DataFrame(buckets)

# z-score per stock, then cross-sectional average
for col in ["rel_spread", "rv", "depth", "imbal_std"]:
    B[col + "_z"] = B.groupby("Symbol")[col].transform(lambda x: (x - x.mean()) / x.std())

cross = B.groupby("bucket").agg(
    rel_spread=("rel_spread", "median"),
    rv=("rv", "median"),
    depth=("depth", "median"),
    rs_z=("rel_spread_z", "mean"),
    rv_z=("rv_z", "mean"),
    depth_z=("depth_z", "mean"),
    imbal_z=("imbal_std_z", "mean")).reset_index()
cross["t"] = cross.bucket * 5
cross["hhmm"] = cross.t.apply(lambda m: f"{9+(30+m)//60:02d}:{(30+m)%60:02d}" if m <= 120 else f"{13+(m-120)//60:02d}:{(m-120)%60:02d}")
savetab(cross, "03_intraday_curve.csv")

# order/trade intensity per 5-min bucket
from lib import load_exe, load_ord, to_sec
exe = load_exe(["Symbol", "Time", "Volume", "Price", "BSFlag"])
ordf = load_ord(["Symbol", "Time", "OrderKind"])
exe["smin"] = smin(to_sec(exe.Time).values)
ordf["smin"] = smin(to_sec(ordf.Time).values)
exe = exe[exe.smin >= 0]
ordf = ordf[ordf.smin >= 0]
exe["bucket"] = (exe.smin // 5).astype(int)
ordf["bucket"] = (ordf.smin // 5).astype(int)
tr = exe.groupby(["Symbol", "bucket"]).Volume.sum().reset_index(name="vol")
new = ordf[ordf.OrderKind == 65].groupby(["Symbol", "bucket"]).size().reset_index(name="new")
cxl = ordf[ordf.OrderKind == 68].groupby(["Symbol", "bucket"]).size().reset_index(name="cxl")
fl = tr.merge(new, on=["Symbol", "bucket"], how="outer").merge(cxl, on=["Symbol", "bucket"], how="outer").fillna(0)
for c in ["vol", "new", "cxl"]:
    fl[c + "_z"] = fl.groupby("Symbol")[c].transform(lambda x: (x - x.mean()) / x.std())
fc = fl.groupby("bucket").agg(vol_z=("vol_z", "mean"), new_z=("new_z", "mean"),
                              cxl_z=("cxl_z", "mean"), cancel_rate=("cxl", "sum")).reset_index()
fc["t"] = fc.bucket * 5
cross = cross.merge(fc, on="bucket")
cross["t"] = cross.bucket * 5

# ---- figure: universal U-shape ----
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
tt = cross.t

ax[0, 0].plot(tt, cross.rs_z, "-o", ms=3, color="#d95f0e")
ax[0, 0].axhline(0, color="k", lw=0.5)
ax[0, 0].set_title("相对价差(横截面 z 值)")
ax[0, 0].set_ylabel("相对价差 z")

ax[0, 1].plot(tt, cross.rv_z, "-o", ms=3, color="#c51b8a")
ax[0, 1].axhline(0, color="k", lw=0.5)
ax[0, 1].set_title("5 分钟已实现波动率")
ax[0, 1].set_ylabel("波动 z")

ax[1, 0].plot(tt, cross.depth_z, "-o", ms=3, color="#238b45", label="十档深度")
ax[1, 0].plot(tt, cross.vol_z, "-s", ms=3, color="#2c7fb8", label="成交量")
ax[1, 0].axhline(0, color="k", lw=0.5)
ax[1, 0].legend()
ax[1, 0].set_title("深度 / 成交量")
ax[1, 0].set_ylabel("z")

ax[1, 1].plot(tt, cross.new_z, "-o", ms=3, color="#31a354", label="新增委托")
ax[1, 1].plot(tt, cross.cxl_z, "-o", ms=3, color="#de2d26", label="撤单")
ax[1, 1].plot(tt, cross.imbal_z, "-^", ms=3, color="#756bb1", label="盘口失衡波动")
ax[1, 1].axhline(0, color="k", lw=0.5)
ax[1, 1].legend()
ax[1, 1].set_title("委托活动")
ax[1, 1].set_ylabel("z")

ticks = [0, 30, 60, 90, 120, 150, 180, 210, 240]
labels = ["09:30", "10:00", "10:30", "11:00", "11:30", "13:00", "13:30", "14:00", "14:30"]
for a in ax.ravel():
    a.set_xticks(ticks)
    a.set_xticklabels(labels, rotation=30)
    a.grid(alpha=0.25)
fig.suptitle("图2  日内节律:5 分钟桶、按股票标准化后横截面平均(n=50)", fontsize=13)
savefig(fig, "03_intraday_ushape.png")

print(cross.to_string(index=False))
print("\nsaved 03_intraday_curve.csv + figure")
