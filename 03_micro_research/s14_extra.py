# -*- coding: utf-8 -*-
"""Supplementary: realized spread, Kyle lambda, cancel vs volatility, order splitting."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib import ROOT, load_exe, load_ord, to_sec, name, savefig, savetab

panel = pd.read_parquet(os.path.join(ROOT, "out", "snap_panel.parquet"))
panel = panel.sort_values(["Symbol", "sec"])
S = {s: (g.sec.values.astype(float), g.mid.values) for s, g in panel.groupby("Symbol")}
exe = load_exe(["Symbol", "Time", "Price", "Volume", "BSFlag"])
exe["sec"] = to_sec(exe.Time)
exe = exe[(exe.BSFlag != 0) & (exe.sec >= 9.5 * 3600) & (exe.sec <= 14 * 3600 + 57 * 60)].copy()
exe["sign"] = np.where(exe.BSFlag == 66, 1.0, -1.0)

# ---- realized spread decomposition (5 min) ----
rows = []
for s, g in exe.groupby("Symbol"):
    secs, mids = S[s]
    t = g.sec.values
    sgn = g.sign.values
    bi = np.clip(np.searchsorted(secs, t - 1, side="left") - 1, 0, len(secs) - 1)
    pi_ = np.clip(np.searchsorted(secs, t + 300), 0, len(secs) - 1)
    m0, m1 = mids[bi], mids[pi_]
    px = g.Price.values
    eff = 2 * sgn * (px - m0) / m0
    imp = 2 * sgn * (m1 - m0) / m0
    rea = eff - imp
    rows.append(dict(Symbol=s, Name=name(s), Eff_bp=eff.mean() * 1e4,
                     Real_bp=rea.mean() * 1e4, Impact_bp=imp.mean() * 1e4))
RS = pd.DataFrame(rows)
savetab(RS, "14_spread_decomp.csv")
print("=== 价差分解 (5 分钟实现价差, bp) ===")
print(RS[["Eff_bp", "Real_bp", "Impact_bp"]].describe().to_string())

# ---- Kyle lambda (1-min, bp per million shares) ----
lam = []
for s, g in exe.groupby("Symbol"):
    g = g.assign(minute=(g.sec // 60).astype(int))
    agg = g.groupby("minute").agg(sv=("sign", lambda x: 0), v=("Volume", "sum")).reset_index()
    agg["sv"] = g.groupby("minute").apply(lambda x: (x.sign * x.Volume).sum(), include_groups=False).values
    secs, mids = S[s]
    mm = np.clip(np.searchsorted(secs, agg.minute.values * 60 + 30), 0, len(secs) - 1)
    p = np.log(mids[mm])
    r = (p - np.roll(p, 1))[1:]
    x = agg.sv.values[1:] / 1e6
    m = np.isfinite(r) & np.isfinite(x)
    if m.sum() > 30:
        b = np.polyfit(x[m], r[m] * 1e4, 1)[0]
        lam.append(dict(Symbol=s, Name=name(s), KyleLambda_bp_per_Mshare=b))
L = pd.DataFrame(lam)
savetab(L, "14_kyle_lambda.csv")
print("\n=== Kyle λ (bp / 百万股) ===")
print(L.KyleLambda_bp_per_Mshare.describe().to_string())

# ---- cancel rate vs volatility regime ----
pv = panel.copy()
pv["smin"] = np.where(pv.sec <= 11.5 * 3600, (pv.sec - 9.5 * 3600) / 60,
                      120 + (pv.sec - 13 * 3600) / 60)
pv = pv[(pv.smin >= 0) & (pv.smin <= 237)]
pv["b5"] = (pv.smin // 5).astype(int)
rv = pv.groupby(["Symbol", "b5"]).apply(
    lambda g: pd.Series(dict(rv=np.sqrt(np.nansum(g.ret.values ** 2)))), include_groups=False).reset_index()

o = load_ord(["Symbol", "Time", "FunctionCode", "OrderKind", "OrderOriNo", "Price", "Volume", "Channel"])
o["sec"] = to_sec(o.Time)
new = o[(o.OrderKind == 65) & (o.Price > 0) & (o.sec >= 9.5 * 3600) & (o.sec <= 14 * 3600 + 57 * 60)].copy()
new["smin"] = np.where(new.sec <= 11.5 * 3600, (new.sec - 9.5 * 3600) / 60,
                       120 + (new.sec - 13 * 3600) / 60)
new["b5"] = (new.smin // 5).astype(int)
key = ["Symbol", "Channel", "OrderOriNo"]
cxl = o[o.OrderKind == 68][key].drop_duplicates()
new = new.merge(cxl.assign(canceled=1), on=key, how="left")
new["canceled"] = new.canceled.fillna(0)
new = new.merge(rv, on=["Symbol", "b5"], how="left")
new["rvq"] = new.groupby("Symbol").rv.transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False))
crv = new.groupby("rvq").agg(cancel_rate=("canceled", "mean"),
                             n_orders=("canceled", "size")).reset_index()
print("\n=== 撤单率 vs 波动区间(每股票 5 分位) ===")
print(crv.to_string(index=False))

# ---- order-flow persistence (splitting proxy) ----
persist = []
for s, g in new.sort_values("sec").groupby("Symbol"):
    side = np.where(g.FunctionCode.values == 66, 1, -1)
    same = (side[1:] == side[:-1]).mean()
    px = g.Price.values
    same_px = np.mean((side[1:] == side[:-1]) & (px[1:] == px[:-1]))
    persist.append(dict(Symbol=s, Name=name(s), SameSideProb=same, SameSideSamePrice=same_px))
P = pd.DataFrame(persist)
savetab(P, "14_order_persistence.csv")
print("\n=== 委托方向持续性(拆单代理) ===")
print(P[["SameSideProb", "SameSideSamePrice"]].describe().to_string())

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ax[0].scatter(RS.Eff_bp, RS.Real_bp, s=24, c="#2c7fb8")
ax[0].set_xlabel("有效价差 (bp)"); ax[0].set_ylabel("实现价差 (bp)")
ax[0].set_title("有效价差 vs 实现价差")
ax[0].grid(alpha=0.25)
ax[1].bar(crv.rvq, crv.cancel_rate * 100, color="#de2d26")
ax[1].set_xlabel("波动分位 (0=最低)"); ax[1].set_ylabel("撤单率 (%)")
ax[1].set_title("波动越高,撤单越多吗?")
ax[1].grid(alpha=0.25)
ax[2].hist(P.SameSideProb * 100, bins=20, color="#238b45")
ax[2].axvline(50, color="k", ls="--", lw=1)
ax[2].set_xlabel("相邻委托同向概率 (%)"); ax[2].set_ylabel("股票数")
ax[2].set_title("委托方向持续性(拆单)")
fig.suptitle("图12  价差分解、撤单-波动与拆单持续性", fontsize=13)
savefig(fig, "14_extra.png")
print("\nsaved 14_extra.png + tables")
