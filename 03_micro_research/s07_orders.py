# -*- coding: utf-8 -*-
"""Order behaviour and cancellation tactics."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib import ROOT, load_ord, to_sec, savefig, savetab, name, TICK

o = load_ord(["Symbol", "Channel", "Time", "FunctionCode", "OrderKind",
              "OrderOriNo", "Price", "Volume"])
o["sec"] = to_sec(o.Time)
new = o[(o.OrderKind == 65) & (o.Price > 0) & (o.sec >= 9.5 * 3600) &
        (o.sec <= 14 * 3600 + 57 * 60)].copy()
cxl = o[o.OrderKind == 68].copy()

key = ["Symbol", "Channel", "OrderOriNo"]
nm = new[key + ["sec", "Volume", "Price", "FunctionCode"]].rename(
    columns={"sec": "nsec", "Volume": "nvol", "Price": "npx", "FunctionCode": "nside"})
j = cxl[key + ["sec"]].rename(columns={"sec": "csec"}).merge(nm, on=key, how="left")
j["ttc"] = j.csec - j.nsec
ttc = j.dropna(subset=["nsec"]).groupby(key).ttc.min().rename("ttc")
new = new.merge(ttc, on=key, how="left")
new["canceled"] = new.ttc.notna()

snap = pd.read_parquet(os.path.join(ROOT, "out", "snap_panel.parquet"),
                       columns=["Symbol", "sec", "mid", "BidPrice1", "AskPrice1"])
agg_rows = []
stats_rows = []
for s, g in new.groupby("Symbol"):
    q = snap[snap.Symbol == s].sort_values("sec")[["sec", "mid", "BidPrice1", "AskPrice1"]]
    m = pd.merge_asof(g.sort_values("sec"), q, on="sec", direction="backward")
    m = m.dropna(subset=["mid"])
    side = np.where(m.FunctionCode.values == 66, 1.0, -1.0)
    m["dist_t"] = side * (m.mid.values - m.Price.values) / TICK
    m["aggr"] = ((m.FunctionCode.values == 66) & (m.Price.values >= m.AskPrice1.values)) | \
                ((m.FunctionCode.values == 83) & (m.Price.values <= m.BidPrice1.values))
    agg_rows.append(m[["Symbol", "FunctionCode", "Volume", "dist_t", "aggr", "canceled", "ttc"]]
                    .rename(columns={"Volume": "nvol"}))
    stats_rows.append(dict(Symbol=s, Name=name(s), NewOrders=len(m),
                           AggrRate=m.aggr.mean(), CancelRate=m.canceled.mean(),
                           MedianTTC=m.ttc.median(), FleetingRate=(m.ttc <= 3).mean(),
                           MedianOrderVol=m.Volume.median()))
A = pd.concat(agg_rows, ignore_index=True)
S = pd.DataFrame(stats_rows)
savetab(S, "07_order_stats.csv")

print("per-symbol order behaviour (describe):")
print(S[["AggrRate", "CancelRate", "MedianTTC", "FleetingRate", "MedianOrderVol"]].describe().to_string())
print("\noverall aggressive-order share = %.1f%%" % (A.aggr.mean() * 100))
print("overall cancel rate          = %.1f%%" % (A.canceled.mean() * 100))
print("fleeting (<3s) share of new  = %.2f%%" % ((A.ttc <= 3).mean() * 100))

bins = [-1e9, 0, 1, 2, 5, 10, 20, 50, 1e9]
lbl = ["aggressive", "at-mid", "1tick", "2tick", "3-5tick", "6-10tick", "11-20tick", ">20tick"]
A["db"] = pd.cut(A.dist_t, bins, labels=lbl)
cr = A.groupby("db", observed=True).agg(cancel_rate=("canceled", "mean"),
                                        n=("canceled", "size"),
                                        med_ttc=("ttc", "median"),
                                        shares=("nvol", "sum")).reset_index()
savetab(cr, "07_cancel_by_distance.csv")
print("\ncancel rate vs distance-from-mid:")
print(cr.to_string(index=False))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ax[0].bar(range(len(cr)), cr.cancel_rate * 100, color="#de2d26")
ax[0].set_xticks(range(len(cr)))
ax[0].set_xticklabels(cr.db, rotation=35)
ax[0].set_ylabel("撤单率 (%)")
ax[0].set_title("撤单率 vs 挂单距中间价")

ttcvals = A.ttc.dropna()
ttcvals = ttcvals[ttcvals >= 0]
ax[1].hist(np.log10(ttcvals.clip(lower=0.1)), bins=60, color="#2c7fb8")
ax[1].set_xlabel("log10(存活时间 秒)")
ax[1].set_ylabel("委托数")
ax[1].set_title("委托存活时间分布")

sz = A.nvol.values
ax[2].hist(np.log10(sz.clip(min=1)), bins=80, color="#31a354")
ax[2].set_xlabel("log10(委托笔数/股)")
ax[2].set_ylabel("委托数")
ax[2].set_title("委托规模分布")
fig.suptitle("图6  委托行为与撤单战术", fontsize=13)
savefig(fig, "07_orders.png")
print("\nsaved 07_orders.png + tables")
