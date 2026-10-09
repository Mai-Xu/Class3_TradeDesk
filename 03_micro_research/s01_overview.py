# -*- coding: utf-8 -*-
"""Sample overview: per-symbol activity, market clock, data-quality facts."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib import (load_ord, load_exe, load_snp, to_sec, name, label, savefig, savetab, TICK, FIG)

ord = load_ord()
exe = load_exe()
snp = load_snp()

syms = sorted(ord.Symbol.unique())
ord["sec"] = to_sec(ord.Time)
exe["sec"] = to_sec(exe.Time)
snp["sec"] = to_sec(snp.Time)

# ---- per-symbol activity table ----
rows = []
for s in syms:
    o = ord[ord.Symbol == s]
    e = exe[exe.Symbol == s]
    sn = snp[snp.Symbol == s]
    new = o[o.OrderKind == 65]
    cxl = o[o.OrderKind == 68]
    buys = e[e.BSFlag == 66]
    sells = e[e.BSFlag == 83]
    auc = e[e.BSFlag == 0]
    turn = float((e.Price * e.Volume).sum())
    px = float(sn.PreClose.dropna().replace(0, np.nan).dropna().iloc[0]) if len(sn) else np.nan
    last = float(sn[sn.Price > 0].Price.iloc[-1]) if (sn.Price > 0).any() else np.nan
    rows.append(dict(
        Symbol=s, Name=name(s),
        Orders=len(o), NewOrders=len(new), Cancels=len(cxl),
        CancelRate=len(cxl) / max(len(new), 1),
        Trades=len(e), AuctionTrades=len(auc),
        BuyVol=int(buys.Volume.sum()), SellVol=int(sells.Volume.sum()),
        Turnover=turn,
        AvgTradeSize=float(e.Volume.mean()),
        PreClose=px, Last=last,
        RetPct=(last / px - 1) * 100 if px and not np.isnan(last) else np.nan,
        Snapshots=int((sn.Price > 0).sum()),
    ))
tab = pd.DataFrame(rows).sort_values("Turnover", ascending=False)
savetab(tab, "01_symbol_activity.csv")

print("=== sample totals ===")
print(f"orders={len(ord):,}  new={int((ord.OrderKind==65).sum()):,}  cancels={int((ord.OrderKind==68).sum()):,}")
print(f"trades={len(exe):,}  auction_trades={int((exe.BSFlag==0).sum()):,}")
print(f"snapshots={len(snp):,}  valid={int((snp.Price>0).sum()):,}")
print(f"symbols={len(syms)}")
print(f"total turnover = {tab.Turnover.sum()/1e8:,.1f} 亿元")
print("\ntop 5 by turnover:")
print(tab.head(5)[["Symbol", "Name", "Turnover", "Trades", "CancelRate", "RetPct"]].to_string(index=False))

# ---- trade clock: trades per minute across the market ----
def hm2s(h, m):
    return h * 3600 + m * 60

tr = exe[(exe.sec >= hm2s(9, 30)) & (exe.sec <= hm2s(15, 0))].copy()
tr["minute"] = (tr.sec // 60).astype(int)
tpm = tr.groupby("minute").agg(Trades=("Volume", "size"), Volume=("Volume", "sum"),
                               Turnover=("Price", lambda x: 0)).reset_index()
turn = tr.assign(amt=tr.Price * tr.Volume).groupby("minute").amt.sum().reset_index(name="Turnover")
tpm = tpm.drop(columns="Turnover").merge(turn, on="minute")
tpm["hhmm"] = tpm.minute.apply(lambda x: f"{x//60:02d}:{x%60:02d}")
savetab(tpm, "01_trades_per_minute.csv")

# ---- order clock ----
od = ord[(ord.sec >= hm2s(9, 15)) & (ord.sec <= hm2s(15, 0))].copy()
od["minute"] = (od.sec // 60).astype(int)
arr = od[od.OrderKind == 65].groupby("minute").size().reset_index(name="NewOrders")
cxl = od[od.OrderKind == 68].groupby("minute").size().reset_index(name="Cancels")
oc = arr.merge(cxl, on="minute", how="outer").fillna(0)
oc["hhmm"] = oc.minute.apply(lambda x: f"{x//60:02d}:{x%60:02d}")
savetab(oc, "01_orders_per_minute.csv")

# ---- figure: market clock ----
# session minute: contiguous 0..239 across the two trading sessions
def sess_minute(minute):
    if minute <= hm2s(11, 30) // 60:
        return minute - hm2s(9, 30) // 60
    return 120 + (minute - hm2s(13, 0) // 60)

for d in (tpm, oc):
    d["smin"] = d.minute.apply(sess_minute)
ticks = [sess_minute(hm2s(h, m) // 60) for h, m in
         [(9, 30), (10, 0), (10, 30), (11, 0), (11, 30), (13, 0), (13, 30), (14, 0), (14, 30), (15, 0)]]
labels = ["09:30", "10:00", "10:30", "11:00", "11:30", "13:00", "13:30", "14:00", "14:30", "15:00"]
fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
ax[0].bar(tpm.smin, tpm.Volume / 1e6, color="#2c7fb8", width=1.0)
ax[0].set_ylabel("成交量 (百万股)")
ax[0].set_title("图1  全样本日内节律(n=50 只, 2026-09-23)")
ax[1].bar(oc.smin, oc.NewOrders / 1e3, color="#31a354", width=1.0, label="新增委托")
ax[1].bar(oc.smin, -oc.Cancels / 1e3, color="#de2d26", width=1.0, label="撤单")
ax[1].axhline(0, color="k", lw=0.6)
ax[1].set_ylabel("委托 (千笔)")
ax[1].legend(loc="upper right")
ax[2].plot(tpm.smin, tpm.Turnover / 1e8, color="#756bb1", marker="o", ms=2)
ax[2].set_ylabel("成交额 (亿元/分钟)")
ax[2].set_xticks(ticks)
ax[2].set_xticklabels(labels)
savefig(fig, "01_market_clock.png")
print("\nsaved figure 01_market_clock.png")
print("\nfirst/last snapshot times:", snp.Time.min(), snp.Time.max())
