# -*- coding: utf-8 -*-
"""Special events: opening/closing auctions, price limits, snapshot-trade consistency."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib import ROOT, load_exe, load_snp, to_sec, savefig, savetab, name

exe = load_exe(["Symbol", "Time", "Price", "Volume", "BSFlag"])
snp = load_snp()
exe["sec"] = to_sec(exe.Time)
snp["sec"] = to_sec(snp.Time)

OAU_END = 9 * 3600 + 25 * 60 + 30   # 09:25:30 opening auction done
CAU_START = 14 * 3600 + 57 * 60     # 14:57 closing auction begins
CLOSE = 15 * 3600

rows = []
for s, g in exe.groupby("Symbol"):
    tot = g.Volume.sum()
    oau = g[(g.sec <= OAU_END) & (g.BSFlag == 0)]
    cau = g[(g.sec >= CAU_START) & (g.sec <= CLOSE) & (g.BSFlag == 0)]
    g2 = g[(g.sec > OAU_END) & (g.sec < CAU_START)]
    sn = snp[snp.Symbol == s]
    pc = sn.PreClose.replace(0, np.nan).dropna()
    pc = pc.iloc[0] if len(pc) else np.nan
    hi = sn.High.max()
    lo = sn.Low[sn.Low > 0].min() if (sn.Low > 0).any() else np.nan
    up = 1.2 if str(s).startswith("688") else 1.1
    rows.append(dict(Symbol=s, Name=name(s),
                     DayVol=tot,
                     OpenAucVolShare=oau.Volume.sum() / tot,
                     CloseAucVolShare=cau.Volume.sum() / tot,
                     OpenAucAmtShare=(oau.Price * oau.Volume).sum() / (g.Price * g.Volume).sum(),
                     CloseAucAmtShare=(cau.Price * cau.Volume).sum() / (g.Price * g.Volume).sum(),
                     DayRet=(sn.Price[sn.Price > 0].iloc[-1] / pc - 1) * 100,
                     HiLimPct=(hi / pc - 1) * 100, LoLimPct=(lo / pc - 1) * 100,
                     UpLimitPct=up * 100))
SP = pd.DataFrame(rows).sort_values("CloseAucAmtShare", ascending=False)
savetab(SP, "10_auctions.cssv".replace(".cssv", ".csv"))
print("=== 集合竞价占全日成交额比例 ===")
print(SP[["OpenAucAmtShare", "CloseAucAmtShare"]].describe().to_string())
print("\n收盘竞价占比最高的 8 只:")
print(SP.head(8)[["Symbol", "Name", "CloseAucAmtShare", "OpenAucAmtShare", "DayRet"]].to_string(index=False))
print("\n开盘竞价占比最高的 8 只:")
print(SP.sort_values("OpenAucAmtShare", ascending=False).head(8)[["Symbol", "Name", "OpenAucAmtShare", "DayRet"]].to_string(index=False))

# price-limit proximity
SP["near_up"] = SP.HiLimPct >= 0.98 * SP.UpLimitPct - 0.5
print("\n触及/接近涨停的股票:", SP[SP.near_up][["Symbol", "Name", "HiLimPct", "UpLimitPct"]].to_string(index=False) if SP.near_up.any() else "无")
print("日内最大涨幅 Top5:")
print(SP.sort_values("HiLimPct", ascending=False).head(5)[["Symbol", "Name", "HiLimPct", "LoLimPct"]].to_string(index=False))

# snapshot vs trade-consistency: AccVolume increment per 3s vs summed trade volume
sn = snp[(snp.Price > 0)].sort_values(["Symbol", "sec"]).copy()
sn["sid"] = sn.groupby("Symbol").cumcount()
sn["dV"] = sn.groupby("Symbol").AccVolume.diff()
sn["dT"] = sn.groupby("Symbol").AccTurnover.diff()
# assign trades to snapshot boundaries (forward)
chk = []
for s, g in sn.groupby("Symbol"):
    gg = exe[exe.Symbol == s].sort_values("sec")
    q = g[["sec", "sid"]].copy()
    m = pd.merge_asof(gg[["sec", "Volume", "Price"]], q, on="sec", direction="forward")
    agg = m.groupby("sid").Volume.sum()
    dV = g.set_index("sid").dV
    common = agg.index.intersection(dV.dropna().index)
    av, dv = agg.loc[common], dV.loc[common]
    err = (av - dv) / dv.replace(0, np.nan)
    chk.append(dict(Symbol=s, Name=name(s), intervals=len(common),
                    exact=(np.isclose(av, dv)).mean(),
                    within1pct=(err.abs() < 0.01).mean(),
                    med_relerr=err.abs().median(),
                    final_AccV=float(g.AccVolume.iloc[-1]),
                    sum_trades=gg.Volume.sum()))
CH = pd.DataFrame(chk).sort_values("within1pct")
CH["total_ratio"] = CH.final_AccV / CH.sum_trades
savetab(CH, "10_snapshot_trade_consistency.csv")
print("\n=== 全日累计成交量一致性(快照收盘累计 / 逐笔成交求和) ===")
print(CH.total_ratio.describe().to_string())
print("区间级一致率(受 3s 网关对齐影响) mean exact = %.1f%%, within1%% = %.1f%%" %
      (CH.exact.mean() * 100, CH.within1pct.mean() * 100))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ax[0].scatter(SP.OpenAucAmtShare * 100, SP.CloseAucAmtShare * 100, s=24, c="#2c7fb8")
ax[0].set_xlabel("开盘竞价成交额占比 (%)")
ax[0].set_ylabel("收盘竞价成交额占比 (%)")
ax[0].set_title("集合竞价参与度")
ax[0].grid(alpha=0.25)

top = SP.head(12).iloc[::-1]
ax[1].barh(range(len(top)), top.CloseAucAmtShare * 100, color="#de2d26")
ax[1].set_yticks(range(len(top)))
ax[1].set_yticklabels(top.Name, fontsize=8)
ax[1].set_xlabel("收盘竞价成交额占比 (%)")
ax[1].set_title("收盘集合竞价最活跃的标的")

ax[2].hist((CH.total_ratio - 1) * 100, bins=20, color="#238b45")
ax[2].set_xlabel("全日成交量偏差 (%)")
ax[2].set_ylabel("股票数")
ax[2].set_title("快照-逐笔总量一致性")
fig.suptitle("图9  集合竞价、极端事件与快照一致性", fontsize=13)
savefig(fig, "10_special.png")
print("\nsaved 10_special.png + tables")
