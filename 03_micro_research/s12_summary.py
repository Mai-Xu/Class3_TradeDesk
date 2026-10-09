# -*- coding: utf-8 -*-
"""Master cross-sectional table + stylised-facts figure."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib import ROOT, TAB, savefig, savetab, SECTOR

liq = pd.read_csv(os.path.join(TAB, "02_liquidity.csv"))
ords = pd.read_csv(os.path.join(TAB, "07_order_stats.csv"))
sz = pd.read_csv(os.path.join(TAB, "08_trade_sizes.csv"))
auc = pd.read_csv(os.path.join(TAB, "10_auctions.csv"))

M = liq[["Symbol", "Name", "RelSpread_bp", "QuotedSpread_ticks", "EffSpread_bp",
         "DepthAmt_wan", "RV_bp", "Amihud_1e8"]].copy()
M = M.merge(ords[["Symbol", "AggrRate", "CancelRate", "MedianTTC", "FleetingRate"]], on="Symbol", how="left")
M = M.merge(sz[["Symbol", "Top1VolShare", "ActiveBuyShare", "VPIN", "Turnover"]], on="Symbol", how="left")
M = M.merge(auc[["Symbol", "OpenAucAmtShare", "CloseAucAmtShare", "DayRet"]], on="Symbol", how="left")
M["Sector"] = M.Symbol.map(lambda s: SECTOR.get(int(s), "其他"))
M = M.sort_values("RelSpread_bp")
savetab(M, "12_master_table.csv")
print(M.round(3).to_string(index=False))

# stylised facts scatter
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
px = pd.read_csv(os.path.join(TAB, "01_symbol_activity.csv"))[["Symbol", "PreClose"]]
M2 = M.merge(px, on="Symbol")
ax[0].scatter(M2.PreClose, M2.RelSpread_bp, s=26, c="#2c7fb8")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("股价 (元, log)"); ax[0].set_ylabel("相对价差 (bp, log)")
rho = np.corrcoef(np.log(M2.PreClose), np.log(M2.RelSpread_bp))[0, 1]
ax[0].set_title(f"价格 vs 相对价差(相关系数={rho:.2f})")
ax[0].grid(alpha=0.25)

ax[1].scatter(M.RelSpread_bp, M.RV_bp, s=26, c="#de2d26")
ax[1].set_xscale("log")
ax[1].set_xlabel("相对价差 (bp)"); ax[1].set_ylabel("已实现波动 (bp/日)")
rho2 = np.corrcoef(np.log(M.RelSpread_bp), np.log(M.RV_bp))[0, 1]
ax[1].set_title(f"价差 vs 波动({rho2:.2f})")
ax[1].grid(alpha=0.25)

ax[2].scatter(M.DepthAmt_wan, M.RelSpread_bp, s=26, c="#238b45")
ax[2].set_xscale("log"); ax[2].set_yscale("log")
ax[2].set_xlabel("十档挂单金额 (万元)")
ax[2].set_ylabel("相对价差 (bp)")
rho3 = np.corrcoef(np.log(M.DepthAmt_wan), np.log(M.RelSpread_bp))[0, 1]
ax[2].set_title(f"深度 vs 价差({rho3:.2f})")
ax[2].grid(alpha=0.25)
fig.suptitle("图11  横截面风格化事实", fontsize=13)
savefig(fig, "12_crosssection.png")
print("\nsaved 12_crosssection.png + master table")
