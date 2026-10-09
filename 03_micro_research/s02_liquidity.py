# -*- coding: utf-8 -*-
"""Snapshot panel + per-symbol liquidity metrics."""
import os
import numpy as np
import pandas as pd
from lib import load_snp, load_exe, to_sec, name, savetab, TICK, ROOT

snp = load_snp()
exe = load_exe()
snp["sec"] = to_sec(snp.Time)
exe["sec"] = to_sec(exe.Time)

v = snp[(snp.Price > 0) & (snp.BidPrice1 > 0) & (snp.AskPrice1 > 0)].copy()
v = v.sort_values(["Symbol", "sec"]).reset_index(drop=True)

v["mid"] = (v.BidPrice1 + v.AskPrice1) / 2.0
v["spread"] = v.AskPrice1 - v.BidPrice1
v["rel_spread"] = v.spread / v.mid

bvol1 = v.BidVolume1
avol1 = v.AskVolume1
v["imbal1"] = (bvol1 - avol1) / (bvol1 + avol1)
v["microprice"] = v.AskPrice1 * bvol1 / (bvol1 + avol1) + v.BidPrice1 * avol1 / (bvol1 + avol1)

bidv = [f"BidVolume{i}" for i in range(1, 11)]
askv = [f"AskVolume{i}" for i in range(1, 11)]
bidp = [f"BidPrice{i}" for i in range(1, 11)]
askp = [f"AskPrice{i}" for i in range(1, 11)]
v["depth_bid"] = v[bidv].sum(axis=1)
v["depth_ask"] = v[askv].sum(axis=1)
v["depth_tot"] = v.depth_bid + v.depth_ask
v["imbal10"] = (v.depth_bid - v.depth_ask) / v.depth_tot
v["depth_amt"] = (v[bidv].values * v[bidp].values).sum(axis=1) + (v[askv].values * v[askp].values).sum(axis=1)

v["ret"] = v.groupby("Symbol").mid.transform(lambda s: np.log(s).diff())
v["absret"] = v.ret.abs()

keep = ["Symbol", "sec", "Time", "Price", "mid", "microprice", "spread", "rel_spread",
        "imbal1", "imbal10", "depth_bid", "depth_ask", "depth_tot", "depth_amt",
        "ret", "absret", "Volume", "AccVolume", "AccTurnover",
        "BidPrice1", "AskPrice1", "PreClose", "High", "Low", "Open",
        "TotalBidVolume", "TotalAskVolume"]
panel = v[keep].copy()
panel.to_parquet(os.path.join(ROOT, "out", "snap_panel.parquet"))
print("panel rows", len(panel))


def realized_vol(x):
    return np.sqrt(np.nansum(x ** 2))


rows = []
for s, g in panel.groupby("Symbol"):
    g = g.sort_values("sec")
    dt = g.sec.diff().shift(-1).fillna(0).clip(lower=0)
    tw_rs = np.nansum(g.rel_spread.values * dt.values) / max(dt.sum(), 1)
    damt = g.AccTurnover.diff().clip(lower=0)
    amihud = np.nanmean(g.absret.values / (damt.values / 1e8 + 1e-9)) if damt.sum() > 0 else np.nan
    rv = realized_vol(g.ret.values)
    rows.append(dict(Symbol=s, Name=name(s), Snapshots=len(g),
                     QuotedSpread_ticks=float(np.nanmean(g.spread) / TICK),
                     RelSpread_bp=tw_rs * 1e4,
                     Depth10_lots=float(np.nanmedian(g.depth_tot) / 100),
                     DepthAmt_wan=float(np.nanmedian(g.depth_amt) / 1e4),
                     Imbal10_std=float(np.nanstd(g.imbal10)),
                     RV_bp=rv * 1e4,
                     Amihud_1e8=amihud))
liq = pd.DataFrame(rows).sort_values("RelSpread_bp")

# trade-based effective spread
exe_v = exe[(exe.BSFlag != 0) & (exe.sec >= 9.5 * 3600)].copy()
eff_rows = []
for s, g in exe_v.groupby("Symbol"):
    q = panel[panel.Symbol == s][["sec", "mid"]].sort_values("sec")
    gg = g.sort_values("sec")
    m = pd.merge_asof(gg[["sec", "Price", "Volume", "BSFlag"]], q, on="sec", direction="backward")
    m = m.dropna(subset=["mid"])
    sign = np.where(m.BSFlag.values == 66, 1.0, -1.0)
    eff = 2.0 * sign * (m.Price.values - m.mid.values) / m.mid.values
    eff_rows.append(dict(Symbol=s,
                         EffSpread_bp=float(np.nanmean(eff) * 1e4),
                         EffSpread_ticks=float(np.nanmean(2.0 * sign * (m.Price.values - m.mid.values)) / TICK),
                         TradeCount=int(len(m))))
eff = pd.DataFrame(eff_rows)
liq = liq.merge(eff, on="Symbol", how="left")
savetab(liq, "02_liquidity.csv")
print(liq.to_string(index=False))
print("\nsaved panel + 02_liquidity.csv")
