# -*- coding: utf-8 -*-
"""Order-book shape: depth at distance, asymmetry, slope, round-number clustering."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib import ROOT, load_snp, load_exe, to_sec, savefig, savetab, TICK

snp = load_snp()
snp["sec"] = to_sec(snp.Time)
v = snp[(snp.Price > 0) & (snp.BidPrice1 > 0) & (snp.AskPrice1 > 0) &
        (snp.sec >= 9.5 * 3600) & (snp.sec <= 14 * 3600 + 57 * 60)].copy()
v = v.sort_values(["Symbol", "sec"])
mid = (v.BidPrice1 + v.AskPrice1) / 2.0

# depth-at-distance: notional (万元) per tick bucket, cross-sectional median
NB = 30
cur_bid = np.zeros((len(v), NB))
cur_ask = np.zeros((len(v), NB))
cur_bid_sh = np.zeros((len(v), NB))
cur_ask_sh = np.zeros((len(v), NB))
mb = mid.values
rowsidx = np.arange(len(v))
for i in range(1, 11):
    bp = v[f"BidPrice{i}"].values
    ap = v[f"AskPrice{i}"].values
    bv = v[f"BidVolume{i}"].values
    av = v[f"AskVolume{i}"].values
    db = np.clip(np.round((mb - bp) / TICK).astype(int), 0, NB - 1)
    da = np.clip(np.round((ap - mb) / TICK).astype(int), 0, NB - 1)
    np.add.at(cur_bid, (rowsidx, db), bp * bv / 1e4)
    np.add.at(cur_ask, (rowsidx, da), ap * av / 1e4)
    np.add.at(cur_bid_sh, (rowsidx, db), bv)
    np.add.at(cur_ask_sh, (rowsidx, da), av)

dcols = [f"d{i}" for i in range(NB)]
cb = pd.DataFrame(cur_bid, columns=dcols)
cb["Symbol"] = v.Symbol.values
ca = pd.DataFrame(cur_ask, columns=dcols)
ca["Symbol"] = v.Symbol.values
cum_bid = cb.groupby("Symbol").mean().cumsum(axis=1)
cum_ask = ca.groupby("Symbol").mean().cumsum(axis=1)
shape = pd.DataFrame({"dist_ticks": np.arange(NB),
                      "bid_wan": cum_bid.median(axis=0).values,
                      "ask_wan": cum_ask.median(axis=0).values})
savetab(shape, "06_depth_at_distance.csv")
print("cum depth (万元, cross-section median) vs distance (ticks)")
print(shape.iloc[[0, 1, 2, 4, 9, 19, 29]].to_string(index=False))

# asymmetry within 1 and 5 ticks
bshare = [f"BidVolume{i}" for i in range(1, 11)]
ashare = [f"AskVolume{i}" for i in range(1, 11)]
v["d1_bid"] = v.BidVolume1.values
v["d1_ask"] = v.AskVolume1.values
v["d5_bid"] = v[bshare[:5]].sum(axis=1).values
v["d5_ask"] = v[ashare[:5]].sum(axis=1).values

imb_rows = []
for s, g in v.groupby("Symbol"):
    imb_rows.append(dict(Symbol=s,
                         imb1=(g.d1_bid - g.d1_ask).mean() / ((g.d1_bid + g.d1_ask).mean() + 1e-9),
                         imb5=(g.d5_bid - g.d5_ask).mean() / ((g.d5_bid + g.d5_ask).mean() + 1e-9),
                         slope_bid=np.polyfit(np.arange(NB), cum_bid.loc[s].values, 1)[0],
                         slope_ask=np.polyfit(np.arange(NB), cum_ask.loc[s].values, 1)[0]))
imb = pd.DataFrame(imb_rows)
savetab(imb, "06_imbalance_slope.csv")
print("\nquote imbalance: mean imb1 = %.4f, mean imb5 = %.4f" % (imb.imb1.mean(), imb.imb5.mean()))

# round-number clustering in trade prices
exe = load_exe(["Symbol", "Time", "Price", "Volume", "BSFlag"])
exe = exe[(exe.BSFlag != 0)]
px = exe.Price.values.astype(float)
res = {}
for gran, nm in [(0.1, "0.1"), (0.5, "0.5"), (1.0, "1.0")]:
    res[nm] = float(np.isclose(np.round(px / gran) * gran, px, atol=1e-4).mean())
cents = np.round(px * 100).astype(int) % 100
tail = pd.Series(cents).value_counts(normalize=True).sort_index()
print("\ntop-10 trade-price cents endings (%):")
print((tail.sort_values(ascending=False).head(10) * 100).round(2).to_string())
print("\nround-number share of trade prices:")
for k, val in res.items():
    print(f"  multiple of {k}: {val*100:.1f}%")
pd.Series(res).to_csv(os.path.join(ROOT, "out", "tables", "06_round_numbers.csv"), encoding="utf-8-sig")
tail.to_csv(os.path.join(ROOT, "out", "tables", "06_price_tails.csv"), encoding="utf-8-sig")

# figure
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ax[0].plot(shape.dist_ticks, shape.bid_wan, "-o", ms=3, color="#2c7fb8", label="买盘")
ax[0].plot(shape.dist_ticks, shape.ask_wan, "-s", ms=3, color="#de2d26", label="卖盘")
ax[0].set_xlabel("距中间价 (跳)")
ax[0].set_ylabel("累积挂单金额 (万元)")
ax[0].set_title("订单簿形态:累积深度")
ax[0].legend()
ax[0].grid(alpha=0.25)
ax[0].set_xlim(0, 20)
ax[1].hist(imb.imb5 * 100, bins=25, color="#756bb1")
ax[1].axvline(0, color="k", lw=0.8)
ax[1].set_xlabel("5 跳内挂单失衡 (%)")
ax[1].set_ylabel("股票数")
ax[1].set_title("盘口深度失衡(50 只)")
tt = (tail.reindex(range(100)).fillna(0) * 100)
ax[2].bar(tt.index, tt.values, color="#238b45")
ax[2].set_xlabel("成交价尾数 (分)")
ax[2].set_ylabel("占比 (%)")
ax[2].set_title("成交价尾数偏好")
fig.suptitle("图5  订单簿形态与价格离散化偏好", fontsize=13)
savefig(fig, "06_orderbook.png")
print("\nsaved 06_orderbook.png + tables")
