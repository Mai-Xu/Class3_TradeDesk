# -*- coding: utf-8 -*-
"""Large-trade price impact and recovery: event study + permanent/temporary split."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib import ROOT, load_exe, to_sec, savefig, savetab, name

panel = pd.read_parquet(os.path.join(ROOT, "out", "snap_panel.parquet"))
panel = panel.sort_values(["Symbol", "sec"])
S = {s: (g.sec.values.astype(float), g.mid.values) for s, g in panel.groupby("Symbol")}

exe = load_exe(["Symbol", "Time", "Volume", "Price", "BSFlag"])
exe["sec"] = to_sec(exe.Time)
exe = exe[(exe.BSFlag != 0) & (exe.sec >= 9.5 * 3600) & (exe.sec <= 14 * 3600 + 57 * 60)].copy()
exe["sign"] = np.where(exe.BSFlag == 66, 1.0, -1.0)
exe["amt"] = exe.Price * exe.Volume


# ---- event study on large trades (top 1% by volume within symbol) ----
offsets = [-60, -30, -15, -6, -3, 0, 3, 6, 15, 30, 60, 120, 300]
paths = []
for s, g in exe.groupby("Symbol"):
    g = g.sort_values("sec")
    thr = g.Volume.quantile(0.99)
    big = g[g.Volume >= thr]
    rng = np.random.default_rng(0)
    if len(big) > 500:
        idx = rng.choice(len(big), 500, replace=False)
        big = big.iloc[np.sort(idx)]
    secs, mids = S[s]
    t = big.sec.values
    sgn = big.sign.values
    bi = np.clip(np.searchsorted(secs, t - 1, side="left") - 1, 0, len(secs) - 1)
    base = mids[bi]
    d = {"Symbol": s, "sign": sgn, "vol": big.Volume.values}
    for off in offsets:
        j = np.clip(np.searchsorted(secs, t + off), 0, len(secs) - 1)
        d[f"o{off}"] = sgn * (mids[j] - base) / base
    paths.append(pd.DataFrame(d))
P = pd.concat(paths, ignore_index=True)
savetab(P.head(2000), "05_impact_paths_sample.csv")

mean_path = {off: P[f"o{off}"].mean() * 1e4 for off in offsets}
med_path = {off: P[f"o{off}"].median() * 1e4 for off in offsets}
print("=== 大单成交前后 mid 价格路径 (bp, 已乘主动方向) ===")
for off in offsets:
    print(f"  t{off:+4d}s : mean={mean_path[off]:+.2f}  median={med_path[off]:+.2f}")

# permanent vs temporary at 300s
print(f"\n即时冲击(+3s) = {mean_path[3]:+.2f} bp; 永久冲击(+300s) = {mean_path[300]:+.2f} bp; "
      f"临时(回撤) = {mean_path[3]-mean_path[300]:+.2f} bp")

# ---- impact by trade-size decile ----
dec = []
for s, g in exe.groupby("Symbol"):
    g = g.sort_values("sec").copy()
    g["bucket"] = np.minimum((g.Volume.rank(method="first", pct=True) * 10).astype(int), 9)
    secs, mids = S[s]
    t = g.sec.values
    sgn = g.sign.values
    bi = np.clip(np.searchsorted(secs, t - 1, side="left") - 1, 0, len(secs) - 1)
    base = mids[bi]
    j3 = np.clip(np.searchsorted(secs, t + 3), 0, len(secs) - 1)
    j300 = np.clip(np.searchsorted(secs, t + 300), 0, len(secs) - 1)
    imm = sgn * (mids[j3] - base) / base
    perm = sgn * (mids[j300] - base) / base
    g["imm"] = imm
    g["perm"] = perm
    for b, gb in g.groupby("bucket"):
        dec.append(dict(Symbol=s, bucket=b, median_vol=gb.Volume.median(), n=len(gb),
                        imm=gb.imm.mean() * 1e4, perm=gb.perm.mean() * 1e4))
DEC = pd.DataFrame(dec)
decagg = DEC.groupby("bucket").agg(median_vol=("median_vol", "median"), n=("n", "sum"),
                                   imm=("imm", "mean"), perm=("perm", "mean")).reset_index()
savetab(decagg, "05_impact_by_decile.csv")
print("\n=== 按成交规模分组(全样本中位) ===")
print(decagg.to_string(index=False))

# ---- figure ----
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
ax[0].plot(offsets, [mean_path[o] for o in offsets], "-o", color="#2c7fb8", label="均值")
ax[0].plot(offsets, [med_path[o] for o in offsets], "-s", color="#31a354", label="中位")
ax[0].axvline(0, color="k", lw=0.6, ls=":")
ax[0].axhline(0, color="k", lw=0.6)
ax[0].set_xscale("symlog", linthresh=3)
ax[0].set_xlabel("相对大单成交时点 (秒, symlog)")
ax[0].set_ylabel("累积 mid 收益 (bp)")
ax[0].set_title("大单冲击-恢复路径(事件研究)")
ax[0].legend()
ax[0].grid(alpha=0.25)

ax[1].bar(decagg.bucket, decagg.imm, color="#c51b8a", label="即时冲击 (+3s)")
ax[1].plot(decagg.bucket, decagg.perm, "-o", color="#2c7fb8", label="永久冲击 (+300s)")
ax[1].set_xlabel("成交量十分位 (0=最小, 9=最大)")
ax[1].set_ylabel("冲击 (bp)")
ax[1].set_title("冲击随成交规模的放大")
ax[1].legend()
ax[1].grid(alpha=0.25)
fig.suptitle("图4  大单成交的价格冲击与恢复", fontsize=13)
savefig(fig, "05_impact.png")
print("\nsaved 05_impact.png + tables")
