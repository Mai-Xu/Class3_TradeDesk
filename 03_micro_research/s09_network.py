# -*- coding: utf-8 -*-
"""Cross-stock lead-lag network: correlation structure, sector blocks, lead scores."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib import ROOT, savefig, savetab, name, SECTOR

panel = pd.read_parquet(os.path.join(ROOT, "out", "snap_panel.parquet"),
                        columns=["Symbol", "sec", "mid"])
panel["minute"] = (panel.sec // 60).astype(int)
last = panel.sort_values("sec").groupby(["Symbol", "minute"]).mid.last().reset_index()
wide = last.pivot(index="minute", columns="Symbol", values="mid")
R = np.log(wide).diff()

# keep liquid continuous minutes only
valid = wide.index >= int(9.5 * 3600 // 60)
R = R[valid]
R = R.dropna(how="all")
syms = list(R.columns)
N = len(syms)

# standardize
Z = (R - R.mean()) / R.std()
Z = Z.dropna()
C = Z.corr().values  # contemporaneous correlation

# lead-lag at +1 min
Zv = Z.values
def leadlag(Zv, k):
    A = Zv[:-k]
    B = Zv[k:]
    A = (A - A.mean(0)) / A.std(0)
    B = (B - B.mean(0)) / B.std(0)
    return (A.T @ B) / len(A)   # corr(r_i(t), r_j(t+k)); row i leads col j

C1 = leadlag(Zv, 1)
D = C1 - C1.T
lead = pd.DataFrame(dict(Symbol=syms, Name=[name(s) for s in syms],
                         leadscore=D.mean(1) * 1e4,
                         beta_to_pc1=np.nan))
savetab(lead.sort_values("leadscore", ascending=False), "09_leadscore.csv")

# market factor (equal-weight) and stock vs market lead-lag
mkt = Z.mean(1)
ml = []
for s in syms:
    x = Z[s].values
    y = mkt.values
    lags = {}
    for k in range(-5, 6):
        if k < 0:
            a, b = x[-k:], y[:k]
        elif k > 0:
            a, b = x[:-k], y[k:]
        else:
            a, b = x, y
        lags[k] = np.corrcoef(a, b)[0, 1]
    best = max(lags, key=lambda kk: lags[kk])
    ml.append(dict(Symbol=s, Name=name(s), best_lag_vs_mkt=best,
                   corr_lag0=lags[0], corr_lead=lags[1], corr_lag1=lags[-1]))
ML = pd.DataFrame(ml)
savetab(ML.sort_values("best_lag_vs_mkt"), "09_vs_market.csv")
print("=== 与等权市场因子领先/滞后(分钟) ===")
print(ML.best_lag_vs_mkt.value_counts().sort_index().to_string())
print("\n领先市场(股票领先市场):")
print(ML[ML.best_lag_vs_mkt > 0].sort_values("best_lag_vs_mkt", ascending=False).head(12).to_string(index=False))
print("\n滞后市场(市场领先股票):")
print(ML[ML.best_lag_vs_mkt < 0].sort_values("best_lag_vs_mkt").head(12).to_string(index=False))

# sector blocks
secs = sorted(set(SECTOR.get(s, "其他") for s in syms))
SEC = np.array([[np.nanmean([C[syms.index(a), syms.index(b)] for a in syms if SECTOR.get(a, "其他") == s1
                            for b in syms if SECTOR.get(b, "其他") == s2])
                 for s2 in secs] for s1 in secs])
SECdf = pd.DataFrame(SEC, index=secs, columns=secs)
savetab(SECdf, "09_sector_corr.csv", index=True)
print("\n=== 板块平均相关(分钟收益) ===")
print(SECdf.round(2).to_string())

# figure
order = sorted(range(N), key=lambda i: (SECTOR.get(syms[i], "其他"), syms[i]))
Cord = C[np.ix_(order, order)]
lab = [name(syms[i]) for i in order]
fig = plt.figure(figsize=(16, 6.2))
ax0 = fig.add_axes([0.05, 0.06, 0.34, 0.84])
im = ax0.imshow(Cord, cmap="RdBu_r", vmin=-0.2, vmax=0.8)
ax0.set_xticks(range(N)); ax0.set_yticks(range(N))
ax0.set_xticklabels(lab, rotation=90, fontsize=6); ax0.set_yticklabels(lab, fontsize=6)
ax0.set_title("同期相关矩阵(按板块排序)")
plt.colorbar(im, ax=ax0, fraction=0.046)

ax1 = fig.add_axes([0.44, 0.55, 0.22, 0.35])
im1 = ax1.imshow(SECdf.values, cmap="RdBu_r", vmin=-0.2, vmax=0.8)
ax1.set_xticks(range(len(secs))); ax1.set_yticks(range(len(secs)))
ax1.set_xticklabels(secs, rotation=90, fontsize=7); ax1.set_yticklabels(secs, fontsize=7)
ax1.set_title("板块平均相关")
plt.colorbar(im1, ax=ax1, fraction=0.046)

ax2 = fig.add_axes([0.72, 0.1, 0.25, 0.8])
ls = lead.sort_values("leadscore")
ax2.barh(range(N), ls.leadscore, color=np.where(ls.leadscore > 0, "#2c7fb8", "#de2d26"))
ax2.set_yticks(range(N)); ax2.set_yticklabels(ls.Name, fontsize=6)
ax2.axvline(0, color="k", lw=0.6)
ax2.set_xlabel("领先得分 (x1e-4)")
ax2.set_title("谁领先谁(1 分钟)")
fig.suptitle("图8  跨股票联动网络", fontsize=13)
savefig(fig, "09_network.png")
print("\nsaved 09_network.png + tables")
