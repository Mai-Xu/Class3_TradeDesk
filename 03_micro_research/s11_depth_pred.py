# -*- coding: utf-8 -*-
"""Does book proximity strengthen predictability? level-1 vs level-10 pressure."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
from lib import ROOT, savefig, savetab

p = pd.read_parquet(os.path.join(ROOT, "out", "snap_panel.parquet"))
p = p.sort_values(["Symbol", "sec"]).reset_index(drop=True)
p["dimb1"] = p.groupby("Symbol").imbal1.diff()
p["dimb10"] = p.groupby("Symbol").imbal10.diff()
p["midret"] = p.groupby("Symbol").mid.transform(lambda x: np.log(x) - np.log(x).shift(1))
for h in [1, 3, 10, 20]:
    p[f"f{h}"] = p.groupby("Symbol").mid.transform(lambda x: np.log(x).shift(-h) - np.log(x))


def ols(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 50:
        return np.nan, np.nan
    X = np.column_stack([np.ones_like(x), x])
    b, *_ = lstsq(X, y, rcond=None)
    res = y - X @ b
    return b[1], 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()


# z-score predictors within symbol
for c in ["dimb1", "dimb10", "imbal1", "imbal10"]:
    p[c + "_z"] = p.groupby("Symbol")[c].transform(lambda x: (x - x.mean()) / x.std())

rows = []
for feat in ["dimb1_z", "dimb10_z", "imbal1_z", "imbal10_z"]:
    for h in [1, 3, 10, 20]:
        bs, r2s = [], []
        for s, g in p.groupby("Symbol"):
            b, r2 = ols(g[feat].values, g[f"f{h}"].values)
            if np.isfinite(r2):
                bs.append(b); r2s.append(r2)
        rows.append(dict(feature=feat, horizon_s=h * 3, beta=np.mean(bs), r2_mean=np.mean(r2s)))
D = pd.DataFrame(rows)
savetab(D, "11_depth_predict.csv")
print(D.pivot(index="feature", columns="horizon_s", values="r2_mean").round(5).to_string())

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for feat, col, nm in [("dimb1_z", "#2c7fb8", "1 档失衡变化"),
                      ("dimb10_z", "#de2d26", "10 档失衡变化")]:
    d = D[D.feature == feat]
    ax[0].plot(d.horizon_s, d.r2_mean * 100, "-o", color=col, label=nm)
ax[0].set_xscale("log")
ax[0].set_xlabel("预测视野 (秒)")
ax[0].set_ylabel("R² (%)")
ax[0].set_title("盘口压力对短期收益的预测")
ax[0].legend(); ax[0].grid(alpha=0.25)

for feat, col, nm in [("imbal1_z", "#2c7fb8", "1 档失衡(水平)"),
                      ("imbal10_z", "#de2d26", "10 档失衡(水平)")]:
    d = D[D.feature == feat]
    ax[1].plot(d.horizon_s, d.r2_mean * 100, "-o", color=col, label=nm)
ax[1].set_xscale("log")
ax[1].set_xlabel("预测视野 (秒)")
ax[1].set_title("盘口失衡水平对短期收益的预测")
ax[1].legend(); ax[1].grid(alpha=0.25)
fig.suptitle("图10  越接近盘口,预测越强?", fontsize=13)
savefig(fig, "11_depth_pred.png")
print("\nsaved 11_depth_pred.png")
