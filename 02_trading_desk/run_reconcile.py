"""交易所体系回放与核对主程序。

用法:
    python run_reconcile.py                 # 全部 50 只
    python run_reconcile.py 601688 600030   # 指定股票

输出:
    metrics.csv    —— 每只股票的细粒度指标
    report.md      —— 人读核对报告 (由 make_report.py 生成)

核对分三块:
  1) 开盘集合竞价 —— 只用逐笔委托驱动集合竞价撮合
  2) 收盘集合竞价 —— 用 (逐笔委托 + 成交回报) 重建的订单簿再做集合竞价
  3) 行情快照     —— 用 (逐笔委托 + 成交回报) 重建订单簿并输出十档快照
"""
from __future__ import annotations

import os
import sys
import time as _time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from replay import reconstruct_book, replay_auction   # noqa: E402
from datadir import resolve as _resolve_data_dir, pick   # noqa: E402

DATA = _resolve_data_dir(sys.argv[1:])
OUT = os.path.dirname(os.path.abspath(__file__))
CONT_OPEN = 93000000
AM_END = 113000000

# 快照与逐笔消息之间的小幅时间偏移(行情网关队列延迟), 用于对齐核对
DELTAS = [0, 150, 300, 500, 700, 1000, 1500]


def load():
    o = pq.read_table(pick(DATA, "ord")).to_pandas()
    e = pq.read_table(pick(DATA, "exe")).to_pandas()
    s = pq.read_table(pick(DATA, "snp")).to_pandas()
    return o, e, s


def _real_keys(real):
    rt = real["Time"].to_numpy(np.int64)
    rp = np.round(real["Price"].to_numpy(np.float64) * 100).astype(np.int64)
    rv = real["Volume"].to_numpy(np.int64)
    rb = real["BidOrder"].to_numpy(np.int64)
    ra = real["AskOrder"].to_numpy(np.int64)
    return [tuple(map(int, x)) for x in zip(rt, rp, rv, rb, ra)]


def compare_auction(gen, real, price):
    rset = set(_real_keys(real))
    gk = [tuple(map(int, row[:5])) for row in gen]
    hit = sum(1 for k in gk if k in rset)
    gset = set(gk)
    real_hit = sum(1 for k in rset if k in gset)
    real_px = sorted(set(round(float(x), 2) for x in real["Price"].to_numpy()))
    ok = len(real_px) == 1 and price is not None and abs(price - real_px[0]) < 0.005
    return {
        "real_n": len(rset), "gen_n": len(gk),
        "price": round(price, 2) if price else None,
        "price_ok": bool(ok),
        "precision": round(hit / max(1, len(gk)), 4),
        "recall": round(real_hit / max(1, len(rset)), 4),
    }


def prep_snp(sp):
    """把 snp 子集预编译为 numpy 数组, 加速循环内比较。"""
    times = sp["Time"].to_numpy(np.int64)
    idx = {int(t): i for i, t in enumerate(times)}
    bp = np.stack([sp[f"BidPrice{k}"].to_numpy(np.float64) for k in range(1, 11)], 1)
    bv = np.stack([sp[f"BidVolume{k}"].to_numpy(np.float64) for k in range(1, 11)], 1)
    ap = np.stack([sp[f"AskPrice{k}"].to_numpy(np.float64) for k in range(1, 11)], 1)
    av = np.stack([sp[f"AskVolume{k}"].to_numpy(np.float64) for k in range(1, 11)], 1)
    return idx, bp, bv, ap, av


def _score_window(book, prep, times):
    """book 的键是 snp 时刻; prep 为同一批 snp 行的预编译数组。"""
    idx, bp, bv, ap, av = prep
    best_n = pm = tot = fm = n = 0
    for t in times:
        i = idx.get(int(t))
        if i is None or bp[i, 0] <= 0:
            continue
        row = book.get(int(t))
        if row is None:
            continue
        n += 1
        mb = row["bids"][0][0] if row["bids"] else np.nan
        ma = row["asks"][0][0] if row["asks"] else np.nan
        if abs(mb - bp[i, 0]) < 0.005 and abs(ma - ap[i, 0]) < 0.005:
            best_n += 1
        for side, arr, rp, rv in (("b", row["bids"], bp[i], bv[i]),
                                  ("a", row["asks"], ap[i], av[i])):
            for k in range(10):
                tot += 1
                r = rp[k]
                if r == 0:
                    if k >= len(arr) or arr[k][1] <= 0:
                        pm += 1
                        fm += 1
                elif k < len(arr) and abs(arr[k][0] - r) < 0.005:
                    pm += 1
                    if abs(arr[k][1] - rv[k]) < 1:
                        fm += 1
    return {"best": best_n / max(1, n), "price": pm / max(1, tot),
            "depth": fm / max(1, tot), "n": n}


def process_symbol(sym, od, ex, sp):
    # ---- 集合竞价 ----
    po, go = replay_auction(sym, od, ex, 92500000)
    pc, gc = replay_auction(sym, od, ex, 150000000)
    ao = compare_auction(go, ex[ex["Time"] == 92500000], po)
    ac = compare_auction(gc, ex[ex["Time"] == 150000000], pc)

    # ---- 行情快照: 一次回放, 在候选对齐时刻上输出, 再评不同 delta ----
    snp_cont = sp[sp["Time"] >= CONT_OPEN]
    base_times = snp_cont["Time"].to_numpy(np.int64)
    all_times = np.unique(np.concatenate([base_times + d for d in DELTAS]))
    book = reconstruct_book(sym, od, ex, all_times)

    am = base_times[base_times < AM_END]
    pm_t = base_times[base_times >= AM_END]
    sp_am = sp[(sp.Time >= CONT_OPEN) & (sp.Time < AM_END)]
    sp_pm = sp[sp.Time >= AM_END]
    p_am, p_pm = prep_snp(sp_am), prep_snp(sp_pm)
    # 用上午的行情校准对齐偏移
    curves = {}
    for d in DELTAS:
        sub = {int(t): book.get(int(t) + d) for t in am}
        curves[d] = _score_window(sub, p_am, am)
    best_delta = max(DELTAS, key=lambda d: curves[d]["depth"])
    # 下午为样本外(未参与校准)评价
    sub_am = {int(t): book.get(int(t) + best_delta) for t in am}
    sub_pm = {int(t): book.get(int(t) + best_delta) for t in pm_t}
    sub_raw = {int(t): book.get(int(t)) for t in base_times}
    r_am = _score_window(sub_am, p_am, am)
    r_pm = _score_window(sub_pm, p_pm, pm_t)
    r_raw = _score_window(sub_raw, prep_snp(snp_cont), base_times)

    return {
        "Symbol": sym,
        "auc_open_real": ao["real_n"], "auc_open_gen": ao["gen_n"],
        "auc_open_px": ao["price"], "auc_open_px_ok": ao["price_ok"],
        "auc_open_recall": ao["recall"], "auc_open_prec": ao["precision"],
        "auc_close_real": ac["real_n"], "auc_close_gen": ac["gen_n"],
        "auc_close_px": ac["price"], "auc_close_px_ok": ac["price_ok"],
        "auc_close_recall": ac["recall"], "auc_close_prec": ac["precision"],
        "delta_raw": 0, "snap_best_raw": round(r_raw["best"], 4),
        "snap_price_raw": round(r_raw["price"], 4), "snap_depth_raw": round(r_raw["depth"], 4),
        "delta_cal": best_delta,
        "snap_best_am": round(r_am["best"], 4), "snap_depth_am": round(r_am["depth"], 4),
        "snap_best_pm": round(r_pm["best"], 4), "snap_price_pm": round(r_pm["price"], 4),
        "snap_depth_pm": round(r_pm["depth"], 4),
        "curve": {d: round(curves[d]["depth"], 4) for d in DELTAS},
    }


def main(symbols=None):
    t0 = _time.time()
    o, e, s = load()
    all_syms = sorted(o["Symbol"].unique().tolist())
    if symbols:
        all_syms = [x for x in all_syms if x in symbols]
    o_g = dict(tuple(o.groupby("Symbol")))
    e_g = dict(tuple(e.groupby("Symbol")))
    s_g = dict(tuple(s.groupby("Symbol")))
    rows = []
    for k, sym in enumerate(all_syms, 1):
        try:
            rows.append(process_symbol(sym, o_g[sym], e_g[sym], s_g[sym]))
        except Exception as ex_:
            print(f"  !! {sym} failed: {ex_}", flush=True)
            rows.append({"Symbol": sym})
        print(f"[{k}/{len(all_syms)}] {sym}  ({_time.time()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.drop(columns=["curve"]).to_csv(os.path.join(OUT, "metrics.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(OUT, "metrics_curve.json"), "w", encoding="utf-8") as f:
        import json
        json.dump({int(r["Symbol"]): r["curve"] for r in rows if "curve" in r}, f, ensure_ascii=False, indent=1)
    return df, _time.time() - t0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    syms = [int(x) for x in args if x.isdigit()] or None
    df, dt = main(syms)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    print(df.drop(columns=["curve"], errors="ignore").to_string(index=False))
    print(f"\nelapsed {dt:.0f}s")
