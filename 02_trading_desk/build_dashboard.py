"""交易台可视化数据构建 (trading-desk dashboard builder)
=========================================================
复用 exchange_sim 的订单簿引擎，把 逐笔委托(ord) + 逐笔成交(exe) 回放成
一条"交易台"时间轴，并与交易所 3 秒快照逐档核对，最后渲染出单文件 HTML。

产出:
    market_data.js          —— 中间数据(可直接被其它页面复用)
    ../交易台可视化.html     —— 自包含交易台页面(数据已内联)

用法:
    python build_dashboard.py                 # 默认股票池 + 全市场核对
    python build_dashboard.py 600519 600030   # 指定股票池(仍核对全市场)
"""
from __future__ import annotations

import os
import sys
import json
import time as _time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from engine import OrderBook, BUY, SELL        # noqa: E402
import replay as R                              # noqa: E402
from datadir import resolve as _resolve_data_dir, pick   # noqa: E402

DATA = _resolve_data_dir(sys.argv[1:])
OUT_JS = os.path.join(HERE, "market_data.js")
TEMPLATE = os.path.join(HERE, "trading_desk_template.html")
OUT_HTML = os.path.join(HERE, "交易台可视化.html")
METRICS = os.path.join(HERE, "metrics.csv")

# 连续竞价时段 (HHMMSSmmm)
CONT_OPEN = 93000000
AM_END = 113000000
PM_START = 130000000
CONT_CLOSE = 145700000

# 成交明细 tape 上限(超出则等间隔抽样)，控制 HTML 体积
TAPE_CAP = 12000

# 交易台默认股票池：兼顾"核对极好/中等/较差"，便于观察订单簿重建的边界
POOL = [601658, 601857, 601398, 601899, 601318,
        600036, 600519, 603259, 603986, 688008]

NAMES = {
    600028: "中国石化", 600030: "中信证券", 600031: "三一重工", 600036: "招商银行",
    600050: "中国联通", 600089: "特变电工", 600111: "北方稀土", 600150: "中国船舶",
    600183: "生益科技", 600276: "恒瑞医药", 600309: "万华化学", 600406: "国电南瑞",
    600519: "贵州茅台", 600760: "中航沈飞", 600809: "山西汾酒", 600887: "伊利股份",
    600900: "长江电力", 600930: "华电新能", 601012: "隆基绿能", 601088: "中国神华",
    601127: "赛力斯", 601166: "兴业银行", 601211: "国泰海通", 601288: "农业银行",
    601318: "中国平安", 601328: "交通银行", 601398: "工商银行", 601600: "中国铝业",
    601601: "中国太保", 601628: "中国人寿", 601658: "邮储银行", 601668: "中国建筑",
    601688: "华泰证券", 601728: "中国电信", 601857: "中国石油", 601888: "中国中免",
    601899: "紫金矿业", 601919: "中远海控", 601988: "中国银行", 603019: "中科曙光",
    603259: "药明康德", 603501: "韦尔股份", 603986: "兆易创新", 603993: "洛阳钼业",
    688008: "澜起科技", 688012: "中微公司", 688041: "海光信息", 688111: "金山办公",
    688256: "寒武纪", 688981: "中芯国际",
}

ADD, DELETE = 65, 68
_LN = 10            # 盘口深度档数


def _r2(x) -> float:
    return round(float(x) + 1e-9, 2)


def _cols(df):
    return (df["Time"].to_numpy(np.int64), df["BizIndex"].to_numpy(np.int64))


def prep_real(sp: pd.DataFrame):
    """把交易所快照预编译成 numpy 数组，便于逐档核对。"""
    sub = sp[(sp["Time"] >= CONT_OPEN) & (sp["Time"] <= CONT_CLOSE)].copy()
    sub = sub.sort_values("Time")
    times = sub["Time"].to_numpy(np.int64)
    idx = {int(t): i for i, t in enumerate(times)}
    def stack(prefix):
        return np.stack([sub[f"{prefix}{k}"].to_numpy(np.float64) for k in range(1, _LN + 1)], 1)
    return times, idx, stack("BidPrice"), stack("BidVolume"), \
        stack("AskPrice"), stack("AskVolume"), sub["Price"].to_numpy(np.float64)


def replay_symbol(sym: int, od: pd.DataFrame, ex: pd.DataFrame, sp: pd.DataFrame,
                  capture: bool):
    """回放一只股票：重建订单簿 -> 逐档核对 + (可选)导出交易台序列。"""
    th, th_bi = _cols(od)
    kind = od["OrderKind"].to_numpy(np.int8)
    oid = od["OrderOriNo"].to_numpy(np.int64)
    fc = od["FunctionCode"].to_numpy(np.int8)
    opx = od["Price"].to_numpy(np.float64)
    ovol = od["Volume"].to_numpy(np.int64)
    o = np.lexsort((th_bi, th))
    th, kind, oid, fc, opx, ovol = (a[o] for a in (th, kind, oid, fc, opx, ovol))

    eh, eh_bi = _cols(ex)
    ebs = ex["BSFlag"].to_numpy(np.int8)
    ebid = ex["BidOrder"].to_numpy(np.int64)
    eask = ex["AskOrder"].to_numpy(np.int64)
    evol = ex["Volume"].to_numpy(np.int64)
    epx = ex["Price"].to_numpy(np.float64)
    e = np.lexsort((eh_bi, eh))
    eh, ebs, ebid, eask, evol, epx = (a[e] for a in (eh, ebs, ebid, eask, evol, epx))

    times, idx, Rbp, Rbv, Rap, Rav, Rlast = prep_real(sp)
    n_snap = len(times)

    bk = OrderBook(sym)
    i = j = 0
    n, m = len(th), len(eh)

    best_tot = best_ok = 0
    tot = pm = fm = 0
    last_tot = last_ok = 0

    s_last = np.zeros(n_snap, np.int64)
    s_open = np.zeros(n_snap, np.int64)
    s_high = np.zeros(n_snap, np.int64)
    s_low = np.zeros(n_snap, np.int64)
    s_vol = np.zeros(n_snap, np.int64)
    s_turn = np.zeros(n_snap, np.int64)
    s_nt = np.zeros(n_snap, np.int64)
    if capture:
        s_bp = np.zeros(n_snap * _LN, np.int32)
        s_bv = np.zeros(n_snap * _LN, np.int64)
        s_ap = np.zeros(n_snap * _LN, np.int32)
        s_av = np.zeros(n_snap * _LN, np.int64)
        # 订单体量分类(按成交金额): 0=超大单 >=100万, 1=大单 20-100万,
        #                          2=中单 4-20万, 3=小单 <4万
        s_cb = np.zeros((4, n_snap), np.int64)   # 累计主动买金额(万元)
        s_cs = np.zeros((4, n_snap), np.int64)   # 累计主动卖金额(万元)
        acc_b = [0, 0, 0, 0]
        acc_s = [0, 0, 0, 0]
        tape = []

    for k in range(n_snap):
        st = int(times[k])
        while i < n and th[i] <= st:
            kk = int(kind[i])
            if kk == ADD:
                bk.add(int(oid[i]), int(fc[i]), _r2(opx[i]), int(ovol[i]))
            elif kk == DELETE:
                bk.reduce(int(oid[i]), int(ovol[i]))
            i += 1
        while j < m and eh[j] <= st:
            bs = int(ebs[j])
            p = float(epx[j]); v = int(evol[j])
            if bs == 0:      # 集合竞价成交：双边扣减
                bk.reduce(int(ebid[j]), v)
                bk.reduce(int(eask[j]), v)
            else:            # 连续竞价：只扣挂在簿上的被动方
                bk.reduce(int(eask[j]) if bs == BUY else int(ebid[j]), v)
            bk.cum_vol += v
            bk.cum_turn += int(round(p * v))
            bk.n_trades += 1
            if bk.open_ == 0.0:
                bk.open_ = p
            bk.last = p
            if p > bk.high or bk.high == 0.0:
                bk.high = p
            if p < bk.low or bk.low == 0.0:
                bk.low = p
            if capture:
                tape.append((int(eh[j]), int(round(p * 100)), v,
                             1 if bs == BUY else (-1 if bs == SELL else 0)))
                # 订单体量: 按成交金额(元)分档, 累计到万元
                if CONT_OPEN <= eh[j] <= CONT_CLOSE and bs != 0:
                    amt = p * v
                    ci = 0 if amt >= 1e6 else 1 if amt >= 2e5 else 2 if amt >= 4e4 else 3
                    if bs == BUY:
                        acc_b[ci] += int(round(amt))
                    else:
                        acc_s[ci] += int(round(amt))
            j += 1

        bids = bk.depth(BUY, _LN)
        asks = bk.depth(SELL, _LN)

        # ---- 与交易所快照逐档核对 ----
        if Rbp[k, 0] > 0:
            best_tot += 1
            mb = bids[0][0] if bids else np.nan
            ma = asks[0][0] if asks else np.nan
            if abs(mb - Rbp[k, 0]) < 0.005 and abs(ma - Rap[k, 0]) < 0.005:
                best_ok += 1
            for arr, rp, rv in ((bids, Rbp[k], Rbv[k]), (asks, Rap[k], Rav[k])):
                for d in range(_LN):
                    tot += 1
                    r = rp[d]
                    if r == 0:
                        if d >= len(arr) or arr[d][1] <= 0:
                            pm += 1; fm += 1
                    elif d < len(arr) and abs(arr[d][0] - r) < 0.005:
                        pm += 1
                        if abs(arr[d][1] - rv[d]) < 1:
                            fm += 1
            if Rlast[k] > 0:
                last_tot += 1
                if abs(bk.last - Rlast[k]) < 0.005:
                    last_ok += 1

        # ---- 交易台序列 ----
        s_last[k] = int(round(bk.last * 100))
        s_open[k] = int(round(bk.open_ * 100))
        s_high[k] = int(round(bk.high * 100))
        s_low[k] = int(round(bk.low * 100))
        s_vol[k] = int(bk.cum_vol)
        s_turn[k] = int(bk.cum_turn)
        s_nt[k] = int(bk.n_trades)
        if capture:
            b = k * _LN
            for d in range(_LN):
                if d < len(bids):
                    s_bp[b + d] = int(round(bids[d][0] * 100)); s_bv[b + d] = int(bids[d][1])
                if d < len(asks):
                    s_ap[b + d] = int(round(asks[d][0] * 100)); s_av[b + d] = int(asks[d][1])
            for ci in range(4):
                s_cb[ci, k] = int(round(acc_b[ci] / 1e4))   # 元 -> 万元(仅此一次取整)
                s_cs[ci, k] = int(round(acc_s[ci] / 1e4))

    metrics = {
        "n": best_tot,
        "best": round(best_ok / max(1, best_tot), 4),
        "price": round(pm / max(1, tot), 4),
        "depth": round(fm / max(1, tot), 4),
        "last": round(last_ok / max(1, last_tot), 4),
    }

    series = None
    if capture:
        # tape 等间隔抽样(保留首尾)
        if len(tape) > TAPE_CAP:
            arr = np.asarray(tape, dtype=np.int64)
            sel = np.linspace(0, len(arr) - 1, TAPE_CAP).round().astype(np.int64)
            arr = arr[sel]
        else:
            arr = np.asarray(tape, dtype=np.int64).reshape(-1, 4)
        series = {
            "t": times.tolist(),
            "last": s_last.tolist(), "open": s_open.tolist(),
            "high": s_high.tolist(), "low": s_low.tolist(),
            "vol": s_vol.tolist(), "turn": s_turn.tolist(),
            "nt": s_nt.tolist(),
            "bp": s_bp.tolist(), "bv": s_bv.tolist(),
            "ap": s_ap.tolist(), "av": s_av.tolist(),
            "cb": s_cb.tolist(), "cs": s_cs.tolist(),
            "tt": arr[:, 0].tolist(), "tp": arr[:, 1].tolist(),
            "tv": arr[:, 2].tolist(), "ts": arr[:, 3].tolist(),
        }
    return series, metrics


def auction_metrics(sym, od, ex):
    """开盘(仅 ord) / 收盘(ord+成交回报) 集合竞价与真实成交的核对。"""
    out = {}
    for tag, at in (("open", 92500000), ("close", 150000000)):
        p, gen = R.replay_auction(sym, od, ex, at)
        real = ex[ex["Time"] == at]
        rk = set(zip(real["Time"].to_numpy(np.int64),
                     np.round(real["Price"].to_numpy(np.float64) * 100).astype(np.int64),
                     real["Volume"].to_numpy(np.int64),
                     real["BidOrder"].to_numpy(np.int64),
                     real["AskOrder"].to_numpy(np.int64)))
        gk = set(tuple(map(int, row[:5])) for row in gen)
        rpx = sorted(set(round(float(x), 2) for x in real["Price"].to_numpy()))
        out[tag] = {
            "real": len(rk), "gen": len(gk),
            "px": round(float(p), 2) if p else None,
            "rpx": rpx[0] if len(rpx) == 1 else None,
            "ok": bool(len(rpx) == 1 and p is not None and abs(p - rpx[0]) < 0.005),
            "recall": round(sum(1 for x in rk if x in gk) / max(1, len(rk)), 4),
            "prec": round(sum(1 for x in gk if x in rk) / max(1, len(gk)), 4),
        }
    return out


def render_html(blob: str):
    """把数据内联进模板，生成单文件 HTML。"""
    with open(TEMPLATE, "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("//__MARKET_DATA__", "window.MARKET_DATA = " + blob + ";")
    with open(OUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {OUT_HTML} ({os.path.getsize(OUT_HTML)/1e6:.1f} MB)")


def main(pool=None):
    t0 = _time.time()
    pool = pool or POOL
    print("loading parquet ...", flush=True)
    o = pq.read_table(pick(DATA, "ord")).to_pandas()
    e = pq.read_table(pick(DATA, "exe")).to_pandas()
    s = pq.read_table(pick(DATA, "snp")).to_pandas()
    o_g = dict(tuple(o.groupby("Symbol")))
    e_g = dict(tuple(e.groupby("Symbol")))
    s_g = dict(tuple(s.groupby("Symbol")))
    all_syms = sorted(o_g.keys())
    del o, e, s

    prev_close, series, metrics = {}, {}, {}
    for sym in all_syms:
        pc = s_g[sym]["PreClose"].to_numpy(np.float64)
        pc = pc[pc > 0]
        prev_close[str(sym)] = int(round(float(pc[0]) * 100)) if len(pc) else 0

    k = 0
    for sym in all_syms:
        cap = sym in pool
        ser, met = replay_symbol(sym, o_g[sym], e_g[sym], s_g[sym], cap)
        metrics[str(sym)] = met
        if cap:
            series[str(sym)] = ser
        k += 1
        print(f"[{k}/{len(all_syms)}] {sym}  depth={met['depth']:.3f} "
              f"({_time.time()-t0:.0f}s)", flush=True)

    print("auction reconciliation ...", flush=True)
    auctions = {}
    for sym in pool:
        auctions[str(sym)] = auction_metrics(sym, o_g[sym], e_g[sym])
        print(f"  auc {sym} ok ({_time.time()-t0:.0f}s)", flush=True)

    # 附加: run_reconcile.py 的行情网关时延校正结果(若 metrics.csv 存在)
    if os.path.exists(METRICS):
        try:
            md = pd.read_csv(METRICS, encoding="utf-8-sig")
            need = {"Symbol", "delta_cal", "snap_best_pm", "snap_price_pm", "snap_depth_pm"}
            if need.issubset(md.columns):
                for _, r in md.iterrows():
                    k = str(int(r["Symbol"]))
                    if k in metrics:
                        metrics[k]["cal"] = {
                            "d": int(r["delta_cal"]),
                            "best": round(float(r["snap_best_pm"]), 4),
                            "price": round(float(r["snap_price_pm"]), 4),
                            "depth": round(float(r["snap_depth_pm"]), 4),
                        }
                print(f"  merged latency-calibration from metrics.csv "
                      f"({len(md)} rows)", flush=True)
        except Exception as exc:                     # noqa: BLE001
            print(f"  (skip metrics.csv: {exc})", flush=True)

    payload = {
        "generated": _time.strftime("%Y-%m-%d %H:%M:%S"),
        "date": "2026-09-23",
        "session": {"open": CONT_OPEN, "amEnd": AM_END,
                    "pmStart": PM_START, "close": CONT_CLOSE},
        "pool": [str(x) for x in pool],
        "allSymbols": [str(x) for x in all_syms],
        "names": {str(k): NAMES.get(k, "") for k in all_syms},
        "prevClose": prev_close,
        "metrics": metrics,
        "auctions": auctions,
        "series": series,
    }

    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    with open(OUT_JS, "w", encoding="utf-8") as f:
        f.write("window.MARKET_DATA = " + blob + ";\n")
    print(f"wrote {OUT_JS} ({os.path.getsize(OUT_JS)/1e6:.1f} MB)", flush=True)

    if os.path.exists(TEMPLATE):
        render_html(blob)
    print(f"done in {_time.time()-t0:.0f}s")


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--render-only" in argv:
        raw = open(OUT_JS, "r", encoding="utf-8").read()
        render_html(raw[raw.index("=") + 1:].rstrip().rstrip(";"))
    else:
        args = [int(x) for x in argv if x.isdigit()]
        main(args or None)
