"""回放与核对 (replay & reconciliation)
=====================================
两种回放：

replay_matching()
    只用「逐笔委托(ord)」驱动撮合引擎，由引擎**主动生成逐笔成交**。
    用于回答："仅凭逐笔委托能不能还原出成交？"

reconstruct_book()
    用「逐笔委托 + 逐笔成交(成交回报)」重建订单簿，**生成 3 秒行情快照**。
    用于回答："订单簿/行情快照能不能被还原？"

差异的根源见 README：上交所逐笔委托只包含"进入订单簿的委托"，
下单即全部成交的委托不会出现在逐笔委托里，因此仅用 ord 无法还原全部成交，
但订单簿(从而行情快照)可以用 ord + 成交回报完整重建。
"""
from __future__ import annotations

import numpy as np

from engine import OrderBook, BUY, SELL, ADD, DELETE

T_OPEN_AUC_END = 92500000
T_CONT_AM = (93000000, 113000000)
T_CONT_PM = (130000000, 145700000)
T_CLOSE_AUC_START = 145700000
T_CLOSE_AUC_END = 150000000


def _ord_arrays(df):
    return (df["Time"].to_numpy(np.int64),
            df["BizIndex"].to_numpy(np.int64),
            df["OrderKind"].to_numpy(np.int8),
            df["OrderOriNo"].to_numpy(np.int64),
            df["FunctionCode"].to_numpy(np.int8),
            df["Price"].to_numpy(np.float32),
            df["Volume"].to_numpy(np.int64))


# ---------------------------------------------------------------------- #
#  路径 1: 仅用逐笔委托 -> 撮合 -> 生成逐笔成交
# ---------------------------------------------------------------------- #
def replay_matching(symbol: int, ord_df) -> np.ndarray:
    """只用逐笔委托驱动撮合引擎，返回生成的成交数组。

    返回列: time, price, volume, bid_oid, ask_oid, aggressor
    """
    t, bi, kind, oid, fc, px, vol = _ord_arrays(ord_df)
    order = np.lexsort((bi, t))
    t, bi, kind, oid, fc, px, vol = (a[order] for a in (t, bi, kind, oid, fc, px, vol))

    bk = OrderBook(symbol)
    state = "AUC_OPEN"
    n = len(t)
    for i in range(n):
        ti = int(t[i])
        if state == "AUC_OPEN" and ti >= T_OPEN_AUC_END:
            bk.auction_clear(T_OPEN_AUC_END)
            state = "QUEUE"
        if state == "QUEUE" and ti >= T_CONT_AM[0]:
            state = "CONT"
        if state == "CONT" and ti >= T_CONT_AM[1]:
            state = "LUNCH"
        if state == "LUNCH" and ti >= T_CONT_PM[0]:
            state = "CONT"
        if state == "CONT" and ti >= T_CLOSE_AUC_START:
            state = "AUC_CLOSE"
        if state == "AUC_CLOSE" and ti >= T_CLOSE_AUC_END:
            bk.auction_clear(T_CLOSE_AUC_END)
            state = "AFTER"

        k = int(kind[i])
        if k == ADD:
            p = round(float(px[i]), 2)
            if state == "CONT":
                bk.match_continuous(int(oid[i]), int(fc[i]), p, int(vol[i]), ti)
            else:
                bk.add(int(oid[i]), int(fc[i]), p, int(vol[i]))
        elif k == DELETE:
            bk.reduce(int(oid[i]), int(vol[i]))

    if state == "AUC_OPEN" or state == "QUEUE":
        bk.auction_clear(T_OPEN_AUC_END)
    elif state == "AUC_CLOSE":
        bk.auction_clear(T_CLOSE_AUC_END)

    g = bk.generated
    if not g:
        return np.empty((0, 6), dtype=np.int64)
    out = np.empty((len(g), 6), dtype=np.int64)
    for i, x in enumerate(g):
        out[i] = (x.time, int(round(x.price * 100)), x.volume, x.bid_oid, x.ask_oid,
                  x.aggressor if x.aggressor else 0)
    return out          # price 以"分"存储, 避免浮点误差


# ---------------------------------------------------------------------- #
#  集合竞价专用回放: 用订单簿 + 最大成交量算法生成集合竞价成交
# ---------------------------------------------------------------------- #
def replay_auction(symbol: int, ord_df, exe_df, at: int):
    """在 `at` 时点执行一次集合竞价撮合。

    * at = 92500000 (开盘):  开盘前没有任何成交，只用逐笔委托即可，结果应与真实完全一致。
    * at = 150000000(收盘): 需要先用 逐笔委托 + 逐笔成交 把 14:57 前的订单簿重建出来，
                             再执行集合竞价撮合。

    返回 (auction_price, 成交数组[time,price_cents,vol,bid_oid,ask_oid,aggressor])
    """
    t, bi, kind, oid, fc, px, vol = _ord_arrays(ord_df)
    order = np.lexsort((bi, t))
    t, kind, oid, fc, px, vol = (a[order] for a in (t, kind, oid, fc, px, vol))
    et = exe_df["Time"].to_numpy(np.int64)
    ebs = exe_df["BSFlag"].to_numpy(np.int8)
    ebid = exe_df["BidOrder"].to_numpy(np.int64)
    eask = exe_df["AskOrder"].to_numpy(np.int64)
    evol = exe_df["Volume"].to_numpy(np.int64)

    bk = OrderBook(symbol)
    for i in range(len(t)):
        if t[i] >= at:
            break
        k = int(kind[i])
        if k == ADD:
            bk.add(int(oid[i]), int(fc[i]), round(float(px[i]), 2), int(vol[i]))
        elif k == DELETE:
            bk.reduce(int(oid[i]), int(vol[i]))
    for j in range(len(et)):
        if et[j] >= at:
            break
        bs = int(ebs[j])
        if bs == 0:
            bk.reduce(int(ebid[j]), int(evol[j]))
            bk.reduce(int(eask[j]), int(evol[j]))
        else:
            resting = int(eask[j]) if bs == BUY else int(ebid[j])
            bk.reduce(resting, int(evol[j]))

    price = bk.auction_price()
    bk.auction_clear(at)
    g = bk.generated
    if not g:
        return price, np.empty((0, 6), dtype=np.int64)
    out = np.empty((len(g), 6), dtype=np.int64)
    for i, x in enumerate(g):
        out[i] = (x.time, int(round(x.price * 100)), x.volume, x.bid_oid, x.ask_oid, 0)
    return price, out


# ---------------------------------------------------------------------- #
#  路径 2: 逐笔委托 + 成交回报 -> 重建订单簿 -> 生成行情快照
# ---------------------------------------------------------------------- #
def reconstruct_book(symbol: int, ord_df, exe_df, snap_times: np.ndarray,
                     continuous_resting_only: bool = True):
    """重建订单簿并在给定时刻输出十档快照。

    返回 dict: 每个快照时刻 -> 快照字段(十档买卖价量、最新价、累计量额)
    """
    t, bi, kind, oid, fc, px, vol = _ord_arrays(ord_df)
    order = np.lexsort((bi, t))
    t, kind, oid, fc, px, vol = (a[order] for a in (t, kind, oid, fc, px, vol))

    et = exe_df["Time"].to_numpy(np.int64)
    ebi = exe_df["BizIndex"].to_numpy(np.int64)
    ebs = exe_df["BSFlag"].to_numpy(np.int8)
    ebid = exe_df["BidOrder"].to_numpy(np.int64)
    eask = exe_df["AskOrder"].to_numpy(np.int64)
    evol = exe_df["Volume"].to_numpy(np.int64)
    epx = exe_df["Price"].to_numpy(np.float64)
    eorder = np.lexsort((ebi, et))
    et, ebs, ebid, eask, evol, epx = (a[eorder] for a in (et, ebs, ebid, eask, evol, epx))

    bk = OrderBook(symbol)
    i = j = 0
    n, m = len(t), len(et)
    rows = {}
    for st in snap_times:
        st = int(st)
        # 逐笔委托先于逐笔成交(同一毫秒内)
        while i < n and t[i] <= st:
            k = int(kind[i])
            if k == ADD:
                bk.add(int(oid[i]), int(fc[i]), round(float(px[i]), 2), int(vol[i]))
            elif k == DELETE:
                bk.reduce(int(oid[i]), int(vol[i]))
            i += 1
        while j < m and et[j] <= st:
            bs = int(ebs[j])
            if bs == 0:
                bk.reduce(int(ebid[j]), int(evol[j]))
                bk.reduce(int(eask[j]), int(evol[j]))
            elif continuous_resting_only:
                resting = int(eask[j]) if bs == BUY else int(ebid[j])
                bk.reduce(resting, int(evol[j]))
            else:
                bk.reduce(int(ebid[j]), int(evol[j]))
                bk.reduce(int(eask[j]), int(evol[j]))
            # 累计行情统计(最新价/高低/累计量额), 供快照核对
            p = float(epx[j])
            v = int(evol[j])
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
            j += 1

        rows[st] = {
            "bids": bk.depth(BUY), "asks": bk.depth(SELL),
            "last": bk.last, "cum_vol": bk.cum_vol, "cum_turn": bk.cum_turn,
            "open": bk.open_, "high": bk.high, "low": bk.low,
            "n_trades": bk.n_trades,
        }
    return rows
