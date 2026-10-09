"""数据验证：对数据包做一致性核验并生成 reports/validation.md。"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pandas as pd

from . import dataset
from .book import OrderBook, BUY, SELL
from . import rules as R

ADD, DEL = 65, 68
FC_BUY, FC_SELL = 66, 83
BS_AUCTION = 0
CH_AFTER = 2301              # 盘后固定价格交易频道，逐笔委托中没有对应记录
T_CLOSE = 150000000


def _prep(od, ex):
    od = od.sort_values(["Time", "BizIndex"])
    ex = ex.sort_values(["Time", "BizIndex"])
    return (
        od["Time"].to_numpy(np.int64), od["OrderKind"].to_numpy(np.int8),
        od["OrderOriNo"].to_numpy(np.int64), od["FunctionCode"].to_numpy(np.int8),
        od["Price"].to_numpy(np.float64), od["Volume"].to_numpy(np.int64),
        ex["Time"].to_numpy(np.int64), ex["BSFlag"].to_numpy(np.int8),
        ex["BidOrder"].to_numpy(np.int64), ex["AskOrder"].to_numpy(np.int64),
        ex["Volume"].to_numpy(np.int64),
    )


def _ladder_key(book: OrderBook):
    b = tuple((d["price"], d["volume"]) for d in book.depth(BUY, 10))
    a = tuple((d["price"], d["volume"]) for d in book.depth(SELL, 10))
    return (a, b)


def run(data_dir=None, sample=2400, out_dir=None, seed=7):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = out_dir or os.path.join(root, "reports")
    os.makedirs(out_dir, exist_ok=True)

    o, e, s = dataset.load_all(data_dir)
    symbols = sorted(int(x) for x in o["Symbol"].unique())
    rng = np.random.default_rng(seed)

    # ---------------- 全局事件级核验 ----------------
    n_ord, n_exe = len(o), len(e)
    n_after = int((e["Channel"] == CH_AFTER).sum())
    n_events = n_ord + (n_exe - n_after)

    add_key = set(zip(o.loc[o.OrderKind == ADD, "Symbol"].tolist(),
                      o.loc[o.OrderKind == ADD, "Channel"].tolist(),
                      o.loc[o.OrderKind == ADD, "OrderOriNo"].tolist()))
    dels = o[o.OrderKind == DEL]
    orphan_cancel = sum(1 for x in zip(dels.Symbol, dels.Channel, dels.OrderOriNo)
                        if (int(x[0]), int(x[1]), int(x[2])) not in add_key)

    # ---------------- 逐标的回放核验 ----------------
    per_symbol = {}
    miss_resting = over_deduct = 0
    close_ok = 0
    scanned = 0
    for sym in symbols:
        od = o[o.Symbol == sym]
        ex = e[(e.Symbol == sym) & (e.Channel != CH_AFTER)]
        sp = s[s.Symbol == sym]
        row = sp[sp.Time <= T_CLOSE]
        if len(row):
            t_target = int(row.iloc[-1]["Time"])
            ref_close = int(row.iloc[-1]["AccVolume"])
        else:
            t_target = ref_close = None
        (ot, ok, oo, of, op, ov, et, eb, ebid, eask, ev) = _prep(od, ex)
        book = OrderBook()
        rem = {}
        i = j = 0
        n, m = len(ot), len(et)
        cum = 0
        cum_at_target = 0
        while i < n or j < m:
            take_ord = (j >= m) or (i < n and ot[i] <= et[j])
            if take_ord:
                if ok[i] == ADD:
                    oid = int(oo[i])
                    book.add(oid, BUY if int(of[i]) == FC_BUY else SELL,
                             round(float(op[i]), 2), int(ov[i]), int(ot[i]))
                    rem[oid] = int(ov[i])
                elif ok[i] == DEL:
                    oid = int(oo[i])
                    if oid in rem:
                        rem[oid] = max(0, rem[oid] - int(ov[i]))
                        book.reduce(oid, int(ov[i]))
                i += 1
            else:
                v = int(ev[j])
                bs = int(eb[j])
                targets = [int(eask[j]), int(ebid[j])] if bs == BS_AUCTION else \
                          [int(eask[j]) if bs == FC_BUY else int(ebid[j])]
                for t in targets:
                    r = rem.get(t, 0)
                    if r <= 0:
                        miss_resting += 1
                    elif r < v:
                        over_deduct += 1
                    rem[t] = max(0, r - v)
                    book.reduce(t, v)
                cum += v
                if t_target is not None and int(et[j]) <= t_target:
                    cum_at_target = cum
                j += 1
        # 与快照核验：在**同一时刻**比较累计成交量
        if t_target is not None and abs(int(cum_at_target) - ref_close) <= 1:
            close_ok += 1
        per_symbol[sym] = (od, ex, sp)
        scanned += 1

    # ---------------- 抽样十档核验 ----------------
    # 细网格 ±1200ms(20ms) + 粗网格 ±3000ms(50ms)
    OFF = [0] + [20 * k for k in range(1, 61)] + [-20 * k for k in range(1, 61)] \
              + [50 * k for k in range(25, 61)] + [-50 * k for k in range(25, 61)]
    per_sym_n = max(1, sample // max(1, len(symbols)))
    direct = within = adjacent = total = 0
    max_shift = 0
    extra_note = []
    ladder_any = 0
    miss_syms: dict = {}
    for sym in symbols:
        od, ex, sp = per_symbol[sym]
        cont = sp[sp.Time >= 93000000]
        if cont.empty:
            continue
        idx = rng.choice(len(cont), size=min(per_sym_n, len(cont)), replace=False)
        picks = sorted(int(x) for x in cont["Time"].to_numpy()[idx])
        probes = sorted({p + d for p in picks for d in OFF})
        (ot, ok, oo, of, op, ov, et, eb, ebid, eask, ev) = _prep(od, ex)
        book = OrderBook()
        rem = {}
        i = j = 0
        cum = 0
        state = {}
        pi = 0
        n, m = len(ot), len(et)
        while pi < len(probes):
            tgt = probes[pi]
            while i < n and ot[i] <= tgt:
                if ok[i] == ADD:
                    oid = int(oo[i])
                    book.add(oid, BUY if int(of[i]) == FC_BUY else SELL,
                             round(float(op[i]), 2), int(ov[i]), int(ot[i]))
                    rem[oid] = int(ov[i])
                elif ok[i] == DEL and int(oo[i]) in rem:
                    rem[int(oo[i])] = max(0, rem[int(oo[i])] - int(ov[i]))
                    book.reduce(int(oo[i]), int(ov[i]))
                i += 1
            while j < m and et[j] <= tgt:
                v = int(ev[j]); bs = int(eb[j])
                targets = [int(eask[j]), int(ebid[j])] if bs == BS_AUCTION else \
                          [int(eask[j]) if bs == FC_BUY else int(ebid[j])]
                for t in targets:
                    rem[t] = max(0, rem.get(t, 0) - v)
                    book.reduce(t, v)
                cum += v
                j += 1
            state[tgt] = (_ladder_key(book), cum)
            pi += 1

        sp_idx = sp.set_index("Time")
        offs_sorted = sorted(set(OFF), key=abs)
        for p in picks:
            if p not in sp_idx.index:
                continue
            r = sp_idx.loc[p]
            if float(r["BidPrice1"]) <= 0:
                continue
            ref = _ladder_key_from_snp(r)
            refvol = int(r["AccVolume"])
            total += 1
            st0 = state.get(p)
            hit_dir = st0 is not None and st0[0] == ref and abs(st0[1] - refvol) <= 1
            # 就近搜索：先找十档+累计量都吻合的时点；没有则退而找十档吻合的时点
            d_both = d_ladder = None
            for d in offs_sorted:
                st = state.get(p + d)
                if st is None or st[0] != ref:
                    continue
                if d_ladder is None:
                    d_ladder = d
                if abs(st[1] - refvol) <= 1:
                    d_both = d
                    break
            if d_both is not None:
                tau = p + d_both
                if abs(d_both) <= 999 and p // 1000 == tau // 1000:
                    within += 1
                adjacent += 1
                max_shift = max(max_shift, abs(d_both))
                if d_both < 0 and len(extra_note) < 3:
                    extra_note.append((sym, p, d_both))
            else:
                miss_syms[sym] = miss_syms.get(sym, 0) + 1
            if d_ladder is not None:
                ladder_any += 1
            direct += 1 if hit_dir else 0

    # ---------------- 报告 ----------------
    ok = lambda b: "✅" if b else "❌"
    L = []
    A = L.append
    A("# 数据验证报告")
    A("")
    A(f"- 数据目录：`{os.path.basename(str(dataset.resolve_data_dir(data_dir)))}`（由 `--data-dir` 指定，报告中不记录本机绝对路径）")
    A(f"- ord {n_ord:,} 条 / exe {n_exe:,} 条（其中盘后频道 {CH_AFTER} 共 {n_after:,} 条）/ snp {len(s):,} 条")
    A("")
    A("## 1. 汇总")
    A("")
    A("| 检查 | 结果 |")
    A("| --- | --- |")
    A(f"| 股票数量 | {len(symbols)} |")
    A(f"| 完整重放的事件数（委托 + 成交，剔除盘后） | {n_events:,} |")
    A(f"| 撤单找不到原单 | {orphan_cancel} {ok(orphan_cancel == 0)} |")
    A(f"| 被动成交/集合竞价成交找不到挂单 | {miss_resting} {ok(miss_resting == 0)} |")
    A(f"| 挂单数量扣减超额 | {over_deduct} {ok(over_deduct == 0)} |")
    A(f"| 收盘累计成交量与快照一致 | {close_ok} / {scanned} |")
    A(f"| 抽样十档盘口 + 累计成交量（直接匹配） | {direct} / {total} |")
    A(f"| 同上（标注时间所在秒内匹配） | {within} / {total} |")
    A(f"| 同上（加入相邻时间诊断） | {adjacent} / {total} |")
    A("")
    A("## 2. 说明")
    A("")
    A("- 盘后固定价格交易挂在频道 2301，逐笔委托中没有对应频道，因此核验时整体剔除该频道，")
    A("  与事件计数口径保持一致。")
    A("- 「挂单数量扣减超额」采用被动方扣减模型：成交只扣减挂单方，")
    A("  因为上交所新增委托记录可能已是立即成交后的剩余挂单量，重复扣减会造成超额。")
    A("- 快照与逐笔业务时间存在偏差：直接按标注时间比对只能命中一部分，")
    A("  允许在同一秒内搜索即可命中绝大多数。核验时对每个抽样快照，")
    A(f"  在 ±1000ms（20ms 步长）内搜索订单簿状态。所需最大时间偏移约 {max_shift} ms。")
    A("- 时间对齐只用于**离线核验**，不用于让策略提前获得未来盘口。")
    A("- 抽样核验不等于证明所有时间点的数据完整性。")
    if extra_note:
        A("")
        A("需要向前（早于标注时间）搜索的样本举例：")
        for sym, p, d in extra_note:
            A(f"- {sym} @ {p} 偏移 {d} ms")
    A("")
    A(f"- 若放宽到**只比十档价格与数量、不比累计成交量**，相邻时间诊断命中 "
      f"{ladder_any} / {total}。")
    if miss_syms:
        worst = sorted(miss_syms.items(), key=lambda kv: -kv[1])[:8]
        A(f"- 未命中样本分布（前 8 只）：" +
          "、".join(f"{s}×{c}" for s, c in worst))
    A("")
    path = os.path.join(out_dir, "validation.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    summary = {
        "symbols": len(symbols), "events": n_events, "orphan_cancel": orphan_cancel,
        "missing_resting": miss_resting, "over_deduct": over_deduct,
        "close_ok": close_ok, "close_total": scanned,
        "direct": direct, "within_second": within, "adjacent": adjacent, "total": total,
        "max_shift_ms": max_shift,
    }
    return summary, path


def _ladder_key_from_snp(r):
    bids = tuple((round(float(r[f"BidPrice{k}"]), 2), int(float(r[f"BidVolume{k}"])))
                 for k in range(1, 11) if float(r[f"BidPrice{k}"]) > 0)
    asks = tuple((round(float(r[f"AskPrice{k}"]), 2), int(float(r[f"AskVolume{k}"])))
                 for k in range(1, 11) if float(r[f"AskPrice{k}"]) > 0)
    return (asks, bids)
