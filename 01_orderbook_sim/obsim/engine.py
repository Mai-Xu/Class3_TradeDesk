"""回放引擎与模拟下单会话。

回放引擎
--------
按时间顺序把「逐笔委托」和「逐笔成交」联合喂进订单簿：

* 新增委托(OrderKind=65) 建单入簿，撤单(OrderKind=68) 按被撤数量减量。
* 成交记录按方向扣减**被动方**（挂单方）；集合竞价成交扣减双方。

  关键点：上交所新增委托记录可能是"立即成交后的剩余挂单量"，
  因此**不能**再把已成交量从后出现的剩余挂单里重复扣减一次。

模拟下单会话
------------
限价委托先与市场簿即时撮合（吃对手价），剩余部分进入用户自己的挂单簿；
后续历史成交流水若价格交叉，则按价格-时间优先对用户挂单做被动成交。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .account import Account
from .book import OrderBook, Trade, BUY, SELL
from . import rules as R

ORD_ADD, ORD_DEL = 65, 68
FC_BUY, FC_SELL = 66, 83
BS_AUCTION = 0


def _sorted_ord(df):
    df = df.sort_values(["Time", "BizIndex"])
    return (df["Time"].to_numpy(np.int64), df["OrderKind"].to_numpy(np.int8),
            df["OrderOriNo"].to_numpy(np.int64), df["FunctionCode"].to_numpy(np.int8),
            df["Price"].to_numpy(np.float64), df["Volume"].to_numpy(np.int64))


def _sorted_exe(df):
    df = df.sort_values(["Time", "BizIndex"])
    return (df["Time"].to_numpy(np.int64), df["BSFlag"].to_numpy(np.int8),
            df["BidOrder"].to_numpy(np.int64), df["AskOrder"].to_numpy(np.int64),
            df["Price"].to_numpy(np.float64), df["Volume"].to_numpy(np.int64))


@dataclass
class OrderRecord:
    oid: int
    symbol: int
    side: str
    price: float
    qty: int
    ts: int = 0
    filled: int = 0
    status: str = "已报"
    reason: str = ""
    frozen_cash: float = 0.0
    fees: float = 0.0
    fills: list = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return self.qty - self.filled

    def as_dict(self):
        return {"oid": self.oid, "symbol": self.symbol, "side": self.side,
                "price": round(self.price, 2), "qty": self.qty, "filled": self.filled,
                "remaining": self.remaining, "status": self.status,
                "reason": self.reason, "ts": self.ts, "fees": round(self.fees, 2)}


class ReplayEngine:
    """单只标的的逐笔回放 + 订单簿。"""

    def __init__(self, symbol: int, ord_df, exe_df, prev_close: float | None = None,
                 on_market_trade=None):
        self.symbol = int(symbol)
        self.book = OrderBook()
        self.ot, self.ok, self.oo, self.of, self.op, self.ov = _sorted_ord(ord_df)
        self.et, self.eb, self.ebid, self.eask, self.ep, self.ev = _sorted_exe(exe_df)
        self.io = self.ie = 0
        self.time = 0
        self.prev_close = float(prev_close) if prev_close else None
        self.last = None
        self.high = None
        self.low = None
        self.cum_vol = 0
        self.cum_turn = 0
        self.n_trades = 0
        self.open_price = None
        self.tape: list[Trade] = []
        self.on_market_trade = on_market_trade

    # ---------------- 回放 ----------------
    def _apply_trade(self, ts, bs, bid, ask, px, vol):
        if bs == BS_AUCTION:
            self.book.reduce(int(bid), int(vol))
            self.book.reduce(int(ask), int(vol))
        else:
            resting = int(ask) if bs == FC_BUY else int(bid)
            self.book.reduce(resting, int(vol))
        px = float(px)
        vol = int(vol)
        if self.open_price is None:
            self.open_price = px
        self.last = px
        self.high = px if self.high is None else max(self.high, px)
        self.low = px if self.low is None else min(self.low, px)
        self.cum_vol += vol
        self.cum_turn += int(round(px * vol))
        self.n_trades += 1
        if self.on_market_trade is not None:
            self.on_market_trade(ts, bs, px, vol)

    def advance_to(self, t: int, trade_log: list | None = None):
        """把回放推进到时间 ``t``（含）。同一毫秒内先处理委托再处理成交。"""
        no, ne = len(self.ot), len(self.et)
        while self.io < no and self.ot[self.io] <= t:
            k = int(self.ok[self.io])
            if k == ORD_ADD:
                self.book.add(int(self.oo[self.io]), BUY if int(self.of[self.io]) == FC_BUY else SELL,
                              round(float(self.op[self.io]), 2), int(self.ov[self.io]),
                              int(self.ot[self.io]))
            elif k == ORD_DEL:
                self.book.reduce(int(self.oo[self.io]), int(self.ov[self.io]))
            self.io += 1
        while self.ie < ne and self.et[self.ie] <= t:
            self._apply_trade(int(self.et[self.ie]), int(self.eb[self.ie]),
                              int(self.ebid[self.ie]), int(self.eask[self.ie]),
                              float(self.ep[self.ie]), int(self.ev[self.ie]))
            self.ie += 1
        self.time = t

    def ladder(self, n=10):
        return {"asks": self.book.depth(SELL, n)[::-1],
                "bids": self.book.depth(BUY, n),
                "last": None if self.last is None else round(self.last, 2),
                "cum_volume": self.cum_vol,
                "n_trades": self.n_trades}

    def reset(self):
        """清空订单簿与回放游标（保留已解析的数据数组，便于反复回放）。"""
        self.book = OrderBook()
        self.io = self.ie = 0
        self.time = 0
        self.last = self.high = self.low = self.open_price = None
        self.cum_vol = self.cum_turn = self.n_trades = 0

    def mark_price(self):
        if self.last:
            return self.last
        bb, ba = self.book.best_bid(), self.book.best_ask()
        if bb and ba:
            return (bb + ba) / 2
        return bb or ba or self.prev_close or 0.0


class TradingSession:
    """模拟账户 + 委托/成交管理。"""

    def __init__(self, engine: ReplayEngine, account: Account | None = None):
        self.engine = engine
        self.account = account or Account()
        self.orders: dict[int, OrderRecord] = {}
        self.seq = 0
        self.fills: list[dict] = []
        self.rejects: list[dict] = []
        engine.on_market_trade = self._on_market_trade

    # ---------------- 下单 ----------------
    def place(self, side: str, price: float, qty: int) -> OrderRecord:
        sym = self.engine.symbol
        self.seq += 1
        rec = OrderRecord(self.seq, sym, side, float(price), int(qty), self.engine.time)
        try:
            self._validate(rec)
        except R.RuleError as e:
            rec.status, rec.reason = "已拒绝", str(e)
            self.orders[rec.oid] = rec
            self.rejects.append(rec.as_dict())
            return rec
        # 冻结
        if side == R.BUY:
            freeze = price * qty + self.account.fees.buy(price * qty)
            try:
                self.account.freeze_cash(freeze)
            except ValueError as e:
                rec.status, rec.reason = "已拒绝", str(e)
                self.orders[rec.oid] = rec
                self.rejects.append(rec.as_dict())
                return rec
            rec.frozen_cash = freeze
        else:
            p = self.account._pos(sym)
            if p["sellable"] < qty:
                rec.status, rec.reason = "已拒绝", f"可卖持仓不足（可卖 {p['sellable']}，T+1 当日买入不可卖）"
                self.orders[rec.oid] = rec
                self.rejects.append(rec.as_dict())
                return rec
            p["sellable"] -= qty
            p["frozen"] += qty
        self.orders[rec.oid] = rec
        # 即时撮合
        self._match_against_market(rec)
        return rec

    def _validate(self, rec: OrderRecord):
        sym = self.engine.symbol
        R.check_quantity(sym, rec.side, rec.qty, self.account.sellable(sym))
        R.check_price(sym, rec.side, rec.price, self.engine.prev_close,
                      self.engine.book.best_bid(), self.engine.book.best_ask(),
                      self.engine.last)

    def _match_against_market(self, rec: OrderRecord):
        """与市场簿即时撮合，剩余部分进入用户挂单簿。"""
        book = self.engine.book
        while rec.remaining > 0:
            opp_best = book.best_ask() if rec.side == R.BUY else book.best_bid()
            if opp_best is None:
                break
            if rec.side == R.BUY and opp_best > rec.price:
                break
            if rec.side == R.SELL and opp_best < rec.price:
                break
            oid2 = self._pop_from_market(book, rec.side, opp_best)
            if oid2 is None:
                break
            o2 = book.get(oid2)
            if o2 is None or o2.remaining <= 0:
                continue
            v = min(rec.remaining, o2.remaining)
            book.reduce(oid2, v)
            self._fill(rec, opp_best, v, counterparty="market", passive=False)
        if rec.remaining > 0 and rec.status not in ("已拒绝",):
            rec.status = "部分成交" if rec.filled else "已报"
        elif rec.remaining == 0:
            rec.status = "全部成交"

    def _pop_from_market(self, book: OrderBook, side: str, price: float):
        """取对手方价位队首订单号（价格-时间优先）。"""
        q = (book.aqueues if side == R.BUY else book.bqueues).get(price)
        while q:
            oid = q[0]
            o = book.get(oid)
            if o is None or o.remaining <= 0:
                q.popleft()
                continue
            return oid
        return None

    # ---------------- 被动成交 ----------------
    def _on_market_trade(self, ts, bs, price, volume):
        """历史成交触发用户挂单的被动成交（价格-时间优先，简化队列模型）。"""
        if bs == BS_AUCTION:
            sides = (R.BUY, R.SELL)
        elif bs == FC_BUY:      # 主动买 -> 吃卖单
            sides = (R.SELL,)
        else:
            sides = (R.BUY,)
        left = volume
        for side in sides:
            if left <= 0:
                break
            for rec in self._resting(side, price):
                if left <= 0:
                    break
                if rec.side == R.BUY and price > rec.price:
                    continue
                if rec.side == R.SELL and price < rec.price:
                    continue
                v = min(rec.remaining, left)
                self._fill(rec, price, v, counterparty="tape", passive=True, ts=ts)
                left -= v

    def _resting(self, side: str, price: float):
        """挂单簿里能与此笔成交成交的委托，按价格优先、时间优先排序。"""
        cand = [r for r in self.orders.values()
                if r.side == side and r.remaining > 0 and r.status in ("已报", "部分成交")
                and not r.reason]
        if side == R.BUY:
            cand = [r for r in cand if r.price >= price]
            cand.sort(key=lambda r: (-r.price, r.oid))
        else:
            cand = [r for r in cand if r.price <= price]
            cand.sort(key=lambda r: (r.price, r.oid))
        return cand

    def _fill(self, rec: OrderRecord, price: float, vol: int,
              counterparty: str, passive: bool, ts: int | None = None):
        prior = rec.filled
        rec.filled += vol
        ts = self.engine.time if ts is None else ts
        if rec.side == R.BUY:
            portion = rec.frozen_cash * vol / rec.qty
            spend = self.account.apply_buy_fill(rec.symbol, price, vol, portion)
            rec.frozen_cash -= portion
            fee = spend - price * vol
        else:
            proceeds = self.account.apply_sell_fill(rec.symbol, price, vol)
            fee = price * vol - proceeds
        rec.fees += fee
        rec.status = "全部成交" if rec.remaining == 0 else "部分成交"
        self.fills.append({"ts": ts, "oid": rec.oid, "symbol": rec.symbol,
                           "side": rec.side, "price": round(price, 2), "volume": vol,
                           "fee": round(fee, 2), "counterparty": counterparty,
                           "passive": passive})

    # ---------------- 撤单 ----------------
    def cancel(self, oid: int) -> OrderRecord | None:
        rec = self.orders.get(oid)
        if rec is None:
            return None
        if rec.remaining <= 0:
            rec.reason = "无可撤数量"
            return rec
        if rec.side == R.BUY:
            self.account.release_cash(rec.frozen_cash)
            rec.frozen_cash = 0.0
        else:
            p = self.account._pos(rec.symbol)
            p["frozen"] -= rec.remaining
            p["sellable"] += rec.remaining
        rec.qty = rec.filled
        rec.status = "已撤单"
        return rec

    # ---------------- 快照 ----------------
    def state(self, depth=10) -> dict:
        marks = {self.engine.symbol: self.engine.mark_price()}
        return {"symbol": self.engine.symbol,
                "time": self.engine.time,
                "market": self.engine.ladder(depth),
                "orders": [r.as_dict() for r in
                           sorted(self.orders.values(), key=lambda r: r.oid, reverse=True)[:60]],
                "fills": list(reversed(self.fills[-60:])),
                "rejects": list(reversed(self.rejects[-20:])),
                "account": self.account.snapshot(marks)}
