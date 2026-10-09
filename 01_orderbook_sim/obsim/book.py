"""价格-时间优先（FIFO）限价订单簿，含连续竞价与集合竞价撮合。"""
from __future__ import annotations

import heapq
from collections import deque

BUY, SELL = "buy", "sell"


class Order:
    __slots__ = ("oid", "side", "price", "volume", "remaining", "ts", "seq", "owner")

    def __init__(self, oid, side, price, volume, ts, seq=0, owner="market"):
        self.oid = oid
        self.side = side
        self.price = price
        self.volume = volume
        self.remaining = volume
        self.ts = ts
        self.seq = seq          # 同价位内的时间优先序号
        self.owner = owner      # "market" 为行情回放, "user" 为模拟下单


class Trade:
    __slots__ = ("ts", "price", "volume", "bid_oid", "ask_oid", "aggressor")

    def __init__(self, ts, price, volume, bid_oid, ask_oid, aggressor):
        self.ts = ts
        self.price = price
        self.volume = volume
        self.bid_oid = bid_oid
        self.ask_oid = ask_oid
        self.aggressor = aggressor   # BUY / SELL / None(集合竞价)

    def as_dict(self):
        return {"ts": self.ts, "price": round(self.price, 2), "volume": self.volume,
                "bid_oid": self.bid_oid, "ask_oid": self.ask_oid,
                "aggressor": self.aggressor or ""}


class OrderBook:
    """levels[(side, price)] 维护价位总量；queues[price] 维护价位内 FIFO 队列。"""

    def __init__(self):
        self.levels: dict = {}
        self.bqueues: dict = {}
        self.aqueues: dict = {}
        self.orders: dict = {}
        self.bid_heap: list = []
        self.ask_heap: list = []
        self._seq = 0

    # ---------------- 基础操作 ----------------
    def add(self, oid, side, price, volume, ts=0, owner="market") -> Order:
        self._seq += 1
        o = Order(oid, side, round(price, 2), int(volume), ts, self._seq, owner)
        self.orders[oid] = o
        key = (side, o.price)
        self.levels[key] = self.levels.get(key, 0) + o.remaining
        if side == BUY:
            q = self.bqueues.get(o.price)
            if q is None:
                self.bqueues[o.price] = deque((oid,))
                heapq.heappush(self.bid_heap, -o.price)
            else:
                q.append(oid)
        else:
            q = self.aqueues.get(o.price)
            if q is None:
                self.aqueues[o.price] = deque((oid,))
                heapq.heappush(self.ask_heap, o.price)
            else:
                q.append(oid)
        return o

    def reduce(self, oid, volume) -> int:
        """按订单号扣减剩余量（成交或撤单），返回实际扣减量。"""
        o = self.orders.get(oid)
        if o is None or o.remaining <= 0:
            return 0
        d = volume if volume < o.remaining else o.remaining
        o.remaining -= d
        self.levels[(o.side, o.price)] -= d
        return d

    def remove(self, oid) -> int:
        o = self.orders.get(oid)
        if o is None:
            return 0
        d = o.remaining
        self.levels[(o.side, o.price)] -= d
        o.remaining = 0
        return d

    def get(self, oid):
        return self.orders.get(oid)

    def _clean(self, side):
        h = self.bid_heap if side == BUY else self.ask_heap
        while h:
            px = -h[0] if side == BUY else h[0]
            if self.levels.get((side, px), 0) > 0:
                return px
            heapq.heappop(h)
        return None

    def best_bid(self):
        return self._clean(BUY)

    def best_ask(self):
        return self._clean(SELL)

    # ---------------- 连续竞价撮合 ----------------
    def match(self, oid, side, price, volume, ts, on_trade=None) -> int:
        """新到委托与对手方最优价撮合；返回剩余未成交量。"""
        rem = int(volume)
        opp = SELL if side == BUY else BUY
        while rem > 0:
            bp = self._clean(opp)
            if bp is None:
                break
            if side == BUY and bp > price:
                break
            if side == SELL and bp < price:
                break
            q = self.aqueues[bp] if opp == SELL else self.bqueues[bp]
            while rem > 0 and q:
                oid2 = q[0]
                o2 = self.orders.get(oid2)
                if o2 is None or o2.remaining <= 0:
                    q.popleft()
                    continue
                tv = rem if rem < o2.remaining else o2.remaining
                o2.remaining -= tv
                self.levels[(opp, bp)] -= tv
                rem -= tv
                bid_oid, ask_oid = (oid, oid2) if side == BUY else (oid2, oid)
                if on_trade:
                    on_trade(Trade(ts, bp, tv, bid_oid, ask_oid, side))
                if o2.remaining <= 0:
                    q.popleft()
            if not q:
                del (self.aqueues if opp == SELL else self.bqueues)[bp]
        return rem

    # ---------------- 集合竞价 ----------------
    def auction_price(self):
        bvol = [(p, self.levels[(BUY, p)]) for p in self.bqueues if self.levels.get((BUY, p), 0) > 0]
        svol = [(p, self.levels[(SELL, p)]) for p in self.aqueues if self.levels.get((SELL, p), 0) > 0]
        if not bvol or not svol:
            return None
        bmap, smap = dict(bvol), dict(svol)
        best = None
        best_vol = -1
        best_gap = None
        for p in sorted(set(bmap) | set(smap)):
            dem = sum(v for k, v in bmap.items() if k >= p)
            sup = sum(v for k, v in smap.items() if k <= p)
            v = min(dem, sup)
            if v <= 0:
                continue
            gap = abs(dem - sup)
            if v > best_vol or (v == best_vol and gap < best_gap):
                best, best_vol, best_gap = p, v, gap
        return best

    def auction_clear(self, ts, on_trade=None):
        p = self.auction_price()
        if p is None:
            return None
        buys = [(o.seq, o) for o in self.orders.values()
                if o.side == BUY and o.remaining > 0 and o.price >= p]
        sells = [(o.seq, o) for o in self.orders.values()
                 if o.side == SELL and o.remaining > 0 and o.price <= p]
        buys.sort(key=lambda x: (-x[1].price, x[1].seq))
        sells.sort(key=lambda x: (x[1].price, x[1].seq))
        i = j = 0
        while i < len(buys) and j < len(sells):
            bo, so = buys[i][1], sells[j][1]
            if bo.remaining <= 0:
                i += 1
                continue
            if so.remaining <= 0:
                j += 1
                continue
            tv = min(bo.remaining, so.remaining)
            bo.remaining -= tv
            so.remaining -= tv
            self.levels[(BUY, bo.price)] -= tv
            self.levels[(SELL, so.price)] -= tv
            if on_trade:
                on_trade(Trade(ts, p, tv, bo.oid, so.oid, None))
            if bo.remaining <= 0:
                i += 1
            if so.remaining <= 0:
                j += 1
        return p

    # ---------------- 行情 ----------------
    def depth(self, side, n=10):
        src = self.bqueues if side == BUY else self.aqueues
        out = []
        for px in sorted(src, reverse=(side == BUY)):
            v = self.levels.get((side, px), 0)
            if v > 0:
                out.append({"price": round(px, 2), "volume": v})
                if len(out) >= n:
                    break
        return out

    def ladder(self, n=10):
        return {"asks": self.depth(SELL, n)[::-1], "bids": self.depth(BUY, n)}

