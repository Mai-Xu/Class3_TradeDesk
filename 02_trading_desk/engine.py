"""交易所撮合与行情体系 (Exchange matching & market-data system)
================================================================
面向上海证券交易所 Level-2 逐笔行情数据的订单簿 / 撮合 / 行情生成实现。

核心组件
--------
OrderBook
    价格-时间优先(FIFO)的限价订单簿，支持：
      * 新增委托(add) / 撤单(cancel)
      * 连续竞价撮合(continuous match)
      * 集合竞价撮合(最大成交量算法, 沪市规则)
      * 十档行情快照(depth snapshot)

本模块提供两条互相对照的回放路径
--------------------------------
(1) replay_matching   ——  只用「逐笔委托」驱动撮合引擎，**主动生成逐笔成交**；
(2) reconstruct_book  ——  用「逐笔委托 + 逐笔成交(成交回报)」重建订单簿，
                          **生成 3 秒行情快照**。

之所以要区分这两条路径，是因为上交所逐笔委托**只披露进入订单簿的委托**
(即"挂单")，下单即全部成交的委托不会出现在逐笔委托里（详见 README）。

字段约定(与数据包一致)
----------------------
FunctionCode / BSFlag : 66='B'(买), 83='S'(卖), 0=集合竞价成交(无方向)
OrderKind             : 65='A'(新增), 68='D'(撤单), 83='S'(状态/阶段标记)
Time                  : int, HHMMSSmmm (毫秒)
OrderOriNo            : 交易所订单号；撤单记录指向被撤订单，成交记录引用它
"""
from __future__ import annotations

import heapq
from collections import deque

# ---- 业务代码 ----
BUY, SELL = 66, 83          # FunctionCode / BSFlag
ADD, DELETE, STATE = 65, 68, 83   # OrderKind

# ---- 交易时段(以 HHMMSSmmm 整数表示) ----
T_AUC_OPEN_START = 91500000   # 开盘集合竞价开始
T_AUC_OPEN_END = 92500000     # 开盘集合竞价撮合时点
T_CONT_AM = (93000000, 113000000)   # 上午连续竞价
T_CONT_PM = (130000000, 145700000)  # 下午连续竞价
T_AUC_CLOSE_START = 145700000       # 收盘集合竞价开始
T_AUC_CLOSE_END = 150000000         # 收盘集合竞价撮合时点
T_AFTER_HOURS = (150500000, 153000000)  # 盘后固定价格交易


class Trade:
    """一笔成交(逐笔成交的中间表示)。"""
    __slots__ = ("time", "price", "volume", "bid_oid", "ask_oid", "aggressor")

    def __init__(self, time, price, volume, bid_oid, ask_oid, aggressor):
        self.time = time
        self.price = price
        self.volume = volume
        self.bid_oid = bid_oid     # 买方订单号(OrderOriNo)
        self.ask_oid = ask_oid     # 卖方订单号(OrderOriNo)
        self.aggressor = aggressor  # BUY / SELL / 0(集合竞价)


class OrderBook:
    """单只股票的限价订单簿。

    内部结构
    --------
    levels : {(side, price): 该价位剩余挂单总量}    —— O(1) 判断价位是否为空
    queues : {price: deque(oid)}                   —— 价位内价格-时间优先(FIFO)
    orders : {oid: [side, price, remaining]}       —— 订单号 -> 订单状态
    heaps  : 买卖两侧的最优价懒删除堆
    """

    __slots__ = ("symbol", "levels", "bqueues", "aqueues", "orders",
                 "bid_heap", "ask_heap", "last", "open_", "high", "low",
                 "cum_vol", "cum_turn", "n_trades", "generated")

    def __init__(self, symbol: int):
        self.symbol = symbol
        self.levels: dict = {}
        self.bqueues: dict = {}
        self.aqueues: dict = {}
        self.orders: dict = {}
        self.bid_heap: list = []
        self.ask_heap: list = []
        self.last = 0.0
        self.open_ = 0.0
        self.high = 0.0
        self.low = 0.0
        self.cum_vol = 0
        self.cum_turn = 0
        self.n_trades = 0
        self.generated: list = []   # 由撮合引擎产生的成交

    # ------------------------------------------------------------------ #
    #  基础操作
    # ------------------------------------------------------------------ #
    def add(self, oid: int, side: int, price: float, volume: int):
        """新增委托，进入订单簿等待撮合。"""
        self.orders[oid] = [side, price, volume]
        self.levels[(side, price)] = self.levels.get((side, price), 0) + volume
        if side == BUY:
            q = self.bqueues.get(price)
            if q is None:
                self.bqueues[price] = deque((oid,))
                heapq.heappush(self.bid_heap, -price)
            else:
                q.append(oid)
        else:
            q = self.aqueues.get(price)
            if q is None:
                self.aqueues[price] = deque((oid,))
                heapq.heappush(self.ask_heap, price)
            else:
                q.append(oid)

    def reduce(self, oid: int, volume: int) -> int:
        """按订单号扣减剩余量(成交或撤单)，返回实际扣减量。"""
        o = self.orders.get(oid)
        if o is None:
            return 0
        rem = o[2]
        if rem <= 0:
            return 0
        d = volume if volume < rem else rem
        o[2] = rem - d
        self.levels[(o[0], o[1])] -= d
        return d

    def remove(self, oid: int):
        o = self.orders.get(oid)
        if o is None or o[2] <= 0:
            return
        self.levels[(o[0], o[1])] -= o[2]
        o[2] = 0

    def _clean(self, side: int):
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

    # ------------------------------------------------------------------ #
    #  成交登记 & 连续竞价撮合
    # ------------------------------------------------------------------ #
    def _record(self, time: int, price: float, volume: int, bid_oid, ask_oid, aggressor):
        self.cum_vol += volume
        self.cum_turn += int(round(price * volume))
        self.n_trades += 1
        if self.open_ == 0.0:
            self.open_ = price
        self.last = price
        if price > self.high or self.high == 0.0:
            self.high = price
        if price < self.low or self.low == 0.0:
            self.low = price
        self.generated.append(Trade(time, price, volume, bid_oid, ask_oid, aggressor))

    def match_continuous(self, oid: int, side: int, price: float, volume: int, time: int):
        """新到的限价委托与对手方最优价撮合(价格-时间优先)；剩余量挂入订单簿。"""
        rem = volume
        if side == BUY:
            while rem > 0:
                bp = self._clean(SELL)
                if bp is None or bp > price:
                    break
                q = self.aqueues[bp]
                while rem > 0 and q:
                    oid2 = q[0]
                    o2 = self.orders[oid2]
                    if o2[2] <= 0:
                        q.popleft()
                        continue
                    tv = rem if rem < o2[2] else o2[2]
                    o2[2] -= tv
                    self.levels[(SELL, bp)] -= tv
                    rem -= tv
                    self._record(time, bp, tv, oid, oid2, BUY)
                    if o2[2] <= 0:
                        q.popleft()
                if not q:
                    del self.aqueues[bp]
        else:
            while rem > 0:
                bp = self._clean(BUY)
                if bp is None or bp < price:
                    break
                q = self.bqueues[bp]
                while rem > 0 and q:
                    oid2 = q[0]
                    o2 = self.orders[oid2]
                    if o2[2] <= 0:
                        q.popleft()
                        continue
                    tv = rem if rem < o2[2] else o2[2]
                    o2[2] -= tv
                    self.levels[(BUY, bp)] -= tv
                    rem -= tv
                    self._record(time, bp, tv, oid2, oid, SELL)
                    if o2[2] <= 0:
                        q.popleft()
                if not q:
                    del self.bqueues[bp]
        if rem > 0:
            self.add(oid, side, price, rem)

    # ------------------------------------------------------------------ #
    #  集合竞价撮合(沪市: 最大成交量)
    # ------------------------------------------------------------------ #
    def _live_prices(self, side: int):
        src = self.bqueues if side == BUY else self.aqueues
        out = []
        for px in src:
            v = self.levels.get((side, px), 0)
            if v > 0:
                out.append((px, v))
        return out

    def auction_price(self):
        """返回集合竞价的理论开盘/收盘价(最大成交量; 并列时取未成交量最小、再取中间价)。"""
        bvol = self._live_prices(BUY)
        svol = self._live_prices(SELL)
        if not bvol or not svol:
            return None
        bvol.sort()            # 升序
        svol.sort()
        prices = sorted({p for p, _ in bvol} | {p for p, _ in svol})
        # 需求: 价格 >= p 的买单总量 ; 供给: 价格 <= p 的卖单总量
        bmap = dict(bvol)
        smap = dict(svol)
        best, best_gap, best_px = None, None, None
        # 从高到低累计买量
        demand = {}
        run = 0
        for p in sorted(bmap, reverse=True):
            run += bmap[p]
            demand[p] = run
        supply = {}
        run = 0
        for p in sorted(smap):
            run += smap[p]
            supply[p] = run
        bkeys = sorted(bmap, reverse=True)
        skeys = sorted(smap)
        for p in prices:
            d = 0
            for k in bkeys:
                if k >= p:
                    d += bmap[k]
                else:
                    break
            s = 0
            for k in skeys:
                if k <= p:
                    s += smap[k]
                else:
                    break
            v = d if d < s else s
            if v <= 0:
                continue
            if best is None or v > best:
                best, best_gap, best_px = v, abs(d - s), p
            elif v == best and abs(d - s) < best_gap:
                best_gap, best_px = abs(d - s), p
        return best_px

    def auction_clear(self, time: int):
        """按集合竞价价格撮合成交(同一价格内价格-时间优先)。"""
        p = self.auction_price()
        if p is None:
            return
        # 可成交买单: 价格>=p ; 可成交卖单: 价格<=p
        buys = [(oid, self.orders[oid])
                for px in self.bqueues if px >= p
                for oid in self.bqueues[px] if self.orders[oid][2] > 0]
        sells = [(oid, self.orders[oid])
                 for px in self.aqueues if px <= p
                 for oid in self.aqueues[px] if self.orders[oid][2] > 0]
        # 价格优先 -> 时间优先(订单号递增近似委托先后)
        buys.sort(key=lambda x: (-x[1][1], x[0]))
        sells.sort(key=lambda x: (x[1][1], x[0]))
        i = j = 0
        while i < len(buys) and j < len(sells):
            boid, bo = buys[i]
            soid, so = sells[j]
            if bo[2] <= 0:
                i += 1
                continue
            if so[2] <= 0:
                j += 1
                continue
            tv = bo[2] if bo[2] < so[2] else so[2]
            bo[2] -= tv
            so[2] -= tv
            self.levels[(BUY, bo[1])] -= tv
            self.levels[(SELL, so[1])] -= tv
            self._record(time, p, tv, boid, soid, 0)
            if bo[2] <= 0:
                i += 1
            if so[2] <= 0:
                j += 1

    # ------------------------------------------------------------------ #
    #  行情快照
    # ------------------------------------------------------------------ #
    def depth(self, side: int, n: int = 10):
        """返回 [(price, volume), ...] 前十档(不含空档)。"""
        out = []
        keys = sorted(self.bqueues if side == BUY else self.aqueues,
                      reverse=(side == BUY))
        for px in keys:
            v = self.levels.get((side, px), 0)
            if v > 0:
                out.append((px, v))
                if len(out) >= n:
                    break
        return out

    def total_volume(self, side: int) -> int:
        return sum(self.levels.get((side, p), 0) for p in
                   (self.bqueues if side == BUY else self.aqueues))

    def avg_price(self, side: int):
        num = den = 0
        for px in (self.bqueues if side == BUY else self.aqueues):
            v = self.levels.get((side, px), 0)
            num += px * v
            den += v
        return num / den if den else 0.0


def is_continuous(t: int) -> bool:
    return (T_CONT_AM[0] <= t < T_CONT_AM[1]) or (T_CONT_PM[0] <= t < T_CONT_PM[1])

