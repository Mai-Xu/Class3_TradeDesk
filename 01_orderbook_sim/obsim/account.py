"""模拟账户：资金/持仓冻结、T+1 可卖数量、交易费用。"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Fees:
    commission_rate: float = 0.00025   # 佣金 万分之 2.5
    commission_min: float = 5.0        # 单笔最低 5 元
    stamp_tax: float = 0.0005          # 印花税（仅卖出）
    transfer_fee: float = 0.00001      # 过户费（双边）

    def commission(self, amount: float) -> float:
        if amount <= 0:
            return 0.0
        return max(self.commission_min, amount * self.commission_rate)

    def buy(self, amount: float) -> float:
        return round(self.commission(amount) + amount * self.transfer_fee, 2)

    def sell(self, amount: float) -> float:
        return round(self.commission(amount) + amount * self.transfer_fee
                     + amount * self.stamp_tax, 2)


class Account:
    """资金与持仓。

    持仓分三个桶：``sellable``（可卖）、``frozen``（挂单冻结）、``today``（当日买入，T+1 前不可卖）。
    ``total = sellable + frozen + today``。
    """

    def __init__(self, cash: float = 1_000_000.0, fees: Fees | None = None):
        self.init_cash = float(cash)
        self.cash = float(cash)
        self.frozen_cash = 0.0
        self.fees = fees or Fees()
        self.positions: dict[int, dict] = {}
        self.realized_pnl = 0.0
        self.total_fee = 0.0

    # ---------------- 持仓 ----------------
    def _pos(self, symbol: int) -> dict:
        p = self.positions.get(symbol)
        if p is None:
            p = self.positions[symbol] = {"sellable": 0, "frozen": 0, "today": 0,
                                          "cost": 0.0}
        return p

    def position(self, symbol: int) -> dict:
        p = self._pos(symbol)
        return {"total": p["sellable"] + p["frozen"] + p["today"],
                "sellable": p["sellable"], "frozen": p["frozen"],
                "today": p["today"], "cost": round(p["cost"], 2)}

    def sellable(self, symbol: int) -> int:
        return self._pos(symbol)["sellable"]

    def settle(self):
        """日终结算：当日买入转为次日可卖（T+1）。"""
        for p in self.positions.values():
            p["sellable"] += p["today"]
            p["today"] = 0

    # ---------------- 资金 ----------------
    def freeze_cash(self, amount: float):
        if amount > self.cash + 1e-9:
            raise ValueError("可用资金不足")
        self.cash -= amount
        self.frozen_cash += amount

    def release_cash(self, amount: float):
        amount = min(amount, self.frozen_cash)
        self.frozen_cash -= amount
        self.cash += amount

    def available(self) -> float:
        return round(self.cash, 2)

    # ---------------- 成交记账 ----------------
    def apply_buy_fill(self, symbol: int, price: float, qty: int,
                       frozen_portion: float):
        """买入成交：``frozen_portion`` 为该笔对应的已冻结资金。"""
        amount = price * qty
        fee = self.fees.buy(amount)
        spend = amount + fee
        self.frozen_cash -= frozen_portion
        self.cash += frozen_portion - spend
        p = self._pos(symbol)
        p["today"] += qty
        p["cost"] += spend
        self.total_fee += fee
        return spend

    def apply_sell_fill(self, symbol: int, price: float, qty: int):
        amount = price * qty
        fee = self.fees.sell(amount)
        proceeds = amount - fee
        p = self._pos(symbol)
        ref = p["cost"] / max(1, p["sellable"] + p["frozen"] + p["today"])
        p["frozen"] -= qty
        p["cost"] = max(0.0, p["cost"] - ref * qty)
        self.realized_pnl += proceeds - ref * qty
        self.cash += proceeds
        self.total_fee += fee
        return proceeds

    # ---------------- 快照 ----------------
    def snapshot(self, marks: dict | None = None) -> dict:
        marks = marks or {}
        mkt = 0.0
        holding = []
        for sym, p in self.positions.items():
            total = p["sellable"] + p["frozen"] + p["today"]
            if total == 0:
                continue
            px = marks.get(sym, 0.0) or 0.0
            mkt += px * total
            holding.append({"symbol": sym, "total": total, "sellable": p["sellable"],
                            "frozen": p["frozen"], "today": p["today"],
                            "price": round(px, 2), "market_value": round(px * total, 2)})
        holding.sort(key=lambda x: x["symbol"])
        return {"cash": round(self.cash, 2), "frozen_cash": round(self.frozen_cash, 2),
                "market_value": round(mkt, 2),
                "total_asset": round(self.cash + self.frozen_cash + mkt, 2),
                "realized_pnl": round(self.realized_pnl, 2),
                "total_fee": round(self.total_fee, 2),
                "holdings": holding}

