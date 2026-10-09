"""交易规则：区分主板与科创板。

设计说明
--------
交易所业务细则版本较多，这里把每一条规则都做成 ``BoardRules`` 上的可配置字段，
默认值取当前主流口径，便于按课程口径调整：

* 涨跌幅：主板 ±10%，科创板 ±20%（以昨收为基准）。
* 申报数量：主板 100 股起、须为 100 股整数倍；科创板 200 股起、超出部分可按 1 股递增。
  卖出时若持仓不足最小申报量，允许一次性全部卖出。
* 连续竞价"有效申报价格范围"（价格笼子）：
  买入申报价 ≤ 买入基准价 × (1+2%)，卖出申报价 ≥ 卖出基准价 × (1-2%)。
  基准价取对手方最优价，缺省时依次退化为最新成交价、昨收。
* 最小价格变动单位 0.01 元。
"""
from __future__ import annotations

from dataclasses import dataclass

MAIN = "main"
STAR = "star"

BUY, SELL = "buy", "sell"
TICK = 0.01


def board_of(symbol: int) -> str:
    """688 开头为科创板，其余（600/601/603/605）视为主板。"""
    return STAR if str(int(symbol)).startswith("688") else MAIN


@dataclass(frozen=True)
class BoardRules:
    board: str
    label: str
    lot_min: int          # 单笔最小申报数量
    lot_step: int         # 递增单位
    lot_round: bool       # 是否必须为 lot_step 的整数倍
    price_limit: float    # 涨跌幅限制
    cage: float           # 连续竞价价格笼子比例


_RULES = {
    MAIN: BoardRules(MAIN, "主板", 100, 100, True, 0.10, 0.02),
    STAR: BoardRules(STAR, "科创板", 200, 1, False, 0.20, 0.02),
}


def rules_for(symbol: int) -> BoardRules:
    return _RULES[board_of(symbol)]


class RuleError(ValueError):
    """委托不满足申报规则。"""


def check_quantity(symbol: int, side: str, qty: int, sellable: int = 0) -> None:
    """校验申报数量。卖出不足最小申报量时允许一次性卖出。"""
    r = rules_for(symbol)
    if qty <= 0:
        raise RuleError("申报数量必须为正整数")
    if side == SELL and 0 < sellable < r.lot_min:
        if qty != sellable:
            raise RuleError(f"持仓不足 {r.lot_min} 股时应一次性卖出（当前可卖 {sellable}）")
        return
    if qty < r.lot_min:
        raise RuleError(f"{r.label}单笔申报不少于 {r.lot_min} 股")
    if r.lot_round and (qty - r.lot_min) % r.lot_step != 0:
        raise RuleError(f"{r.label}申报数量须为 {r.lot_step} 股的整数倍")
    if (not r.lot_round) and qty > r.lot_min and r.lot_step == 1:
        return  # 科创板超出 200 股部分可按 1 股递增


def price_limits(symbol: int, prev_close: float):
    """返回 (跌停价, 涨停价)，按最小变动单位取整。"""
    if prev_close is None or prev_close <= 0:
        return None, None
    r = rules_for(symbol)
    up = round(prev_close * (1 + r.price_limit) / TICK) * TICK
    dn = round(prev_close * (1 - r.price_limit) / TICK) * TICK
    return round(dn, 2), round(up, 2)


def check_price(symbol: int, side: str, price: float, prev_close: float,
                best_bid=None, best_ask=None, last=None) -> None:
    """校验申报价格：跳动单位、涨跌幅、连续竞价价格笼子。"""
    r = rules_for(symbol)
    if price is None or price <= 0:
        raise RuleError("申报价格必须为正数")
    # 价格单位
    if abs(price / TICK - round(price / TICK)) > 1e-6:
        raise RuleError(f"申报价格须为 {TICK:.2f} 元的整数倍")
    # 涨跌幅
    dn, up = price_limits(symbol, prev_close)
    if up is not None and not (dn - 1e-9 <= price <= up + 1e-9):
        raise RuleError(f"申报价格超出涨跌幅限制 [{dn:.2f}, {up:.2f}]")
    # 价格笼子
    if side == BUY:
        base = best_ask if best_ask else (last if last else prev_close)
        if base and price > round(base * (1 + r.cage), 2) + 1e-9:
            raise RuleError(f"买入申报价高于价格笼子上限 {base * (1 + r.cage):.2f}")
    else:
        base = best_bid if best_bid else (last if last else prev_close)
        if base and price < round(base * (1 - r.cage), 2) - 1e-9:
            raise RuleError(f"卖出申报价低于价格笼子下限 {base * (1 - r.cage):.2f}")

