"""核心测试（11 项）。

运行: python -m unittest discover -s tests -v
"""
from __future__ import annotations

import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from obsim import rules as R                       # noqa: E402
from obsim.account import Account                  # noqa: E402
from obsim.book import OrderBook, BUY, SELL        # noqa: E402
from obsim.engine import ReplayEngine, TradingSession, ORD_ADD  # noqa: E402


def empty_ord():
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in [
        ("Time", "int64"), ("BizIndex", "int64"), ("OrderKind", "int8"),
        ("OrderOriNo", "int64"), ("FunctionCode", "int8"),
        ("Price", "float64"), ("Volume", "int64")]})


def empty_exe():
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in [
        ("Time", "int64"), ("BizIndex", "int64"), ("BSFlag", "int8"),
        ("BidOrder", "int64"), ("AskOrder", "int64"),
        ("Price", "float64"), ("Volume", "int64")]})


def mk_engine(symbol=600000, prev_close=10.0):
    return ReplayEngine(symbol, empty_ord(), empty_exe(), prev_close)


class TestBook(unittest.TestCase):

    def test_01_auction_price_maximises_volume(self):
        """集合竞价按最大成交量定价。"""
        bk = OrderBook()
        bk.add(1, BUY, 10.20, 100)
        bk.add(2, BUY, 10.00, 200)
        bk.add(3, SELL, 9.90, 50)
        bk.add(4, SELL, 9.95, 100)
        bk.add(5, SELL, 10.00, 100)
        # 候选价可成交量: 9.90->50, 9.95->150, 10.00->250(最大且唯一), 10.20->100
        self.assertEqual(bk.auction_price(), 10.00)
        trades = []
        bk.auction_clear(92500000, trades.append)
        self.assertEqual(sum(t.volume for t in trades), 250)
        self.assertTrue(all(abs(t.price - 10.00) < 1e-9 for t in trades))

    def test_02_price_then_time_priority(self):
        """先价格优先，同价再时间优先(FIFO)。"""
        bk = OrderBook()
        bk.add(1, SELL, 10.00, 100)
        bk.add(2, SELL, 10.00, 100)
        bk.add(3, SELL, 9.99, 100)
        fills = []
        rem = bk.match(9, BUY, 10.00, 250, 0, lambda t: fills.append((t.price, t.ask_oid)))
        self.assertEqual(fills, [(9.99, 3), (10.00, 1), (10.00, 2)])
        self.assertEqual(rem, 0)

    def test_03_partial_fill_rests(self):
        """部分成交后剩余量留在订单簿上。"""
        bk = OrderBook()
        bk.add(1, SELL, 10.00, 100)
        rem = bk.match(2, BUY, 10.00, 400, 0)
        self.assertEqual(rem, 300)
        bk.add(2, BUY, 10.00, rem)
        self.assertEqual(bk.best_bid(), 10.00)
        self.assertEqual(bk.depth(BUY)[0]["volume"], 300)


class TestAccount(unittest.TestCase):

    def test_04_cancel_releases_frozen_cash(self):
        eng = mk_engine()
        sess = TradingSession(eng, Account(100_000))
        before = sess.account.cash
        rec = sess.place("buy", 10.00, 100)
        self.assertEqual(rec.status, "已报")
        self.assertLess(sess.account.cash, before)
        self.assertGreater(sess.account.frozen_cash, 0)
        sess.cancel(rec.oid)
        self.assertAlmostEqual(sess.account.cash, before, places=2)
        self.assertAlmostEqual(sess.account.frozen_cash, 0.0, places=2)

    def test_05_cancel_releases_frozen_position(self):
        eng = mk_engine()
        acc = Account(100_000)
        acc.positions[600000] = {"sellable": 500, "frozen": 0, "today": 0, "cost": 0.0}
        sess = TradingSession(eng, acc)
        rec = sess.place("sell", 10.00, 300)
        self.assertEqual(acc.position(600000)["frozen"], 300)
        self.assertEqual(acc.position(600000)["sellable"], 200)
        sess.cancel(rec.oid)
        self.assertEqual(acc.position(600000)["frozen"], 0)
        self.assertEqual(acc.position(600000)["sellable"], 500)

    def test_06_t_plus_1_blocks_intraday_sell(self):
        """当日买入计入今仓、当日不可卖；日终结算后才可卖。"""
        acc = Account(100_000)
        acc.freeze_cash(10.00 * 100)
        acc.apply_buy_fill(600000, 10.00, 100, frozen_portion=10.00 * 100)
        self.assertEqual(acc.position(600000)["today"], 100)
        self.assertEqual(acc.sellable(600000), 0)
        eng = mk_engine()
        rec = TradingSession(eng, acc).place("sell", 10.00, 100)
        self.assertEqual(rec.status, "已拒绝")
        self.assertIn("T+1", rec.reason)
        acc.settle()
        self.assertEqual(acc.sellable(600000), 100)
        rec2 = TradingSession(mk_engine(), acc).place("sell", 10.00, 100)
        self.assertEqual(rec2.status, "已报")


class TestRules(unittest.TestCase):

    def test_07_main_board_lot(self):
        """主板 100 股起、须为 100 股整数倍。"""
        self.assertEqual(R.board_of(600000), R.MAIN)
        with self.assertRaises(R.RuleError):
            R.check_quantity(600000, "buy", 150)
        with self.assertRaises(R.RuleError):
            R.check_quantity(600000, "buy", 50)
        R.check_quantity(600000, "buy", 200)

    def test_08_star_board_lot(self):
        """科创板 200 股起、超出部分可按 1 股递增。"""
        self.assertEqual(R.board_of(688981), R.STAR)
        with self.assertRaises(R.RuleError):
            R.check_quantity(688981, "buy", 199)
        R.check_quantity(688981, "buy", 200)
        R.check_quantity(688981, "buy", 201)

    def test_09_price_limit_differs_by_board(self):
        """主板 ±10%、科创板 ±20%。"""
        self.assertEqual(R.price_limits(600000, 10.00), (9.00, 11.00))
        self.assertEqual(R.price_limits(688981, 10.00), (8.00, 12.00))
        # 用对手价把价格笼子放开，单独检验涨跌幅限制
        with self.assertRaises(R.RuleError):
            R.check_price(600000, "buy", 11.10, 10.00, best_ask=11.10)
        R.check_price(688981, "buy", 11.10, 10.00, best_ask=11.10)

    def test_10_price_cage(self):
        """连续竞价价格笼子：买价不得高于卖一 ×102%。"""
        with self.assertRaises(R.RuleError):
            R.check_price(600000, "buy", 10.50, 10.00, best_bid=10.00, best_ask=10.00)
        R.check_price(600000, "buy", 10.20, 10.00, best_bid=10.00, best_ask=10.00)
        with self.assertRaises(R.RuleError):
            R.check_price(600000, "sell", 9.50, 10.00, best_bid=10.00, best_ask=10.00)


class TestJointReplay(unittest.TestCase):

    def test_11_aggressor_add_not_double_deducted(self):
        """委托与成交联合处理：只扣被动方，主动方的挂单记录不再重复扣减。"""
        ord_df = pd.DataFrame([
            {"Time": 100000000, "BizIndex": 1, "OrderKind": ORD_ADD,
             "OrderOriNo": 1, "FunctionCode": 66, "Price": 10.00, "Volume": 100},
            {"Time": 100000000, "BizIndex": 2, "OrderKind": ORD_ADD,
             "OrderOriNo": 2, "FunctionCode": 83, "Price": 10.00, "Volume": 100},
        ])
        exe_df = pd.DataFrame([
            {"Time": 100000000, "BizIndex": 1, "BSFlag": 66,
             "BidOrder": 1, "AskOrder": 2, "Price": 10.00, "Volume": 40},
        ])
        eng = ReplayEngine(600000, ord_df, exe_df, prev_close=10.0)
        eng.advance_to(100000000)
        self.assertEqual(eng.book.get(1).remaining, 100)   # 主动方不再被扣
        self.assertEqual(eng.book.get(2).remaining, 60)    # 被动方扣 40
        self.assertEqual(eng.book.levels[(BUY, 10.00)], 100)
        self.assertEqual(eng.book.levels[(SELL, 10.00)], 60)


if __name__ == "__main__":
    unittest.main(verbosity=2)
