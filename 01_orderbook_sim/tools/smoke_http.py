"""浏览器接口冒烟检查：在本机起服务并依次调用页面用到的全部接口。

运行: python tools/smoke_http.py [--data-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from obsim import server  # noqa: E402


def call(base, path, body=None):
    url = base + path
    if body is None:
        req = urllib.request.Request(url)
    else:
        req = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    server.Handler.state = server.SimState(args.data_dir)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    port = httpd.server_address[1]
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"
    checks = []

    def ck(name, cond, detail=""):
        checks.append((name, bool(cond), detail))
        print(f"{'PASS' if cond else 'FAIL'}  {name}  {detail}")

    try:
        info = call(base, "/api/symbols")
        ck("GET /api/symbols", info.get("count") == 50, f"count={info.get('count')}")

        sym = info["symbols"][0]
        st = call(base, "/api/select", {"symbol": sym, "time": 100000000})
        ck("POST /api/select", st.get("time") == 100000000,
           f"symbol={sym} last={st['market']['last']} cum={st['market']['cum_volume']}")
        ck("十档盘口完整", len(st["market"]["bids"]) == 10 and len(st["market"]["asks"]) == 10,
           f"bids={len(st['market']['bids'])} asks={len(st['market']['asks'])}")

        ask1 = st["market"]["asks"][-1]["price"]
        bid1 = st["market"]["bids"][0]["price"]

        r = call(base, "/api/order", {"side": "buy", "price": ask1, "qty": 100})
        ck("POST /api/order 买入吃单", r["result"]["status"] == "全部成交",
           f"{r['result']['status']} filled={r['result']['filled']} fee={r['result']['fees']}")
        ck("资金扣减、今仓入账",
           r["account"]["cash"] < 1_000_000 and r["account"]["holdings"][0]["today"] == 100,
           f"cash={r['account']['cash']} today={r['account']['holdings'][0]['today']}")

        r2 = call(base, "/api/order", {"side": "sell", "price": bid1, "qty": 100})
        ck("T+1 当日买入不可卖", r2["result"]["status"] == "已拒绝",
           r2["result"]["reason"])

        r3 = call(base, "/api/order", {"side": "buy", "price": round(ask1 * 0.5, 2), "qty": 100})
        ck("涨跌幅校验", r3["result"]["status"] == "已拒绝", r3["result"]["reason"])

        r4 = call(base, "/api/order", {"side": "buy", "price": bid1, "qty": 500})
        ck("挂单冻结资金", r4["result"]["status"] in ("已报", "部分成交")
           and r4["account"]["frozen_cash"] > 0,
           f"oid={r4['result']['oid']} {r4['result']['status']} frozen={r4['account']['frozen_cash']}")

        r5 = call(base, "/api/cancel", {"oid": r4["result"]["oid"]})
        ck("POST /api/cancel 解冻", r5["result"]["status"] == "已撤单",
           f"frozen={r5['account']['frozen_cash']}")

        r6 = call(base, "/api/settle", {})
        ck("POST /api/settle T+1 转可卖",
           r6["account"]["holdings"][0]["sellable"] >= 100,
           f"sellable={r6['account']['holdings'][0]['sellable']}")

        r7 = call(base, "/api/seek", {"time": 100030000})
        ck("POST /api/seek 前进", r7["time"] == 100030000, f"last={r7['market']['last']}")
        r8 = call(base, "/api/seek", {"time": 100000000})
        ck("POST /api/seek 回退重放", r8["time"] == 100000000,
           f"last={r8['market']['last']} cum={r8['market']['cum_volume']}")

        with urllib.request.urlopen(base + "/", timeout=30) as resp:
            html = resp.read().decode("utf-8")
        ck("GET / 返回操作页面", resp.status == 200 and "模拟下单" in html,
           f"{len(html)} bytes")

        r9 = call(base, "/api/order", {"side": "buy", "price": ask1, "qty": 150})
        ck("主板申报数量校验(非100整数倍被拒)",
           r9["result"]["status"] == "已拒绝", r9["result"]["reason"])
        r10 = call(base, "/api/cancel", {"oid": 999999})
        ck("撤不存在的委托不报错", r10.get("result") is None, "result=None")
    finally:
        httpd.shutdown()

    bad = [c for c in checks if not c[1]]
    print(f"\n通过 {len(checks) - len(bad)} / {len(checks)}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
