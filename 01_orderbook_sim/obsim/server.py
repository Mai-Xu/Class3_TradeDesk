"""本地 HTTP 服务 + 浏览器操作页面（仅绑定 127.0.0.1）。"""
from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

from . import dataset
from .account import Account
from .engine import ReplayEngine, TradingSession

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


class SimState:
    """持有当前会话；引擎按标的小容量缓存，避免反复读盘。"""

    def __init__(self, data_dir: str | None = None, cash: float = 1_000_000.0):
        self.data_dir = data_dir
        self.cash = cash
        self.cache: OrderedDict[int, ReplayEngine] = OrderedDict()
        self.cache_cap = 4
        self.symbol = None
        self.engine = None
        self.session = None
        self.lock = threading.RLock()

    def _engine(self, symbol: int) -> ReplayEngine:
        if symbol in self.cache:
            self.cache.move_to_end(symbol)
            return self.cache[symbol]
        od = dataset.load_ord(self.data_dir, symbol)
        ex = dataset.load_exe(self.data_dir, symbol)
        _, _, prev = dataset.session_bounds(self.data_dir, symbol)
        eng = ReplayEngine(symbol, od, ex, prev)
        self.cache[symbol] = eng
        while len(self.cache) > self.cache_cap:
            self.cache.popitem(last=False)
        return eng

    def select(self, symbol: int, t0: int | None = None, cash: float | None = None):
        with self.lock:
            if cash is not None:
                self.cash = float(cash)
            eng = self._engine(int(symbol))
            eng.reset()                       # 回到开盘前的空簿
            self.symbol = int(symbol)
            self.engine = eng
            self.session = TradingSession(eng, Account(self.cash))
            if t0:
                eng.advance_to(int(t0))
            return self.state()

    def seek(self, t: int):
        with self.lock:
            if self.engine is None:
                raise RuntimeError("请先选择标的")
            t = int(t)
            if t < self.engine.time:
                # 回退：清空后重放
                eng = self._engine(self.symbol)
                eng.reset()
                self.session = TradingSession(eng, Account(self.cash))
            self.engine.advance_to(t)
            return self.state()

    def state(self):
        if self.session is None:
            return {"ready": False}
        st = self.session.state()
        st["ready"] = True
        st["prev_close"] = self.engine.prev_close
        return st

    def order(self, side, price, qty):
        with self.lock:
            if self.session is None:
                raise RuntimeError("请先选择标的")
            rec = self.session.place(side, float(price), int(qty))
            st = self.state()
            st["result"] = rec.as_dict()
            return st

    def cancel(self, oid):
        with self.lock:
            if self.session is None:
                raise RuntimeError("请先选择标的")
            rec = self.session.cancel(int(oid))
            st = self.state()
            st["result"] = rec.as_dict() if rec else None
            return st

    def settle(self):
        with self.lock:
            if self.session is None:
                raise RuntimeError("请先选择标的")
            self.session.account.settle()
            return self.state()


class Handler(BaseHTTPRequestHandler):
    state: SimState = None           # 由 serve() 注入

    server_version = "obsim/1.0"

    def log_message(self, fmt, *args):   # 静音
        pass

    # ---------------- 工具 ----------------
    def _send(self, code, body: bytes, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    # ---------------- 路由 ----------------
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                p = os.path.join(WEB_DIR, "index.html")
                with open(p, "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if u.path == "/api/symbols":
                syms = dataset.list_symbols(self.state.data_dir)
                return self._json({"symbols": syms, "count": len(syms),
                                   "data_dir": str(dataset.resolve_data_dir(self.state.data_dir))})
            if u.path == "/api/state":
                if "symbol" in q and (self.state.symbol != int(q["symbol"][0])
                                      or self.state.session is None):
                    t0 = int(q["time"][0]) if "time" in q else None
                    return self._json(self.state.select(int(q["symbol"][0]), t0))
                return self._json(self.state.state())
            return self._json({"error": "not found"}, 404)
        except Exception as e:
            return self._json({"error": str(e)}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        b = self._body()
        try:
            if u.path == "/api/select":
                return self._json(self.state.select(int(b["symbol"]),
                                                    b.get("time"), b.get("cash")))
            if u.path == "/api/seek":
                return self._json(self.state.seek(b["time"]))
            if u.path == "/api/order":
                return self._json(self.state.order(b["side"], b["price"], b["qty"]))
            if u.path == "/api/cancel":
                return self._json(self.state.cancel(b["oid"]))
            if u.path == "/api/settle":
                return self._json(self.state.settle())
            return self._json({"error": "not found"}, 404)
        except Exception as e:
            return self._json({"error": str(e)}, 400)


def serve(data_dir=None, host="127.0.0.1", port=8770, open_browser=False):
    Handler.state = SimState(data_dir)
    httpd = ThreadingHTTPServer((host, port), Handler)
    url = f"http://{host}:{port}/"
    print(f"订单簿回放与模拟下单原型已启动: {url}")
    print(f"数据目录: {dataset.resolve_data_dir(data_dir)}")
    if open_browser:
        import webbrowser
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        httpd.server_close()
