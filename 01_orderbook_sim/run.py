#!/usr/bin/env python
"""订单簿回放与模拟下单原型 —— 本地启动入口。

示例::

    python run.py                                # 启动本地页面（自动定位数据目录）
    python run.py --data-dir D:\\sse50_20260923   # 指定数据目录
    python run.py --validate                     # 运行数据验证并生成报告
    python run.py --validate --sample 2400
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description="订单簿回放与模拟下单原型（本地运行）")
    ap.add_argument("--data-dir", default=None,
                    help="数据目录，需含 ord_*.parquet / exe_*.parquet / snp_*.parquet")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址（默认仅本机）")
    ap.add_argument("--port", type=int, default=8770, help="监听端口")
    ap.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    ap.add_argument("--validate", action="store_true", help="运行数据验证并生成报告")
    ap.add_argument("--sample", type=int, default=2400, help="验证抽样条数")
    args = ap.parse_args()

    from obsim import dataset

    try:
        data_dir = dataset.resolve_data_dir(args.data_dir)
    except FileNotFoundError as ex:
        print(f"[错误] {ex}", file=sys.stderr)
        return 2

    if args.validate:
        from obsim.validate import run as validate
        summary, path = validate(args.data_dir, args.sample)
        print(f"验证报告已生成: {path}")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return 0

    from obsim import server
    server.serve(str(data_dir), args.host, args.port, args.open)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

