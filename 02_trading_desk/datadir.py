"""数据目录定位（不依赖个人电脑路径）。

优先级：命令行 --data-dir > 环境变量 SSE50_DATA_DIR / OBSIM_DATA_DIR
        > 本项目同级 `sse50_*` 目录 > 当前工作目录 > 向上搜索 ord_*.parquet。
原始行情数据由课程 Release 下载，仓库内不重复包含。
"""
from __future__ import annotations

import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_VARS = ("SSE50_DATA_DIR", "OBSIM_DATA_DIR")


def resolve(argv=None, start=None) -> str:
    argv = list(argv) if argv is not None else []
    if "--data-dir" in argv:
        i = argv.index("--data-dir")
        if i + 1 < len(argv):
            p = os.path.abspath(os.path.expanduser(argv[i + 1]))
            if not os.path.isdir(p):
                raise SystemExit(f"--data-dir 不存在: {p}")
            return p
    for var in ENV_VARS:
        p = os.environ.get(var)
        if p and os.path.isdir(p):
            return os.path.abspath(p)
    roots = [start or os.path.dirname(HERE), HERE, os.getcwd()]
    for base in roots:
        cand = [d for d in glob.glob(os.path.join(base, "sse50_*")) if os.path.isdir(d)]
        if cand:
            return sorted(cand)[-1]
        if glob.glob(os.path.join(base, "ord_*.parquet")):
            return os.path.abspath(base)
    for base in roots[:1]:
        for hit in sorted(glob.glob(os.path.join(base, "**", "ord_*.parquet"), recursive=True)):
            return os.path.dirname(os.path.abspath(hit))
    raise SystemExit(
        "找不到数据目录。请用 --data-dir 指定，或设置环境变量 SSE50_DATA_DIR，\n"
        "或把 sse50_20260923 放在本项目同级目录下。")


def pick(data_dir: str, kind: str) -> str:
    """按 ord_/exe_/snp_ 前缀匹配数据文件（文件名日期自动识别）。"""
    for pat in (f"{kind}_*.parquet", f"{kind}*.parquet"):
        hits = sorted(glob.glob(os.path.join(data_dir, pat)))
        if hits:
            return hits[-1]
    raise SystemExit(f"{data_dir} 下找不到 {kind}_*.parquet")

