"""数据包加载：自动识别文件名日期，数据目录可通过 --data-dir 配置。"""
from __future__ import annotations

import os
import glob
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

_KINDS = ("ord", "exe", "snp")


def _pick(folder: Path, kind: str) -> Path:
    hits = sorted(glob.glob(str(folder / f"{kind}_*.parquet")))
    if not hits:
        hits = sorted(glob.glob(str(folder / f"{kind}*.parquet")))
    if not hits:
        raise FileNotFoundError(f"在 {folder} 下找不到 {kind}_*.parquet")
    return Path(hits[-1])


def resolve_data_dir(data_dir: str | None = None) -> Path:
    """按优先级定位数据目录：显式参数 > 环境变量 > 项目同级目录 > 向上搜索。"""
    if data_dir:
        p = Path(data_dir).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"数据目录不存在: {p}")
        return p
    env = os.environ.get("OBSIM_DATA_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
    here = Path(__file__).resolve()
    for base in [here.parents[2], here.parents[1], Path.cwd()]:
        for cand in sorted(base.glob("sse50_*")):
            if cand.is_dir() and glob.glob(str(cand / "ord_*.parquet")):
                return cand
        if glob.glob(str(base / "ord_*.parquet")):
            return base
    # 向上逐级搜索
    for base in [here.parents[2], here.parents[3]]:
        for cand in base.rglob("ord_*.parquet"):
            return cand.parent
    raise FileNotFoundError("未找到数据目录，请用 --data-dir 指定")


def files(data_dir: str | None = None) -> dict:
    folder = resolve_data_dir(data_dir)
    return {"dir": folder, **{k: _pick(folder, k) for k in _KINDS}}


def _read(path, symbol=None) -> pd.DataFrame:
    if symbol is None:
        return pq.read_table(path).to_pandas()
    try:
        tbl = pq.read_table(path, filters=[("Symbol", "=", int(symbol))])
    except Exception:
        tbl = pq.read_table(path)
        tbl = tbl.filter(__import__("pyarrow").compute.equal(tbl["Symbol"], int(symbol)))
    return tbl.to_pandas()


def load_ord(data_dir: str | None = None, symbol=None) -> pd.DataFrame:
    return _read(files(data_dir)["ord"], symbol)


def load_exe(data_dir: str | None = None, symbol=None) -> pd.DataFrame:
    return _read(files(data_dir)["exe"], symbol)


def load_snp(data_dir: str | None = None, symbol=None) -> pd.DataFrame:
    return _read(files(data_dir)["snp"], symbol)


def list_symbols(data_dir: str | None = None):
    """只读 Symbol 列，快速列出标的。"""
    import pyarrow.compute as pc
    col = pq.read_table(files(data_dir)["snp"], columns=["Symbol"])["Symbol"]
    return sorted(int(x) for x in pc.unique(col).to_pylist())


def session_bounds(data_dir: str | None = None, symbol=None):
    """返回该标的回放的时间范围与昨收。"""
    snp = load_snp(data_dir, symbol)
    t0 = int(snp["Time"].min())
    t1 = int(snp["Time"].max())
    pc = snp.loc[snp["PreClose"] > 0, "PreClose"]
    return t0, t1, (float(pc.iloc[0]) if len(pc) else None)


def load_all(data_dir: str | None = None):
    f = files(data_dir)
    return (pq.read_table(f["ord"]).to_pandas(),
            pq.read_table(f["exe"]).to_pandas(),
            pq.read_table(f["snp"]).to_pandas())
