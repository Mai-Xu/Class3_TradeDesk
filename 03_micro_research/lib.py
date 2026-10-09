# -*- coding: utf-8 -*-
"""Shared helpers for the SSE50 Level-2 microstructure research."""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"

# 项目根目录 = 本文件所在目录(不再写死本机路径)
ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve_base() -> str:
    """定位数据目录：环境变量 SSE50_DATA_DIR/OBSIM_DATA_DIR > 项目同级 sse50_* > 当前目录。

    原始行情数据由课程 Release 下载，仓库内不重复包含。
    """
    for var in ("SSE50_DATA_DIR", "OBSIM_DATA_DIR"):
        p = os.environ.get(var)
        if p and os.path.isdir(p):
            return os.path.abspath(p)
    for base in (os.path.dirname(ROOT), ROOT, os.getcwd()):
        cand = sorted(d for d in glob.glob(os.path.join(base, "sse50_*"))
                      if os.path.isdir(d))
        if cand:
            return cand[-1]
        if glob.glob(os.path.join(base, "ord_*.parquet")):
            return os.path.abspath(base)
    for hit in sorted(glob.glob(os.path.join(os.path.dirname(ROOT), "**", "ord_*.parquet"),
                                 recursive=True)):
        return os.path.dirname(os.path.abspath(hit))
    raise SystemExit("找不到数据目录。请设置环境变量 SSE50_DATA_DIR，"
                     "或把 sse50_20260923 放在本项目同级目录下。")


def _pick(kind: str) -> str:
    for pat in (f"{kind}_*.parquet", f"{kind}*.parquet"):
        hits = sorted(glob.glob(os.path.join(BASE, pat)))
        if hits:
            return hits[-1]
    raise SystemExit(f"{BASE} 下找不到 {kind}_*.parquet")


BASE = _resolve_base()
ORD_PATH = _pick("ord")
EXE_PATH = _pick("exe")
SNP_PATH = _pick("snp")
FIG = os.path.join(ROOT, "out", "figures")
TAB = os.path.join(ROOT, "out", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# A-share tick size
TICK = 0.01

# SSE50 constituents (2026-09-23 sample). Names best-known.
NAMES = {
    600028: "中国石化", 600030: "中信证券", 600031: "三一重工", 600036: "招商银行",
    600050: "中国联通", 600089: "特变电工", 600111: "北方稀土", 600150: "中国船舶",
    600183: "生益科技", 600276: "恒瑞医药", 600309: "万华化学", 600406: "国电南瑞",
    600519: "贵州茅台", 600760: "中航沈飞", 600809: "山西汾酒", 600887: "伊利股份",
    600900: "长江电力", 600930: "华电新能", 601012: "隆基绿能", 601088: "中国神华",
    601127: "赛力斯", 601166: "兴业银行", 601211: "国泰海通", 601288: "农业银行",
    601318: "中国平安", 601328: "交通银行", 601398: "工商银行", 601600: "中国铝业",
    601601: "中国太保", 601628: "中国人寿", 601658: "邮储银行", 601668: "中国建筑",
    601688: "华泰证券", 601728: "中国电信", 601857: "中国石油", 601888: "中国中免",
    601899: "紫金矿业", 601919: "中远海控", 601988: "中国银行", 603019: "中科曙光",
    603259: "药明康德", 603501: "韦尔股份", 603986: "兆易创新", 603993: "洛阳钼业",
    688008: "澜起科技", 688012: "中微公司", 688041: "海光信息", 688111: "金山办公",
    688256: "寒武纪", 688981: "中芯国际",
}

# Rough sector tags for the linkage network
SECTOR = {
    601398: "银行", 601939: "银行", 601288: "银行", 601988: "银行", 601328: "银行",
    601658: "银行", 600036: "银行", 601166: "银行",
    600030: "券商", 601688: "券商", 601211: "券商",
    601601: "保险", 601628: "保险", 601318: "保险",
    600519: "白酒食品", 600809: "白酒食品", 600887: "白酒食品",
    600028: "石油石化", 601857: "石油石化",
    600900: "电力公用", 601088: "煤炭电力", 600930: "电力公用", 600406: "电力设备",
    600089: "电力设备", 601012: "电力设备",
    601899: "有色", 601600: "有色", 603993: "有色", 600111: "有色",
    688981: "半导体", 688041: "半导体", 688256: "半导体", 688111: "软件", 688012: "半导体",
    688008: "半导体", 603501: "半导体", 603986: "半导体",
    600276: "医药", 603259: "医药",
    601888: "消费", 601127: "汽车", 600031: "机械", 600760: "军工", 601668: "建筑",
    601919: "航运", 600050: "通信", 601728: "通信", 600309: "化工", 600183: "电子",
    600150: "军工", 603019: "计算机",
}


def name(sym):
    return NAMES.get(int(sym), str(sym))


def label(sym):
    return f"{int(sym)} {name(sym)}"


def to_sec(t):
    """HHMMSSmmm int -> seconds after 00:00:00 (float)."""
    t = t.astype("int64")
    h = t // 10_000_000
    m = (t // 100_000) % 100
    s = (t // 1000) % 100
    ms = t % 1000
    return h * 3600 + m * 60 + s + ms / 1000.0


def load_ord(cols=None):
    return pd.read_parquet(ORD_PATH, columns=cols)


def load_exe(cols=None):
    return pd.read_parquet(EXE_PATH, columns=cols)


def load_snp(cols=None):
    return pd.read_parquet(SNP_PATH, columns=cols)


def depth_cols(side, n=10):
    return [f"{'Bid' if side=='b' else 'Ask'}Volume{i}" for i in range(1, n + 1)] + \
           [f"{'Bid' if side=='b' else 'Ask'}Price{i}" for i in range(1, n + 1)]


def savefig(fig, fname):
    p = os.path.join(FIG, fname)
    fig.savefig(p)
    plt.close(fig)
    return p


def savetab(df, fname, index=False):
    p = os.path.join(TAB, fname)
    df.to_csv(p, index=index, encoding="utf-8-sig")
    return p


def mid(row_bid, row_ask):
    m = (row_bid + row_ask) / 2.0
    return m
