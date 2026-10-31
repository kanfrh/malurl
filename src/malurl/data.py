# -*- coding: utf-8 -*-
"""数据加载、清洗与切分。

这里有一个关键设计：**分组切分**。
同一个域名下的 URL 高度相似，随机切分会让训练集和测试集共享域名，
模型等于背答案，指标虚高。按 host 分组切分才是可信的评估方式。
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .features import host_of

URL_KEYS = ["url", "URL", "Url", "网址", "link", "Link"]
LABEL_KEYS = ["label", "Label", "标签", "class", "type", "result", "malicious"]

LABEL_MAP = {
    "malicious": 1, "phishing": 1, "bad": 1, "1": 1, "true": 1, "恶意": 1,
    "benign": 0, "good": 0, "normal": 0, "0": 0, "false": 0, "正常": 0,
}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """读取 CSV，自动识别 URL 列与标签列，返回规范化的 DataFrame。

    结果列：url / label / host
    """
    df = pd.read_csv(path)
    url_col = next((c for c in df.columns if c in URL_KEYS), df.columns[0])
    label_col = next((c for c in df.columns if c in LABEL_KEYS),
                     df.columns[1] if len(df.columns) > 1 else df.columns[-1])

    df = df[[url_col, label_col]].dropna()
    df.columns = ["url", "label"]
    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"] != ""]

    if not pd.api.types.is_numeric_dtype(df["label"]):
        df["label"] = df["label"].astype(str).str.lower().str.strip().map(LABEL_MAP)
        df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    df["host"] = df["url"].map(host_of)
    df = df[df["host"] != ""].reset_index(drop=True)
    return df


def split_random(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """随机分层切分（大多数论文用的方式，存在域名泄漏）。"""
    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def split_group(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """按域名分组切分：同一域名只会出现在训练集或测试集之一。

    这是本项目的核心评估改进，也是论文里最值得写的一段。
    """
    if df["host"].nunique() < 2:
        raise ValueError(
            "数据集中可用域名不足 2 个，无法做分组切分；"
            "请检查 URL 列是否完整（例如是否缺少 http:// 前缀导致解析失败）。")

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, df["label"], groups=df["host"]))
    train = df.iloc[train_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    # 极端情况下某一类可能没进测试集，退回随机切分避免训练流程中断
    if test["label"].nunique() < 2 or train["label"].nunique() < 2:
        return split_random(df, test_size, seed)
    return train, test


def overlap_rate(train: pd.DataFrame, test: pd.DataFrame) -> float:
    """测试集里有多少比例的样本，其域名在训练集中出现过。"""
    seen = set(train["host"])
    hit = test["host"].isin(seen).sum()
    return float(hit) / max(len(test), 1)
