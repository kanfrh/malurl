# -*- coding: utf-8 -*-
"""切分逻辑测试：分组的核心价值是测试集的域名不会在训练集里出现过。"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from malurl.data import overlap_rate, split_group, split_random


def _make_df(n_domains: int = 60, per_domain: int = 5) -> pd.DataFrame:
    rows = []
    for d in range(n_domains):
        host = "site%d.com" % d
        for k in range(per_domain):
            rows.append(("http://%s/path%d" % (host, k), d % 2))
    df = pd.DataFrame(rows, columns=["url", "label"])
    df["host"] = df["url"].map(lambda u: u.split("//")[1].split("/")[0])
    return df


def test_group_split_has_no_domain_overlap():
    df = _make_df()
    train, test = split_group(df, 0.2)
    assert overlap_rate(train, test) == 0.0


def test_random_split_usually_overlaps():
    df = _make_df()
    train, test = split_random(df, 0.2)
    # 同一域名被随机分到两边几乎是必然的
    assert overlap_rate(train, test) > 0.5


def test_group_split_keeps_both_classes():
    df = _make_df()
    train, test = split_group(df, 0.2)
    assert train["label"].nunique() == 2
    assert test["label"].nunique() == 2


def test_group_split_sizes():
    df = _make_df()
    train, test = split_group(df, 0.2)
    assert len(train) + len(test) == len(df)
    assert 0 < len(test) < len(df)
