# -*- coding: utf-8 -*-
"""从公开真实数据集构建训练用的 urls.csv。

数据来源（二选一，任选其一即可，两者同源）：
  A. ISCX-URL-2016 派生集（本脚本默认格式，列名 url,label，label 为 good/bad）
     https://github.com/faizann24/Using-machine-learning-to-detect-malicious-URLs
  B. Kaggle Malicious URL Dataset（列名 url,type，type 为 benign/malicious）

用法：
    python scripts/prepare_real_data.py --raw data/raw/labeled.csv --out data/urls.csv
    python scripts/prepare_real_data.py --raw data/raw/kaggle.csv --out data/urls.csv --n-good 40000

处理逻辑：
  1. 去掉含空白、尖括号、引号的脏行
  2. 只保留带真实路径的 URL（形如 host/path，排除只有根路径的 host/）
     —— 否则"有没有路径"会变成泄漏特征
  3. 补全协议头、按 URL 去重、解析出 host
  4. 按原始标签比例分层抽样
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from malurl.features import host_of, normalize

DIRTY = re.compile(r"[\s<>\"']")
HAS_REAL_PATH = re.compile(r"/[^/]")


def has_real_path(url: str) -> bool:
    rest = re.sub(r"^[a-zA-Z]+://", "", str(url))
    return bool(HAS_REAL_PATH.search(rest))


def build(raw_path: Path, out_path: Path, n_good: int, n_bad: int, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(raw_path, dtype=str).dropna()
    url_col = "url" if "url" in df.columns else df.columns[0]
    label_col = "label" if "label" in df.columns else df.columns[-1]
    df = df[[url_col, label_col]]
    df.columns = ["url", "label"]
    df["url"] = df["url"].astype(str).str.strip()

    total = len(df)
    df = df[~df["url"].str.contains(DIRTY, regex=True, na=False)]
    after_dirty = len(df)

    # 统一标签
    lab = df["label"].str.lower().str.strip()
    df["label"] = lab.map({
        "good": 0, "benign": 0, "normal": 0, "0": 0,
        "bad": 1, "malicious": 1, "phishing": 1, "1": 1,
    })
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # 只保留带真实路径的样本，避免"有无路径"成为泄漏特征
    df = df[df["url"].map(has_real_path)]
    after_path = len(df)

    df["url"] = df["url"].map(normalize)
    df = df.drop_duplicates(subset=["url"])
    df["host"] = df["url"].map(host_of)
    df = df[df["host"] != ""].reset_index(drop=True)
    after_dedup = len(df)

    good = df[df.label == 0]
    bad = df[df.label == 1]
    take_good = min(n_good, len(good))
    take_bad = min(n_bad, len(bad))
    out = pd.concat([
        good.sample(n=take_good, random_state=seed),
        bad.sample(n=take_bad, random_state=seed),
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out[["url", "label"]].to_csv(out_path, index=False, encoding="utf-8-sig")

    print("原始行数            %d" % total)
    print("去掉脏字符后        %d" % after_dirty)
    print("带真实路径的        %d" % after_path)
    print("去重且能解析出域名  %d" % after_dedup)
    print("  其中正常 %d / 恶意 %d" % (len(good), len(bad)))
    print("抽样输出            %d（正常 %d / 恶意 %d）" % (len(out), take_good, take_bad))
    print("唯一域名数          %d" % out["host"].nunique())
    print("已保存              %s" % out_path)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="原始 CSV 路径")
    ap.add_argument("--out", default="data/urls.csv")
    ap.add_argument("--n-good", type=int, default=40000)
    ap.add_argument("--n-bad", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    build(Path(args.raw), Path(args.out), args.n_good, args.n_bad, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
