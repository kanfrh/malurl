# -*- coding: utf-8 -*-
"""判定日志：每次请求落盘，是样本回流与重训练的原料。

格式：JSON Lines，一行一条，便于后续用 pandas 直接读。
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

_lock = threading.Lock()


def url_digest(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8", "ignore")).hexdigest()[:16]


def append(path: Path, records: list, keep_url: bool = True) -> None:
    """追加写入。批量请求会展开成多条记录。"""
    if not records:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    for r in records:
        item = {
            "ts": ts,
            "url_sha": url_digest(r["url"]),
            "score": round(float(r["score"]), 6),
            "label": r["label"],
            "risk": r["risk"],
            "model_version": r.get("model_version", ""),
            "source": r.get("source", "api"),
            # model / whitelist：回流时要能区分"模型判的"和"白名单放行的"
            "decision": r.get("decision", ""),
        }
        if keep_url:
            item["url"] = r["url"]
        lines.append(json.dumps(item, ensure_ascii=False))

    with _lock:
        with open(path, "a", encoding="utf-8") as fp:
            fp.write("\n".join(lines) + "\n")


def read_recent(path: Path, limit: int = 1000) -> list:
    """读最近 N 条，供重放与漂移分析使用。"""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as fp:
        lines = fp.readlines()[-limit:]
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out
