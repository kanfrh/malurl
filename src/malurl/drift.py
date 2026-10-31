# -*- coding: utf-8 -*-
"""概念漂移监控：PSI（群体稳定性指数）。

PSI 衡量"线上特征分布"与"训练时特征分布"的差异：
    < 0.1   稳定
    0.1~0.25 有变化，需要关注
    > 0.25  显著漂移，应触发重训练
"""
from __future__ import annotations

import numpy as np

from .features import FEATURE_CN, FEATURE_NAMES


def compute_psi(reference: dict, X: np.ndarray) -> list:
    """返回每个特征的 PSI 值。"""
    out = []
    for j, (edges, ref_pct) in enumerate(zip(reference["edges"], reference["ref_pct"])):
        e = np.asarray(edges, dtype=float)
        if len(e) < 2:
            out.append(0.0)
            continue
        # 分箱边界是训练集的 min~max。线上样本可能落在范围之外，
        # np.histogram 对范围外的值直接丢弃，会让"全都跑出训练范围"的严重
        # 漂移被算成稳定。把首尾边界扩成无穷，让这些值落进首/尾箱。
        e = e.copy()
        if np.isfinite(e[0]):
            e[0] = -np.inf
        if np.isfinite(e[-1]):
            e[-1] = np.inf
        counts, _ = np.histogram(X[:, j], bins=e)
        cur = (counts + 1e-6) / (counts.sum() + 1e-6 * len(counts))
        ref = np.asarray(ref_pct, dtype=float)
        if len(cur) != len(ref):
            cur, ref = cur[:min(len(cur), len(ref))], ref[:min(len(cur), len(ref))]
        out.append(float(np.sum((cur - ref) * np.log(cur / ref))))
    return out


def drift_report(reference: dict, X: np.ndarray, top_k: int = 8) -> dict:
    values = compute_psi(reference, X)
    pairs = sorted(zip(FEATURE_NAMES, values), key=lambda kv: kv[1], reverse=True)
    max_psi = pairs[0][1] if pairs else 0.0
    if max_psi > 0.25:
        level, advice = "显著漂移", "建议立即重训练"
    elif max_psi > 0.10:
        level, advice = "轻度漂移", "持续观察，准备重训练"
    else:
        level, advice = "稳定", "无需处理"
    return {
        "max_psi": round(max_psi, 4),
        "level": level,
        "advice": advice,
        "top": [{"feature": f, "name_cn": FEATURE_CN.get(f, f), "psi": round(v, 4)}
                for f, v in pairs[:top_k]],
    }
