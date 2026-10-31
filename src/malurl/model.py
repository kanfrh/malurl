# -*- coding: utf-8 -*-
"""训练、评估、模型持久化。"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .features import FEATURE_NAMES, extract_many
from .data import overlap_rate, split_group, split_random

METRIC_COLS = ["accuracy", "precision", "recall", "f1", "auc"]


def build_models() -> dict:
    """候选模型。

    注意：随机森林和决策树**不加** class_weight。
    实测（分组切分、41 维特征）加权的 F1 是 0.8123，不加权是 0.8276——
    加权虽然提高召回，但把精确率从 85.9% 压到 75.2%，整体反而更差。
    类别不平衡的影响通过调阈值来消化，比在训练时强行加权更可控。
    """
    models = {
        "逻辑回归": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]),
        "决策树": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=3, random_state=42),
        "随机森林": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=1, n_jobs=-1, random_state=42),
        # 直方图梯度提升：与 GBDT 同族，但分箱实现，速度快一个数量级
        "梯度提升树": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=42),
    }

    # XGBoost 是可选依赖：装了才参与对比，没装不影响其他模型
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=42, n_jobs=1)
    except ImportError:
        pass

    return models


def predict_proba_safe(model, X):
    """并行推理失败时自动降级为单线程（受限环境下的兜底）。"""
    try:
        return model.predict_proba(X)
    except Exception:
        try:
            model.set_params(n_jobs=1)
        except Exception:
            pass
        return model.predict_proba(X)


def evaluate(model, X_test, y_test) -> dict:
    proba = predict_proba_safe(model, X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "auc": float(roc_auc_score(y_test, proba)),
        "confusion": confusion_matrix(y_test, pred).tolist(),
    }


def compare_splits(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """在随机切分与分组切分上各训练一遍，输出对比表。

    这张表是论文里最有说服力的一张：说明随机切分高估了多少。
    """
    rows = []
    for mode, splitter in (("随机切分", split_random), ("分组切分", split_group)):
        train, test = splitter(df, 0.2, seed)
        X_train = np.asarray(extract_many(train["url"]), dtype=float)
        X_test = np.asarray(extract_many(test["url"]), dtype=float)
        model = build_models()["随机森林"]
        model.fit(X_train, train["label"].values)
        m = evaluate(model, X_test, test["label"].values)
        m["split"] = mode
        m["domain_overlap"] = round(overlap_rate(train, test), 4)
        m["n_train"] = len(train)
        m["n_test"] = len(test)
        rows.append(m)
    return pd.DataFrame(rows)[
        ["split", "accuracy", "precision", "recall", "f1", "auc", "domain_overlap", "n_train", "n_test"]]


def build_psi_reference(X: np.ndarray) -> dict:
    """保存训练集各特征的分布，供漂移监控使用。"""
    edges, ref = [], []
    for j in range(X.shape[1]):
        qs = np.unique(np.quantile(X[:, j], np.linspace(0, 1, 11)))
        if len(qs) < 2:
            qs = np.array([qs[0] - 0.5, qs[0] + 0.5])
        counts, _ = np.histogram(X[:, j], bins=qs)
        pct = (counts + 1e-6) / (counts.sum() + 1e-6 * len(counts))
        edges.append(qs.tolist())
        ref.append(pct.tolist())
    return {"edges": edges, "ref_pct": ref}


def train(df: pd.DataFrame, out_dir: Path, seed: int = 42,
          data_file: str = "") -> dict:
    """用分组切分训练全部模型，按规则选优，保存模型与元信息。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = split_group(df, 0.2, seed)
    X_train = np.asarray(extract_many(train_df["url"]), dtype=float)
    X_test = np.asarray(extract_many(test_df["url"]), dtype=float)
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    rows, fitted = [], {}
    for name, model in build_models().items():
        t0 = time.time()
        model.fit(X_train, y_train)
        m = evaluate(model, X_test, y_test)
        m["model"] = name
        m["train_sec"] = round(time.time() - t0, 2)

        # 单条推理延迟：先预热，再测 50 次取平均
        probe = X_test[:1]
        predict_proba_safe(model, probe)
        t1 = time.perf_counter()
        for _ in range(50):
            predict_proba_safe(model, probe)
        m["latency_ms"] = round((time.perf_counter() - t1) / 50 * 1000, 2)

        rows.append(m)
        fitted[name] = model

    result = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)

    # 选型规则（v3）：先按 F1 排序，再在"与第一名差距小于 0.002（噪声范围内）"的候选里
    # 取推理最快的那一个。理由：0.002 以内的 F1 差异没有统计意义，
    # 但延迟差异可能是十几倍——实测 HistGB 与 XGBoost 差 0.0008，延迟却差 8 倍。
    # 可解释性的理由已不成立，选型回归"指标优先、体积与延迟作为并列判据"。
    best_row = result.iloc[0]
    best_name = best_row["model"]
    reason = "F1 最优"
    if len(result) > 1:
        top_f1 = float(best_row["f1"])
        close = result[top_f1 - result["f1"] < 0.002]
        if len(close) > 1:
            fastest = close.sort_values("latency_ms").iloc[0]
            if fastest["model"] != best_name:
                reason = ("%s 的 F1 只低 %.4f（噪声范围内），但单条推理快 %.1f 倍，故选用它"
                          % (fastest["model"],
                             top_f1 - float(fastest["f1"]),
                             float(best_row["latency_ms"]) / max(float(fastest["latency_ms"]), 0.01)))
                best_name = fastest["model"]
            else:
                reason = "F1 最优，且在 F1 相近的候选里推理最快"
        else:
            second = result.iloc[1]
            reason = "F1 最优，领先第二名 %s %.4f" % (
                second["model"], top_f1 - float(second["f1"]))

    model = fitted[best_name]
    best_row = result[result["model"] == best_name].iloc[0]
    version = time.strftime("%Y%m%d-%H%M%S")

    joblib.dump({"model": model, "feature_names": FEATURE_NAMES, "model_name": best_name,
                 "version": version, "threshold": 0.5},
                out_dir / "model.joblib")

    info = {
        "model_name": best_name,
        "version": version,
        "select_reason": reason,
        "threshold": 0.5,
        "data_file": data_file or "unknown",
        "metrics": {k: float(best_row[k]) for k in METRIC_COLS},
        "confusion": best_row["confusion"],
        "split": "group",
        "domain_overlap": round(overlap_rate(train_df, test_df), 4),
        "n_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out_dir / "model_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    result.drop(columns=["confusion"]).to_csv(
        out_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")

    joblib.dump(build_psi_reference(X_train), out_dir / "psi_reference.joblib")
    return info


def load(model_path: str | Path) -> dict:
    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict):
        bundle = {"model": bundle, "model_name": type(bundle).__name__,
                  "version": "unknown", "feature_names": FEATURE_NAMES}
    return bundle
