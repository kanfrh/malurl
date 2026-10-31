# -*- coding: utf-8 -*-
"""接口测试：覆盖鉴权、批量上限、超长 URL、健康检查与指标。

注意：环境变量必须在导入 app 之前设置，因为配置在模块导入时就被读取了。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from malurl.features import FEATURE_NAMES, extract_features  # noqa: E402

TMP = Path(__file__).resolve().parent / "_tmp"
TMP.mkdir(parents=True, exist_ok=True)


def _make_model() -> None:
    rows = []
    for i in range(80):
        rows.append(("http://phish%d.tk/login" % i, 1))
        rows.append(("https://www.site%d.com/index.html" % i, 0))
    X = np.asarray([extract_features(u) for u, _ in rows], dtype=float)
    y = np.asarray([lab for _, lab in rows])
    model = RandomForestClassifier(n_estimators=20, random_state=0, n_jobs=1).fit(X, y)
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES,
                 "model_name": "随机森林", "version": "test-1", "threshold": 0.5},
                TMP / "model.joblib")


_make_model()
os.environ["MALURL_MODEL_PATH"] = str(TMP / "model.joblib")
os.environ["MALURL_LOG_PATH"] = str(TMP / "judgements.jsonl")
os.environ["MALURL_API_KEY"] = "test-key"
os.environ["MALURL_MAX_BATCH"] = "3"

from fastapi.testclient import TestClient  # noqa: E402

from malurl.service.api import app  # noqa: E402

HEADERS = {"X-API-Key": "test-key"}


@pytest.fixture(scope="module")
def client():
    # 用上下文管理器才会触发 lifespan（加载模型）
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] == "test-1"


def test_predict_single(client):
    r = client.post("/predict", json={"urls": ["http://paypal.tk/login"]}, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    item = body["results"][0]
    assert item["label"] in ("malicious", "benign")
    assert item["risk"] in ("high", "medium", "low")
    assert 0.0 <= item["score"] <= 1.0
    assert item["source"] in ("model", "whitelist")


def test_whitelist_bypasses_model(client):
    """白名单域名直接放行，且不经过模型（score 恒为 0）。"""
    r = client.post("/predict",
                    json={"urls": ["https://www.icbc.com.cn/login"]}, headers=HEADERS)
    assert r.status_code == 200
    item = r.json()["results"][0]
    assert item["source"] == "whitelist"
    assert item["label"] == "benign"
    assert item["risk"] == "low"
    assert item["score"] == 0.0


def test_whitelist_does_not_match_lookalike(client):
    """钓鱼常用手法：把可信域名放在子域或路径里，不能被白名单放行。"""
    r = client.post("/predict", json={"urls": [
        "http://icbc.com.cn.evil.tk/login",
        "http://fakeicbc.com.cn/login",
    ]}, headers=HEADERS)
    body = r.json()
    for item in body["results"]:
        assert item["source"] == "model"


def test_predict_batch(client):
    r = client.post("/predict", json={"urls": [
        "http://a.tk/login", "https://www.example.com/index.html"]}, headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["count"] == 2


def test_auth_required(client):
    r = client.post("/predict", json={"urls": ["http://a.tk/"]})
    assert r.status_code == 401


def test_batch_limit(client):
    r = client.post("/predict", json={"urls": ["http://a.tk/"] * 4}, headers=HEADERS)
    assert r.status_code == 422


def test_url_too_long(client):
    r = client.post("/predict", json={"urls": ["http://a.com/" + "x" * 5000]},
                    headers=HEADERS)
    assert r.status_code == 422


def test_empty_list_rejected(client):
    r = client.post("/predict", json={"urls": []}, headers=HEADERS)
    assert r.status_code == 422


def test_malformed_payload(client):
    r = client.post("/predict", json={"foo": "bar"}, headers=HEADERS)
    assert r.status_code == 422


def test_metrics_endpoint(client):
    client.post("/predict", json={"urls": ["http://a.tk/"]}, headers=HEADERS)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "malurl_requests_total" in r.text
    assert "malurl_score" in r.text


def test_judgement_log_written(client):
    log_path = Path(os.environ["MALURL_LOG_PATH"])
    if log_path.exists():
        log_path.unlink()
    client.post("/predict", json={"urls": ["http://log-test.tk/login"]}, headers=HEADERS)
    assert log_path.exists()
    content = log_path.read_text(encoding="utf-8").strip()
    assert "log-test.tk" in content
