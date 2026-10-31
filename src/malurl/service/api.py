# -*- coding: utf-8 -*-
"""HTTP 推理服务。

启动：
    uvicorn malurl.service.api:app --host 0.0.0.0 --port 8000
或：
    malurl serve

接口：
    POST /predict   批量判定
    GET  /healthz   健康检查
    GET  /metrics   Prometheus 指标
    GET  /docs      交互式接口文档
"""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)

from .. import store
from ..config import get_settings
from ..features import extract_features
from ..model import load, predict_proba_safe
from .schemas import (HealthResponse, ModelInfoResponse, PredictItem, PredictRequest,
                     PredictResponse)

settings = get_settings()

REQ_TOTAL = Counter("malurl_requests_total", "接口请求数", ["endpoint", "status"])
REQ_LATENCY = Histogram("malurl_request_seconds", "接口耗时（秒）", ["endpoint"])
SCORE_HIST = Histogram("malurl_score", "恶意概率分布",
                       buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
MALICIOUS_TOTAL = Counter("malurl_malicious_total", "判定为恶意的 URL 数量")
WHITELIST_TOTAL = Counter("malurl_whitelist_total", "命中可信白名单、未走模型的 URL 数量")

_state: dict = {"bundle": None, "error": "模型尚未加载"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载模型；加载失败也让服务起来，由 /healthz 暴露问题。"""
    try:
        _state["bundle"] = load(settings.resolve(settings.model_path))
        _state["error"] = ""
    except Exception as e:  # noqa: BLE001
        _state["bundle"] = None
        _state["error"] = "模型加载失败：%s" % e
    yield
    _state["bundle"] = None


app = FastAPI(
    title="恶意 URL 检测服务",
    description="基于 41 维 URL 静态特征与集成学习模型的检测接口。不访问目标站点。",
    version="0.1.0",
    lifespan=lifespan,
)


def require_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    """配置了 MALURL_API_KEY 时才校验，方便本地调试。"""
    if settings.api_key and x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="无效的 API Key")


def _risk(score: float) -> str:
    """三级风险。高危线取 max(配置值, 阈值)，保证"高危"一定包含在"恶意"之内。"""
    if score >= max(settings.high_risk, settings.threshold):
        return "high"
    if score >= settings.threshold:
        return "medium"
    return "low"


@app.get("/", include_in_schema=False)
def index(request: Request):
    """人类友好的入口页；机器请直接调 /predict。"""
    bundle = _state["bundle"]
    version = bundle.get("version", "-") if bundle else "未加载"
    model_name = bundle.get("model_name", "-") if bundle else "-"
    status = "运行中" if bundle else "模型不可用"
    color = "#1a7f37" if bundle else "#cf222e"

    html = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>恶意 URL 检测服务</title>
<style>
  body{{margin:0;background:#f6f8fa;font-family:"Microsoft YaHei",system-ui,sans-serif;color:#1f2328}}
  .wrap{{max-width:760px;margin:0 auto;padding:48px 24px}}
  h1{{font-size:26px;margin:0 0 8px}}
  .sub{{color:#656d76;font-size:14px;margin-bottom:28px}}
  .card{{background:#fff;border:1px solid #d0d7de;border-radius:10px;padding:20px 22px;margin-bottom:18px}}
  .row{{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px dashed #eaeef2;font-size:14px}}
  .row:last-child{{border-bottom:none}}
  .k{{color:#656d76}}
  .badge{{color:{color};font-weight:600}}
  a{{color:#0969da;text-decoration:none}}
  a:hover{{text-decoration:underline}}
  code,pre{{font-family:Consolas,monospace;font-size:13px}}
  pre{{background:#f6f8fa;border:1px solid #eaeef2;border-radius:8px;padding:14px;overflow-x:auto}}
  .tip{{font-size:12px;color:#8c959f;margin-top:26px;text-align:center}}
</style></head><body><div class="wrap">
  <h1>恶意 URL 检测服务</h1>
  <div class="sub">基于 41 维 URL 静态特征与机器学习模型，仅分析 URL 字符串本身，不访问目标站点。</div>

  <div class="card">
    <div class="row"><span class="k">服务状态</span><span class="badge">{status}</span></div>
    <div class="row"><span class="k">当前模型</span><span>{model_name}</span></div>
    <div class="row"><span class="k">模型版本</span><span>{version}</span></div>
    <div class="row"><span class="k">判定阈值</span><span>{threshold}</span></div>
  </div>

  <div class="card">
    <div class="row"><a href="/docs">接口文档（可在线调试）</a><span class="k">/docs</span></div>
    <div class="row"><a href="/healthz">健康检查</a><span class="k">/healthz</span></div>
    <div class="row"><a href="/metrics">运行指标（Prometheus）</a><span class="k">/metrics</span></div>
    <div class="row"><span>批量判定接口</span><span class="k">POST /predict</span></div>
  </div>

  <div class="card">
    <div class="k" style="margin-bottom:10px">调用示例</div>
    <pre>curl -X POST http://127.0.0.1:{port}/predict \\
  -H "Content-Type: application/json" \\
  -d '{{"urls":["http://secure-paypal.tk/login.php"]}}'</pre>
  </div>

  <div class="tip">本页由服务自动生成；给机器调用请直接用 /predict，无需经过本页。</div>
</div></body></html>"""

    return HTMLResponse(html.format(
        status=status, color=color, model_name=model_name, version=version,
        threshold=settings.threshold, port=request.url.port or settings.port))


@app.get("/healthz", response_model=HealthResponse, tags=["运维接口"],
         summary="健康检查：服务是否可用、模型是否加载")
def healthz():
    bundle = _state["bundle"]
    return {
        "status": "ok" if bundle else "degraded",
        "model_loaded": bundle is not None,
        "model_version": bundle.get("version", "") if bundle else "",
    }


@app.get("/model", response_model=ModelInfoResponse, tags=["运维接口"],
         summary="模型信息：名称、版本、阈值与测试集指标")
def model_info():
    """给前端展示用。指标从模型同目录的 model_info.json 读取，读不到就返回空指标。"""
    bundle = _state["bundle"]
    if not bundle:
        raise HTTPException(status_code=503, detail=_state["error"] or "模型未加载")

    meta: dict = {}
    try:
        info_path = Path(str(settings.resolve(settings.model_path))).with_name("model_info.json")
        if info_path.exists():
            meta = json.loads(info_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001  ——元信息读失败不影响服务
        meta = {}

    return {
        "model_name": bundle.get("model_name", "-"),
        "model_version": bundle.get("version", "-"),
        "threshold": float(bundle.get("threshold", settings.threshold)),
        "feature_dim": len(bundle.get("feature_names", []) or []),
        "n_total": meta.get("n_total"),
        "n_train": meta.get("n_train"),
        "n_test": meta.get("n_test"),
        "data_file": meta.get("data_file"),
        "trained_at": meta.get("trained_at"),
        "split": meta.get("split"),
        "domain_overlap": meta.get("domain_overlap"),
        "select_reason": meta.get("select_reason"),
        "metrics": {k: float(v) for k, v in (meta.get("metrics") or {}).items()},
        "confusion": meta.get("confusion"),
    }


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictResponse, dependencies=[Depends(require_key)],
          tags=["检测接口"], summary="批量检测 URL，返回恶意概率与风险等级")
def predict(req: PredictRequest):
    bundle = _state["bundle"]
    if bundle is None:
        REQ_TOTAL.labels("predict", "503").inc()
        raise HTTPException(status_code=503, detail=_state["error"] or "模型不可用")

    if len(req.urls) > settings.max_batch:
        REQ_TOTAL.labels("predict", "422").inc()
        raise HTTPException(status_code=422,
                            detail="单次最多 %d 条 URL" % settings.max_batch)

    for i, u in enumerate(req.urls):
        if len(u) > settings.max_url_len:
            REQ_TOTAL.labels("predict", "422").inc()
            raise HTTPException(status_code=422,
                                detail="第 %d 条 URL 超过 %d 字符上限" % (i + 1, settings.max_url_len))

    t0 = time.perf_counter()

    # ---- 信誉层：白名单命中的域名不走模型 ----
    from ..reputation import check as reputation_check

    rep = [reputation_check(u) for u in req.urls]
    scores = [0.0] * len(req.urls)
    model_idx = [i for i, r in enumerate(rep) if not r["trusted"]]
    if model_idx:
        X = np.asarray([extract_features(req.urls[i]) for i in model_idx], dtype=float)
        preds = predict_proba_safe(bundle["model"], X)[:, 1]
        for i, s in zip(model_idx, preds):
            scores[i] = float(s)

    n_trusted = len(req.urls) - len(model_idx)
    if n_trusted:
        WHITELIST_TOTAL.inc(n_trusted)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    version = bundle.get("version", "")
    results = []
    for i, u in enumerate(req.urls):
        s = scores[i]
        if rep[i]["trusted"]:
            results.append(PredictItem(url=u, score=0.0, label="benign",
                                       risk="low", source="whitelist"))
            continue
        SCORE_HIST.observe(s)
        results.append(PredictItem(
            url=u, score=round(s, 4),
            label="malicious" if s >= settings.threshold else "benign",
            risk=_risk(s), source="model"))

    n_mal = sum(1 for r in results if r.label == "malicious")
    MALICIOUS_TOTAL.inc(n_mal)

    # 落盘判定日志，供样本回流与漂移分析
    store.append(settings.resolve(settings.log_path),
                 [{"url": r.url, "score": r.score, "label": r.label,
                   "risk": r.risk, "model_version": version, "source": req.source}
                  for r in results],
                 keep_url=settings.log_urls)

    REQ_TOTAL.labels("predict", "200").inc()
    REQ_LATENCY.labels("predict").observe(elapsed_ms / 1000.0)
    return PredictResponse(
        model_version=version,
        model_name=bundle.get("model_name", ""),
        threshold=settings.threshold,
        count=len(results),
        elapsed_ms=round(elapsed_ms, 3),
        results=results,
    )
