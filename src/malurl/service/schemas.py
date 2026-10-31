# -*- coding: utf-8 -*-
"""接口的请求/响应模型。

Pydantic 会自动做类型校验：非法输入直接返回 422，业务代码里不用再写一堆 if。
json_schema_extra 里的 example 会显示在 /docs 页面上，点 "Try it out" 就自动填好。
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "urls": [
                "http://secure-paypal.tk/login.php",
                "https://www.icbc.com.cn/login",
            ],
            "source": "docs",
        }
    })

    urls: List[str] = Field(
        ..., description="待检测的 URL 列表，单次最多 500 条", min_length=1)
    source: str = Field("api", description="调用来源，会写进判定日志，便于区分渠道")


class PredictItem(BaseModel):
    url: str = Field(..., description="原始 URL")
    score: float = Field(..., description="恶意概率，0~1，越大越可疑；白名单命中时恒为 0")
    label: str = Field(..., description="malicious 表示拦截，benign 表示放行")
    risk: str = Field(..., description="high / medium / low，对应高危/可疑/正常")
    source: str = Field("model", description="判定来源：model 走模型，whitelist 命中可信白名单")


class PredictResponse(BaseModel):
    model_version: str = Field(..., description="模型版本号，便于回溯")
    model_name: str = Field(..., description="当前使用的模型")
    threshold: float = Field(..., description="判定阈值")
    count: int = Field(..., description="本次判定的 URL 条数")
    elapsed_ms: float = Field(..., description="服务端处理耗时（毫秒）")
    results: List[PredictItem]


class HealthResponse(BaseModel):
    status: str = Field(..., description="ok 表示服务可用，degraded 表示模型未加载")
    model_loaded: bool
    model_version: str
