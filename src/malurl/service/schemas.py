# -*- coding: utf-8 -*-
"""接口的请求/响应模型。

Pydantic 会自动做类型校验：非法输入直接返回 422，业务代码里不用再写一堆 if。
json_schema_extra 里的 example 会显示在 /docs 页面上，点 "Try it out" 就自动填好。
"""
from __future__ import annotations

from typing import Dict, List, Optional

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


class ModelInfoResponse(BaseModel):
    """模型元信息。给前端/看板展示用，缺失的字段返回 None 而不是报错。"""

    model_name: str = Field(..., description="模型名称，如 XGBoost")
    model_version: str = Field(..., description="训练时生成的版本号")
    threshold: float = Field(..., description="服务端判定阈值")
    feature_dim: int = Field(..., description="特征维度")
    n_total: Optional[int] = Field(None, description="训练样本总数")
    n_train: Optional[int] = Field(None, description="训练集条数")
    n_test: Optional[int] = Field(None, description="测试集条数")
    data_file: Optional[str] = Field(None, description="训练数据文件名")
    trained_at: Optional[str] = Field(None, description="训练时间")
    split: Optional[str] = Field(None, description="切分方式：group / random")
    domain_overlap: Optional[float] = Field(None, description="训练测试域名重叠率")
    select_reason: Optional[str] = Field(None, description="模型选型理由")
    metrics: Dict[str, float] = Field(default_factory=dict,
                                      description="测试集指标：accuracy/precision/recall/f1/auc")
    confusion: Optional[List[List[int]]] = Field(None, description="混淆矩阵 [[TN,FP],[FN,TP]]")
