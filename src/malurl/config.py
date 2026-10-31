# -*- coding: utf-8 -*-
"""服务配置：全部来自环境变量，带安全默认值。

真实部署时不要在代码里写死路径和密钥，这一层就是干这个的。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


def _env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(int(default))).lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    # ---- 路径 ----
    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2])
    model_path: Path = field(default_factory=lambda: Path(os.environ.get(
        "MALURL_MODEL_PATH", "artifacts/model.joblib")))
    log_path: Path = field(default_factory=lambda: Path(os.environ.get(
        "MALURL_LOG_PATH", "artifacts/judgements.jsonl")))

    # ---- 判定 ----
    threshold: float = field(default_factory=lambda: _env_float("MALURL_THRESHOLD", 0.5))
    high_risk: float = field(default_factory=lambda: _env_float("MALURL_HIGH_RISK", 0.8))

    # ---- 服务 ----
    host: str = field(default_factory=lambda: os.environ.get("MALURL_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("MALURL_PORT", 8000))
    api_key: str = field(default_factory=lambda: os.environ.get("MALURL_API_KEY", ""))
    max_batch: int = field(default_factory=lambda: _env_int("MALURL_MAX_BATCH", 500))
    max_url_len: int = field(default_factory=lambda: _env_int("MALURL_MAX_URL_LEN", 4096))

    # ---- 日志与隐私 ----
    log_urls: bool = field(default_factory=lambda: _env_bool("MALURL_LOG_URLS", True))

    def resolve(self, path: Path) -> Path:
        """相对路径按项目根目录解析，绝对路径原样返回。"""
        return path if path.is_absolute() else (self.root / path)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
