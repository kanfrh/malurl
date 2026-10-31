FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# pip 源做成构建参数：
#   默认用官方 PyPI（CI 在境外，速度快）
#   国内构建时由 docker-compose.yml 覆盖成清华镜像，否则连 files.pythonhosted.org 会超时
ARG PIP_INDEX_URL=https://pypi.org/simple

# 先装依赖，利用 Docker 层缓存
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --retries 5 --timeout 60 -i "${PIP_INDEX_URL}" . \
    && rm -rf /root/.cache

# 数据与模型：构建时直接训练一个可用的模型，镜像自带，不依赖宿主机
COPY data ./data
RUN malurl train --data data/demo_urls.csv --out /app/model

ENV MALURL_MODEL_PATH=/app/model/model.joblib \
    MALURL_LOG_PATH=/app/logs/judgements.jsonl \
    MALURL_HOST=0.0.0.0 \
    MALURL_PORT=8000

RUN mkdir -p /app/logs
VOLUME ["/app/logs"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "malurl.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
