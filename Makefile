PY ?= python
DATA ?= data/demo_urls.csv

.PHONY: help install train compare serve test predict drift replay docker

help:
	@echo "make install   安装依赖"
	@echo "make train     训练模型（DATA=data/urls.csv 可指定数据集）"
	@echo "make compare   对比随机切分与分组切分"
	@echo "make serve     启动 HTTP 服务"
	@echo "make test      运行测试"
	@echo "make drift     计算特征漂移"
	@echo "make replay    统计判定日志"
	@echo "make docker    构建并启动容器"

install:
	$(PY) -m pip install -e ".[dev]"

train:
	$(PY) -m malurl train --data $(DATA)

compare:
	$(PY) -m malurl compare --data $(DATA)

serve:
	$(PY) -m malurl serve

test:
	$(PY) -m pytest -q

predict:
	$(PY) -m malurl predict "http://secure-paypal.tk/login.php" "https://www.taobao.com/item/detail?id=1"

drift:
	$(PY) -m malurl drift --data $(DATA)

replay:
	$(PY) -m malurl replay --limit 200

docker:
	docker compose up --build
