# malurl —— 恶意 URL 检测服务（可部署版）

> 这是毕设原型 `恶意URL检测毕设材料/code/` 的工程化版本：
> 从一个"能演示的网页"，变成一个**别人能安装、能调用、能部署的服务**。

| | 毕设原型 | 本项目 |
|---|---|---|
| 入口 | Streamlit 网页 | HTTP API + CLI + 网页面板 |
| 评估 | 随机切分 | **按域名分组切分**（避免域名泄漏） |
| 配置 | 写在代码里 | 环境变量 |
| 日志 | 无 | 判定日志 JSONL，可回流 |
| 测试 | 无 | 27 个用例（特征边界、切分、接口、信誉层） |
| 部署 | 手动 | Docker + 健康检查 + 指标 |

---

## 1. 五分钟跑起来（本地）

需要 Python 3.9+。

```bash
cd malurl
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/Mac: source .venv/bin/activate

pip install -e ".[dev]"
malurl train --data data/demo_urls.csv     # 训练，产物写到 artifacts/
malurl serve                               # 启动服务，默认 8000 端口
```

另开一个终端：

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"urls\":[\"http://secure-paypal.tk/login.php\"]}"
```

返回：

```json
{
  "model_version": "20261030-155838",
  "model_name": "随机森林",
  "threshold": 0.5,
  "count": 1,
  "elapsed_ms": 0.82,
  "results": [
    {"url": "http://secure-paypal.tk/login.php", "score": 0.9513,
     "label": "malicious", "risk": "high"}
  ]
}
```

打开 <http://127.0.0.1:8000/docs> 可以在浏览器里直接调试接口。

---

## 2. 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/predict` | 批量判定，单次上限由 `MALURL_MAX_BATCH` 控制（默认 500） |
| GET | `/healthz` | 健康检查，返回模型是否加载成功与版本号 |
| GET | `/model` | 模型元信息：名称、版本、阈值、特征维度、测试集指标（远程界面用） |
| GET | `/metrics` | Prometheus 指标：QPS、延迟分位、恶意判定数、概率分布 |
| GET | `/docs` | 交互式接口文档（Swagger） |

**鉴权**：设置 `MALURL_API_KEY` 后，所有 `/predict` 请求必须带 `X-API-Key` 头，否则 401。
留空则不校验，方便本地调试。

**错误约定**：

| 状态码 | 场景 |
|---|---|
| 422 | 请求体不合法、URL 数超限、单条 URL 超过 4096 字符 |
| 401 | API Key 不对 |
| 503 | 模型未加载（服务起来了但模型文件有问题，错误信息在 detail 里） |

---

## 3. 配置

全部走环境变量，复制 `.env.example` 改即可。

| 变量 | 默认 | 说明 |
|---|---|---|
| `MALURL_MODEL_PATH` | `artifacts/model.joblib` | 模型路径 |
| `MALURL_LOG_PATH` | `artifacts/judgements.jsonl` | 判定日志 |
| `MALURL_THRESHOLD` | `0.5` | 判为恶意的概率阈值 |
| `MALURL_HIGH_RISK` | `0.8` | 判为高危的阈值 |
| `MALURL_API_KEY` | 空 | 接口鉴权密钥，生产必须设置 |
| `MALURL_MAX_BATCH` | `500` | 单次请求 URL 上限 |
| `MALURL_MAX_URL_LEN` | `4096` | 单条 URL 长度上限（防 DoS） |
| `MALURL_LOG_URLS` | `true` | 是否在日志里存原始 URL；`false` 只存哈希 |

---

## 4. 训练与评估

```bash
malurl train   --data data/urls.csv      # 训练并保存模型
malurl compare --data data/urls.csv      # 随机切分 vs 分组切分对照
malurl predict "http://a.tk/login"       # 命令行判定
malurl drift   --data data/urls.csv      # 特征漂移（PSI）
malurl replay  --limit 200               # 统计判定日志
```

### 关于「分组切分」，结论很明确

同一个域名下的 URL 高度相似（`evil.tk/login`、`evil.tk/verify`），随机切分会让训练集和测试集共享域名，模型等于背答案。所以本项目默认**按 host 分组切分**。

在真实数据上（53 048 条、41 维特征、随机森林）：

| 切分方式 | 准确率 | F1 | 域名重叠率 |
|---|---|---|---|
| 随机切分 | 93.32% | 0.8464 | **60.5%** |
| 分组切分 | 90.62% | 0.8007 | 0% |

**随机切分把 F1 高估了 0.046、准确率高估 2.70 个百分点**——60% 的测试样本其域名在训练集出现过，这就是"背答案"。

> 早期用合成数据做同样对照时，分组切分反而略高（+0.46）。因为合成数据里恶意域名之间没有共享结构，泄漏形不成优势。换成真实数据后规律立刻显现——**评估方法本身也要在真实数据上验证**。

**不要在论文里预设"分组切分会让指标下降"**，跑出来是多少就写多少。

---

## 5. 测试

```bash
pytest -q
```

27 个用例，覆盖四类最容易出事的地方：

- **特征边界**：空串、超长 URL、中文域名、`@` 混淆、IP、乱码、`http://[::1]` 等
- **切分逻辑**：分组切分后域名重叠率必须为 0，且两类样本都要在
- **接口**：鉴权 401、批量超限 422、超长 URL 422、空列表 422、指标可读、日志落盘

---

## 6. Docker

```bash
docker compose up --build
```

镜像构建时会自动用 `data/demo_urls.csv` 训练一个模型并烘焙进去，所以**不需要先在宿主机训练**。日志写到容器内 `/app/logs`，compose 已经挂到 `./logs`。

> 注意：镜像默认**不安装 xgboost**（wheel 约 200 MB，占镜像体积大头），因此训练时会在剩余四个模型里选优。若要与论文里的结果完全一致，在 Dockerfile 的 `RUN pip install ...` 后面追加一句 `RUN pip install --no-cache-dir xgboost` 再构建。

换成自己的模型：

```bash
# 宿主机先训练
malurl train --data data/urls.csv
# 再改 compose，把 ./artifacts 挂到 /app/model 并设置 MALURL_MODEL_PATH
```

---

## 7. 换成真实数据

现在 `data/demo_urls.csv` 是**合成数据**，只用来跑通流程，指标没有学术意义。

真实数据来源：

| 数据集 | 规模 | 地址 |
|---|---|---|
| Malicious URL Dataset | 约 65 万 | Kaggle 搜同名 |
| PhiUSIIL Phishing URL | 约 23 万 | Kaggle 搜 `PhiUSIIL` |
| URL-Phish | 约 23 万 | Kaggle 搜 `URL-Phish` |
| Malicious URLs (UNB) | 大 | unb.ca/cic/datasets/url-2016.html |

放到 `data/urls.csv` 即可，`load_dataset` 会自动识别列名（`url/URL/网址`、`label/type/标签`），
`malicious/benign` 这类文字标签也会自动转成 1/0。

建议再加一步：真实负样本从 Tranco Top 1M 里抽，别用自己拼的 URL——真实的正常 URL 里有一堆 `/login`、`/verify`，比合成的难得多。

---

## 8. 目录结构

```
malurl/
├── src/malurl/
│   ├── config.py        配置（环境变量）
│   ├── features.py      41 维特征
│   ├── data.py          加载 / 清洗 / 随机切分 / 分组切分
│   ├── model.py         训练、评估、持久化、PSI 参考分布
│   ├── drift.py         PSI 漂移计算
│   ├── store.py         判定日志（JSONL）
│   ├── cli.py           命令行入口
│   └── service/
│       ├── api.py       FastAPI 应用
│       └── schemas.py   请求/响应模型
├── tests/               27 个用例
├── data/                数据集（真实数据不入库）
├── artifacts/           模型与指标产物（不入库）
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

---

## 9. 论文里可以写什么

这套东西能撑起"系统实现与部署"一章：

1. **工程架构图**：CLI / HTTP / 容器三层，模型以文件形式交付
2. **接口设计**：请求校验、错误码、鉴权、限流上限
3. **可观测性**：健康检查 + Prometheus 指标 + 判定日志
4. **评估方法的改进**：分组切分与对照实验（含上表，哪怕结论是"没有显著差异"）
5. **漂移监控**：PSI 指标与阈值策略

---

## 10. 已知限制

- 只用 URL 字符串，不抓页面内容，因此对"域名干净、内容恶意"的站点无效
- 没有实现在线学习，概念漂移需要人工触发重训练
- 判定日志默认落本地文件，多实例部署时要换成中心化存储（Kafka / ClickHouse）
- 没有做分布式限流，`MALURL_MAX_BATCH` 只是单请求上限
- 合成数据的指标不能用于论文
