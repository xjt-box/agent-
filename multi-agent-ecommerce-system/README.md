# Multi-Agent E-Commerce Recommendation System

一个基于 FastAPI、asyncio 和 Supervisor 模式的多 Agent 电商推荐系统。系统协调用户画像、商品推荐、库存决策和营销文案 4 个 Agent，提供 HTTP API、简单前端和离线质量门禁。

## 功能

- 用户画像：基于用户行为和 RFM 信息生成分群与偏好。
- 商品推荐：候选商品从 SQLite 目录读取后，结合用户画像进行重排。
- 库存决策：从 SQLite 查询实时库存，过滤不可售商品，并输出低库存预警和限购建议。
- 保守降级：库存为空或库存服务不可用时不推荐商品、不生成文案，并通过 `degradation_reasons` 返回原因。
- 营销文案：按用户分群生成结构化商品文案，并过滤敏感词。
- 实验与监控：内置 A/B 分组、Thompson Sampling 和指标查询接口。
- 质量门禁：使用 Fake Agent、黄金用例和确定性合同离线验证推荐主链路；GitHub Actions 会在推送和 PR 时自动运行检查。

## 架构

```text
Request
  -> Supervisor
     -> user_profile + product_recall        (parallel)
     -> rerank + inventory                   (parallel)
     -> marketing_copy
  -> RecommendationResponse
```

核心代码位于 `python/`：

```text
python/
  agents/          # 4 个业务 Agent
  orchestrator/    # Supervisor 与 LangGraph 工作流
  services/        # A/B 测试、特征与指标服务
  models/          # Pydantic 请求/响应模型
  frontend/        # 浏览器界面
  tests/           # 离线回归测试与黄金用例
  scripts/         # 质量门禁执行器
```

## 快速开始

### 1. 创建环境并安装依赖

```powershell
cd multi-agent-ecommerce-system\python
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. 配置模型

在 `python/.env` 中设置 OpenAI 兼容模型配置。以下是阿里云百炼的示例：

```env
ECOM_LLM_API_KEY=你的百炼_API_Key
ECOM_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
ECOM_LLM_MODEL=qwen-plus
```

也可以使用其他 OpenAI 兼容服务，只需替换 `ECOM_LLM_BASE_URL`、`ECOM_LLM_MODEL` 和密钥。不要提交 `.env` 文件。服务首次启动时会自动创建 `ecommerce.db` 并写入本地种子商品。

### 3. 启动服务

```powershell
.\.venv\Scripts\python.exe .\main.py
```

启动成功后打开：

- 前端界面：`http://127.0.0.1:8000/`
- API 文档：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`

终端持续显示日志是正常的；按 `Ctrl+C` 停止服务。

## 调用推荐 API

```powershell
$body = @{
  user_id = "user_001"
  scene = "homepage"
  num_items = 5
  context = @{
    recent_views = @("phone", "headphones")
    avg_order_amount = 500
  }
} | ConvertTo-Json -Depth 3

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/recommend" `
  -ContentType "application/json" `
  -Body $body
```

常用接口：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/v1/recommend` | Supervisor 推荐链路 |
| `POST` | `/api/v1/recommend/graph` | LangGraph 推荐链路 |
| `GET` | `/api/v1/experiments` | 查看 A/B 实验状态 |
| `GET` | `/api/v1/metrics` | 查看 Agent 和业务指标 |
| `GET` | `/health` | 健康检查 |

## 运行质量检查

质量检查不调用模型，也不需要 API Key：

```powershell
cd multi-agent-ecommerce-system\python
.\.venv\Scripts\python.exe .\scripts\run_quality_gate.py
```

检查内容包括：

- A/B 测试引擎行为。
- SQLite 商品目录、库存更新和非法库存校验。
- 商品推荐 Agent 使用 SQLite 候选目录和用户偏好排序。
- 数据库库存为 0 的商品会被 Supervisor 从最终推荐和营销文案中排除。
- 推荐数量、商品去重和库存范围。
- 营销文案与推荐商品的一致性。
- Agent 结果完整性、空输入和异常边界。
- 5 个固定黄金用例。

预期结果：

```text
[PASS] test_ab_test.py
[PASS] test_catalog_repository.py
[PASS] test_product_rec_catalog.py
[PASS] test_inventory_catalog.py
[PASS] test_recommendation_eval.py
Quality gate summary: passed=5 failed=0
```

## 开发说明

- 线上推荐入口保持 `SupervisorOrchestrator()` 的原有用法。
- 测试通过 `AgentBundle` 注入 Fake Agent，因此可在无模型、无 Redis、无数据库环境下验证核心编排逻辑。
- `.github/workflows/quality-gate.yml` 会在推送 `main`、提交 PR 或手动触发时执行相同的质量检查。

## 技术栈

Python, FastAPI, asyncio, LangGraph, LangChain, Pydantic, Redis, Prometheus, Docker, GitHub Actions。
## Resume a recommendation run

The Supervisor stores phase checkpoints in the configured SQLite database. A normal
`POST /api/v1/recommend` response returns `request_id`; submit the same request with
that value as `resume_run_id` to continue a run after an interruption. Checkpoints
are restricted to the original `user_id`.

```json
{
  "user_id": "user_001",
  "scene": "homepage",
  "num_items": 5,
  "resume_run_id": "the-previous-request-id"
}
```

A checkpoint after Phase 1 resumes from re-ranking and inventory validation. A
checkpoint after Phase 2 only regenerates marketing copy. Completed runs return the
persisted response without repeating agent calls.
## LangGraph checkpoint recovery

`POST /api/v1/recommend/graph` now stores a durable checkpoint after every LangGraph
node in SQLite. The response `request_id` is also the graph run ID. Submit the same
request with `resume_run_id` after an interruption to continue from the most recent
checkpoint. Recovery is restricted to the original `user_id`.
### Recordable checkpoint demo

Run the deterministic end-to-end demo below to show a FastAPI graph request and a
second request recovered with the returned run ID. It does not call an external model.

```powershell
cd multi-agent-ecommerce-system\python
.\.venv\Scripts\python.exe .\scripts\demo_graph_checkpoint_recovery.py
```