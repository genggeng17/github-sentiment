# GitHub Rust 社区情感分析流水线

第一阶段后端实现，覆盖 GitHub 历史回填/增量采集、MySQL 幂等存储、统一语料、
DeepSeek 方面级标注、运行记录与中断恢复。详细业务范围见 `需求说明.md`。

## 快速开始

要求 Python 3.11+、MySQL 8.0+。

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                   # Windows 可手动复制
docker compose up -d mysql
python pipeline.py init-db
python pipeline.py run
```

`.env` 中至少填写：

- `GITHUB_TOKEN`：只读 GitHub Token；
- `GITHUB_REPOSITORIES`：逗号分隔的 `owner/repo` 白名单；
- `DATABASE_URL`：MySQL SQLAlchemy URL；
- `DEEPSEEK_API_KEY`：执行标注时需要。

只验证采集和语料构建时可运行：

```bash
python pipeline.py run --skip-label
```

## 命令

```text
python pipeline.py init-db        创建表
python pipeline.py collect        只采集
python pipeline.py build-corpus   只构建/更新语料
python pipeline.py label          只标注未成功标注的语料
python pipeline.py run            串联第一阶段流水线
python pipeline.py status         查询最近 10 次运行
```

## 增量与恢复语义

- 每个仓库分别维护 Issue/PR、普通评论、Review comment 三条游标；
- 请求 `since` 默认回退 5 分钟，同一对象依靠唯一键 upsert 去重；
- 每页解析完成后立即用一个小事务写入 MySQL；
- 只有一条数据流的全部 Link 分页成功后才推进该游标；
- 中途失败时已提交页保留、游标不动，下次会重取并幂等覆盖；
- MySQL named lock 阻止两个周任务并发执行；
- `pipeline_runs` 和 `collection_stream_runs` 可查询整体及每仓库/流的页数、读取数、
  写入数、重试数和错误。

## 语料与标注追溯

评论的 `model_input` 固定分为 `[CONTEXT]` 和 `[TARGET]`。普通评论和 Review comment
都只携带父级 Issue/PR 标题，不将代码文件路径传给模型。Prompt 明确只标注 TARGET
表达的情感。

`llm_annotations` 同时保存原始响应、结构校验后的 JSON、错误、标签体系版本、Prompt
版本与模型名。校验仅允许需求定义的 16 个 aspect 和 3 个 class；失败结果会记录但不会
伪装成成功，下次运行会重试。

`corpus` 按来源和内容版本哈希保存不可变版本。GitHub 原文或父标题发生变化时会创建
新版本并重新进入标注队列，旧标注仍指向当时的原始文本和模型输入。

## 每周调度（Ubuntu cron）

先手动验证命令和日志目录，再用 `crontab -e` 加入：

```cron
15 3 * * 1 cd /opt/github-sentiment && /opt/github-sentiment/.venv/bin/python pipeline.py run >> /var/log/github-sentiment.log 2>&1
```

生产密钥应由受限权限的 EnvironmentFile 或 cron 环境提供，不提交 `.env`。

## 开发验证

```bash
ruff check .
pytest --cov
```
