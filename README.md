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
- `GITHUB_REPOSITORIES`：可选的首次初始化白名单，仅在仓库表为空时导入；
- `DATABASE_URL`：MySQL SQLAlchemy URL；
- `DEEPSEEK_API_KEY`：执行标注时需要。

只验证采集和语料构建时可运行：

```bash
python pipeline.py run --skip-label
```

## 命令

```text
python pipeline.py init-db        创建缺失的表（不会清空或重建已有表）
python pipeline.py collect        只采集
python pipeline.py build-corpus   只构建/更新语料
python pipeline.py label          只标注未成功标注的语料
python pipeline.py run            串联第一阶段流水线
python pipeline.py status         查询最近 10 次运行
python pipeline.py repo list      查询仓库白名单
python pipeline.py repo add owner/repo       添加并启用仓库
python pipeline.py repo enable owner/repo    重新启用仓库
python pipeline.py repo disable owner/repo   停用仓库并保留历史数据
```

## 仓库白名单

`repositories` 表是运行期仓库白名单的唯一数据源。`init-db` 仅在该表完全为空时，把
`.env` 中的 `GITHUB_REPOSITORIES` 导入一次；以后修改 `.env` 不会覆盖数据库状态。

日常通过命令管理：

```bash
python pipeline.py repo list
python pipeline.py repo add rust-lang/rust
python pipeline.py repo disable rust-lang/rust
python pipeline.py repo enable rust-lang/rust
```

停用不会删除仓库、游标或历史语料，只会让后续采集跳过该仓库。每次采集启动时读取一次
所有 `enabled=1` 的仓库，作为本次运行的白名单快照。

## 增量与恢复语义

- 每个仓库分别维护 Issue/PR、普通评论、Review comment 三条游标；
- 请求 `since` 默认回退 5 分钟，同一对象依靠唯一键 upsert 去重；
- 每页数据、隔离记录和 `repository_cursors` 游标在同一个 MySQL 事务中提交；
- 每段最多读取 250 页，然后从当前游标重新发起 `since` 请求，避开仓库级
  comments 接口第 301 页的分页限制；
- 游标表示最近安全提交高水位；中途失败时下次从该游标向前重叠 5 分钟继续；
- PR 正常采集直接使用仓库 Issue 列表中的 PR 表示，不再逐 PR 请求详情；
- 评论缺少父记录时先按父编号补采；无法补采的原始响应进入
  `unresolved_collection_items`，后续评论流启动时自动对账；
- 单条解析或字段异常进入隔离表后继续当前页；HTTP 和数据库瞬时错误有限重试；
  认证、权限和数据库 Schema 错误立即终止任务；
- Issue/PR 父流失败时，该仓库两条评论流标记为 `skipped_dependency`；
- `run` 发现任一采集流失败时停止在采集阶段，不构建 corpus，也不启动 LLM 标注；
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
