# GitHub Rust 社区情感分析流水线

本项目面向 GitHub Rust 社区，采集指定仓库中的 Issue、Pull Request 及评论，
构建统一语料，并生成方面级情感标签，为后续训练 Rust 社区情感分析模型提供数据。

## 数据流程

```text
GitHub 数据采集 → 文本清洗与去重 → LLM 标注（默认 GLM）→ 人工抽检 → 导出训练数据
```

## 版本变量与历史

流水线不会用单一的“项目版本”覆盖所有产物，而是分别记录数据处理和模型相关版本。
判断两条结果能否直接比较时，应同时核对这些变量：

| 阶段 | 变量或字段 | 当前值 | 作用 |
| --- | --- | --- | --- |
| GitHub 采集 | `GITHUB_API_VERSION` | `2022-11-28` | 固定 GitHub REST API 契约；不参与语料或标注唯一键 |
| 语料构建 | `CLEANING_VERSION` / `corpus.cleaning_version` | `clean-v2` | 标识清洗和模型输入构造规则，并参与 `content_hash` 计算；`model_input_chars` 保存输入字符数供采样过滤 |
| 采样 | `corpus_sample_sets.name` | 运行时指定，如 `rust-v1` | 标识不可变采样集；还需结合清洗版本、每仓库上限、`seed` 和候选语料字符上限追溯 |
| 标签体系 | `TAXONOMY_VERSION` / `llm_annotations.taxonomy_version` | `rust-aspects-v2` | 标识允许输出的方面及其定义 |
| Prompt | `PROMPT_VERSION` / `llm_annotations.prompt_version` | `aspect-sentiment-zh-v8` | 标识提示词、输入输出协议和标注规则 |
| LLM | `LLM_PROVIDER`、`GLM_MODEL` / `llm_annotations.model_name` | `glm`、`glm-5.3-flash` | 默认智谱 BigModel；显式选择 deepseek 时读取 `DEEPSEEK_MODEL` |
| BERT | `bert_predictions.model_version` | 暂无固定值 | 为后续 BERT/ONNX 推理预留，当前阶段尚未提供具体模型 |

现有代码和 Git 历史中的变更记录如下：

| 时间 | 变量变化 | 说明 |
| --- | --- | --- |
| 2026-07-17 | `clean-v1`；`rust-aspects-v1`；`aspect-sentiment-v1`；`deepseek-chat` | 第一阶段初始版本；英文 Prompt，16 个方面，单条语料请求 |
| 2026-07-27 | `rust-aspects-v2`；`aspect-sentiment-zh-v3` | 标签合并、重命名为当前 13 个方面；改用中文规则并明确稀疏输出，默认模型仍为 `deepseek-chat` |
| 2026-07-30 | 引入版本化采样集 | 采样集名称、每仓库上限和随机种子共同确定训练候选快照；同名采样集不允许用不同参数覆盖 |
| 2026-08-04 | `aspect-sentiment-zh-v4`；`deepseek-v4-flash` | Prompt 改为批量提交并按 `corpus_id` 对齐结果；默认模型同步更新 |
| 2026-08-15 | `clean-v2` | 完整保留 `raw_text` 和 `target_text`；仅在 `clean_text`/`model_input` 中用可审计占位符折叠 fenced 代码块、长日志、堆栈和空模板段落 |
| 2026-08-15 | `aspect-sentiment-zh-v5` | 收紧 13 个方面的语义边界，并明确技术材料、上下文回应和多标签的判定规则 |
| 2026-08-16 | `aspect-sentiment-zh-v6` | 增加不可信输入与批次隔离，统一确认和否定表达，规定同方面褒贬并存及 API/库生态边界 |
| 2026-08-16 | `aspect-sentiment-zh-v7` | 改为一条语料一个独立请求和单条 JSON 响应；增加可控异步并发、固定 `user_id`、缓存命中统计及批量写库 |
| 2026-08-16 | 标注安全熔断 | 并发前强制串行预检；401/403 等全局鉴权或配置错误立即停止，连续非致命失败达到阈值后停止并取消剩余请求 |

`GITHUB_API_VERSION=2022-11-28` 自项目建立后尚未变更。现有 Git 记录中没有
`aspect-sentiment-v2`；版本号以数据库实际保存值为准，不应推测或补写缺失版本。升级
清洗规则、标签体系、Prompt 或模型时必须使用新版本值，保留旧记录用于复现，不要原地
覆盖已有版本的含义。新采样默认从当前 `CLEANING_VERSION` 选择语料，也可通过
`--cleaning-version` 选择数据库中已经生成的其他版本；不同清洗版本必须使用不同的
采样集名称。

## 仓库白名单

运行时以服务器 `repositories` 表中 `enabled=1` 的记录为准。README 中的候选池是
仓库扩展计划，不会自动修改数据库。

截至 2026-07-17，服务器已启用的仓库为：

- `rust-lang/cargo`
- `rust-lang/rust-clippy`
- `rust-lang/rust-analyzer`
- `tokio-rs/tokio`
- `serde-rs/serde`

### 50 个目标仓库与优先度

候选仓库按以下原则筛选：

- 仓库公开、非 Fork、未归档、启用 GitHub Issues，且主语言为 Rust；
- 近期仍有维护活动，Issue、PR、普通评论和 Review comment 具有可采集性；
- 讨论能够覆盖 `rust-aspects-v2`，而不只是具体产品 Bug；
- 仓库组合覆盖语言、工具链、并发网络、Web、数据库、Wasm、GUI、嵌入式和终端工具；
- 优先选择具有解释性讨论、API 权衡、性能分析、安全性讨论和用户体验反馈的项目；
- 避免仅按 Star 或数据量选择，降低机器人、模板、日志和单一业务问题造成的语料污染。

优先度表示建议的接入顺序，而不是项目质量排名：

- **P0（核心）**：标签命中预期最高，优先完成历史回填和持续增量采集；
- **P1（重点扩展）**：能显著补充领域和标签覆盖，在 P0 稳定后分批启用；
- **P2（补充观察）**：领域代表性强，但产品问题或专用场景可能稀释 Rust 情感信号；
  应先小规模采集并人工评估非空标签率。

| # | 优先度 | 仓库 | 主要领域 | 预期主要标签 | 选择理由 |
| ---: | :---: | --- | --- | --- | --- |
| 1 | P0 | `rust-lang/rust` | 编译器与标准库 | `ownership`、`type_system`、`safety`、`compile_time`、`diagnostics_debugging` | 语言核心机制和编译器体验的最直接语料来源 |
| 2 | P0 | `rust-lang/cargo` | 包管理与构建 | `package_manager`、`compile_time`、`tooling_documentation` | Cargo、依赖解析、feature、workspace 和构建体验高度集中 |
| 3 | P0 | `rust-lang/rust-clippy` | Lint 与代码质量 | `diagnostics_debugging`、`tooling_documentation`、`readability_maintainability` | 诊断质量、修复建议和代码可维护性讨论密集 |
| 4 | P0 | `rust-lang/rust-analyzer` | IDE 与语言服务器 | `tooling_documentation`、`diagnostics_debugging`、`runtime_performance` | IDE 响应、补全、诊断和开发体验的核心来源 |
| 5 | P0 | `tokio-rs/tokio` | 异步运行时 | `runtime_performance`、`safety`、`api_extensibility` | 并发、安全、延迟和异步 API 权衡讨论丰富 |
| 6 | P0 | `serde-rs/serde` | 序列化基础库 | `api_extensibility`、`libraries_frameworks`、`type_system` | 泛型、trait、派生宏和生态兼容性语料质量高 |
| 7 | P0 | `rust-lang/rustup` | 工具链安装与版本管理 | `tooling_documentation`、`package_manager`、`community` | 覆盖安装、升级、目标平台和新用户工具链体验 |
| 8 | P0 | `rust-lang/rustfmt` | 格式化工具 | `tooling_documentation`、`readability_maintainability` | 格式规则、可读性、配置和编辑器集成讨论集中 |
| 9 | P0 | `rust-lang/miri` | 未定义行为检测 | `safety`、`diagnostics_debugging`、`runtime_performance` | unsafe、内存模型、诊断和解释执行性能的高纯度语料 |
| 10 | P0 | `rust-lang/rustlings` | Rust 学习 | `learning_curve`、`diagnostics_debugging`、`community` | 补足初学者困难、练习反馈和学习支持语料 |
| 11 | P1 | `rust-lang/mdBook` | 文档工具 | `tooling_documentation`、`api_extensibility` | 文档编写、插件、渲染和使用体验讨论丰富 |
| 12 | P1 | `rust-lang/rust-bindgen` | C/C++ FFI 绑定 | `safety`、`tooling_documentation`、`diagnostics_debugging` | FFI、安全、生成代码和跨平台诊断具有代表性 |
| 13 | P1 | `rust-lang/libc` | 系统接口基础库 | `safety`、`api_extensibility`、`libraries_frameworks` | unsafe 边界、平台兼容性和底层 API 设计讨论集中 |
| 14 | P1 | `rayon-rs/rayon` | 数据并行 | `runtime_performance`、`safety`、`api_extensibility` | 并行性能、线程安全和易用 API 的典型项目 |
| 15 | P1 | `crossbeam-rs/crossbeam` | 并发原语 | `safety`、`runtime_performance`、`api_extensibility` | 内存模型、无锁结构和并发 API 的高价值语料 |
| 16 | P1 | `rust-lang/regex` | 正则表达式引擎 | `runtime_performance`、`api_extensibility`、`libraries_frameworks` | 性能、内存、语法与 API 权衡讨论较多 |
| 17 | P1 | `clap-rs/clap` | CLI 参数解析 | `api_extensibility`、`tooling_documentation`、`readability_maintainability` | 宏、派生 API、错误提示和文档体验覆盖良好 |
| 18 | P1 | `tokio-rs/axum` | Web 框架 | `api_extensibility`、`libraries_frameworks`、`type_system` | extractor、trait 约束和框架易用性讨论密集 |
| 19 | P1 | `hyperium/hyper` | HTTP 基础设施 | `runtime_performance`、`safety`、`api_extensibility` | 网络性能、协议实现和底层 API 权衡具有代表性 |
| 20 | P1 | `seanmonstar/reqwest` | HTTP 客户端 | `api_extensibility`、`libraries_frameworks`、`tooling_documentation` | 面向普通用户的 API、TLS、异步和文档反馈较丰富 |
| 21 | P1 | `tower-rs/tower` | 异步服务抽象 | `api_extensibility`、`type_system`、`readability_maintainability` | trait 组合、抽象复杂度和中间件扩展讨论集中 |
| 22 | P1 | `actix/actix-web` | Web 框架 | `runtime_performance`、`api_extensibility`、`libraries_frameworks` | 成熟用户群带来性能、API 和迁移体验语料 |
| 23 | P1 | `diesel-rs/diesel` | ORM 与数据库 | `type_system`、`compile_time`、`api_extensibility` | 类型级查询、编译错误、编译耗时和 API 易用性讨论突出 |
| 24 | P1 | `transact-rs/sqlx` | 异步 SQL 工具包 | `compile_time`、`type_system`、`api_extensibility` | 编译期查询检查、宏诊断和异步数据库体验覆盖良好 |
| 25 | P1 | `rustls/rustls` | TLS 与安全 | `safety`、`runtime_performance`、`api_extensibility` | 安全边界、性能、密码 API 和可用性讨论价值高 |
| 26 | P1 | `bytecodealliance/wasmtime` | WebAssembly 运行时 | `runtime_performance`、`safety`、`compile_time` | JIT/AOT 性能、资源隔离、安全和编译成本讨论丰富 |
| 27 | P1 | `wasm-bindgen/wasm-bindgen` | Rust/Wasm 互操作 | `tooling_documentation`、`diagnostics_debugging`、`api_extensibility` | 跨语言类型、工具链错误和 Web API 易用性语料集中 |
| 28 | P1 | `gfx-rs/wgpu` | GPU 图形 API | `safety`、`runtime_performance`、`api_extensibility` | 生命周期、安全性、性能与跨平台 API 权衡讨论丰富 |
| 29 | P1 | `rust-embedded/embedded-hal` | 嵌入式抽象 | `api_extensibility`、`libraries_frameworks`、`community` | trait 设计、硬件抽象和生态协作具有代表性 |
| 30 | P1 | `embassy-rs/embassy` | 异步嵌入式 | `safety`、`runtime_performance`、`api_extensibility` | 无堆/实时约束、异步设计和嵌入式易用性讨论较多 |
| 31 | P2 | `rust-lang/cc-rs` | 原生构建辅助 | `compile_time`、`package_manager`、`tooling_documentation` | 可补充 build script、编译器探测和跨平台构建问题 |
| 32 | P2 | `dtolnay/anyhow` | 错误处理库 | `diagnostics_debugging`、`api_extensibility` | 错误上下文和人体工程学信号强，但总体讨论量较小 |
| 33 | P2 | `dtolnay/thiserror` | 错误派生宏 | `diagnostics_debugging`、`type_system`、`api_extensibility` | 派生宏、类型约束和错误表达语料纯度高但规模较小 |
| 34 | P2 | `rwf2/Rocket` | Web 框架 | `api_extensibility`、`learning_curve`、`libraries_frameworks` | API 易用性和学习体验突出，但近期活跃度需持续观察 |
| 35 | P2 | `SeaQL/sea-orm` | 异步 ORM | `api_extensibility`、`libraries_frameworks`、`compile_time` | 补充不同 ORM 设计和框架使用体验 |
| 36 | P2 | `quinn-rs/quinn` | QUIC 网络协议 | `runtime_performance`、`safety`、`api_extensibility` | 高性能异步网络和协议 API 讨论具有补充价值 |
| 37 | P2 | `libp2p/rust-libp2p` | P2P 网络栈 | `api_extensibility`、`runtime_performance`、`libraries_frameworks` | 模块组合、异步网络和复杂生态集成语料丰富 |
| 38 | P2 | `apache/datafusion` | 查询引擎 | `runtime_performance`、`api_extensibility`、`readability_maintainability` | 大型工程中的性能、扩展和维护权衡具有代表性 |
| 39 | P2 | `pola-rs/polars` | DataFrame 引擎 | `runtime_performance`、`libraries_frameworks`、`api_extensibility` | 性能和数据生态信号强，但 Python 用户问题占比较高 |
| 40 | P2 | `quickwit-oss/tantivy` | 全文搜索引擎 | `runtime_performance`、`api_extensibility`、`libraries_frameworks` | 索引性能、内存和搜索 API 设计讨论较集中 |
| 41 | P2 | `qdrant/qdrant` | 向量数据库 | `runtime_performance`、`safety`、`readability_maintainability` | 大型生产系统性能与工程维护语料丰富，产品问题也较多 |
| 42 | P2 | `bevyengine/bevy` | 游戏引擎 | `api_extensibility`、`compile_time`、`libraries_frameworks` | ECS、编译时间和插件 API 讨论丰富，但仓库规模很大 |
| 43 | P2 | `rust-windowing/winit` | 跨平台窗口 | `api_extensibility`、`readability_maintainability`、`libraries_frameworks` | 跨平台 API 和长期兼容性权衡具有代表性 |
| 44 | P2 | `emilk/egui` | 即时模式 GUI | `api_extensibility`、`runtime_performance`、`libraries_frameworks` | GUI 易用性、性能和扩展讨论较多 |
| 45 | P2 | `iced-rs/iced` | 响应式 GUI | `api_extensibility`、`learning_curve`、`libraries_frameworks` | 架构学习成本、类型设计和 GUI 生态反馈具有价值 |
| 46 | P2 | `tauri-apps/tauri` | 桌面应用框架 | `libraries_frameworks`、`tooling_documentation`、`compile_time` | 用户规模大且构建体验丰富，但产品与平台问题噪声较高 |
| 47 | P2 | `DioxusLabs/dioxus` | 跨平台 UI 框架 | `learning_curve`、`api_extensibility`、`libraries_frameworks` | 新兴 UI 框架的学习、API 和生态体验语料丰富 |
| 48 | P2 | `leptos-rs/leptos` | 全栈 Web 框架 | `type_system`、`learning_curve`、`compile_time` | 响应式类型、宏、编译和学习体验讨论具有补充价值 |
| 49 | P2 | `nushell/nushell` | Shell 与终端工具 | `runtime_performance`、`api_extensibility`、`community` | 大型终端应用和活跃社区可补充真实用户反馈 |
| 50 | P2 | `BurntSushi/ripgrep` | 命令行搜索工具 | `runtime_performance`、`tooling_documentation`、`api_extensibility` | 性能分析质量高，但当前待处理 Issue 规模相对较小 |

以上名称使用 2026-07-30 GitHub 返回的规范仓库名。其中：

- 原 `launchbadge/sqlx` 当前为 `transact-rs/sqlx`；
- 原 `rustwasm/wasm-bindgen` 当前为 `wasm-bindgen/wasm-bindgen`；
- Rocket 当前规范名为 `rwf2/Rocket`。

### 启用策略

50 个仓库不应一次性全量回填。生产服务器只有两核 4 GB，且 GitHub API 存在主限流和
次级限流，建议：

1. 保持现有 5 个仓库持续增量采集；
2. 先对新增 P0 仓库做小规模采集和人工抽检，再逐个历史回填；
3. P0 稳定后按领域分批接入 P1，每批 3～5 个仓库；
4. P2 每个仓库先抽取 200～500 条语料，统计非空标签率、机器人比例、纯日志/代码比例；
5. 只有能补充现有标签或领域缺口的 P2 仓库才长期启用；
6. 训练集继续使用每仓库上限，并进一步监控来源类型、时间和标签分布，避免超大仓库
   主导样本。

优先度和候选池应至少每季度复核一次；仓库归档、迁移、长期停更或有效标签率过低时，
可在数据库中停用，但保留历史数据和游标。

## 采集内容

- Issue
- Pull Request
- Issue comment
- PR 普通评论与 Review comment

Issue 和 PR 的标题、正文作为标注目标；评论以父级 Issue/PR 标题作为辅助上下文，
当前评论作为标注目标。

## 标签体系

当前标签体系版本为 `rust-aspects-v2`，包含 13 个方面：

| 类别 | 标签 | 含义 |
| --- | --- | --- |
| Language | `ownership` | 所有权、借用、生命周期、移动语义和借用检查器 |
| Language | `type_system` | 静态类型、类型推导、泛型和 trait |
| Language | `safety` | 内存安全、线程安全、数据竞争和 unsafe 代码 |
| Language | `runtime_performance` | 运行速度、延迟、吞吐量、内存和 CPU 效率 |
| Experience | `learning_curve` | Rust 的入门难度、理解成本和掌握时间 |
| Experience | `compile_time` | 初次构建、增量编译、重新编译和 CI 构建速度 |
| Experience | `diagnostics_debugging` | 编译错误、修复建议、调试和问题定位体验 |
| Experience | `tooling_documentation` | IDE、Clippy、rustfmt、教程和 API 文档质量 |
| Engineering | `readability_maintainability` | 代码的可读性、可理解性、可修改性和可维护性 |
| Engineering | `api_extensibility` | API 的直观性、组合性、复用性和扩展能力 |
| Ecosystem | `package_manager` | Cargo、依赖解析、版本、workspace、feature 和发布体验 |
| Ecosystem | `libraries_frameworks` | 第三方库和框架的质量、成熟度及领域覆盖 |
| Ecosystem | `community` | 社区活跃度、交流氛围、维护者响应和协作体验 |

每个已提及方面使用一种情感类别：

- `negative`：明确负面
- `neutral`：明确提及但无明显褒贬
- `positive`：明确正面

## 核心标注原则

- 当前提示词版本为 `aspect-sentiment-zh-v8`，标签体系仍为 `rust-aspects-v2`。
- 先判断目标作者是否实质讨论某方面，再独立判断情感；机制说明、具体问题和设计建议
  可以标为 `neutral`，关键词或机械提及本身不足以成立方面。
- 明确的使用困难、负面影响或改进收益可以表达情感，不要求出现情绪词。
- 上下文用于理解；只有目标作者明确采纳的命题才可归于作者，不自动继承标题或引用的情感。
- 不将具体项目 Bug 强行归入 Rust 标签体系。
- Cargo 相关体验归入 `package_manager`；普通库 Bug 不自动归入
  `libraries_frameworks`。
- 程序运行性能、编译速度和工具响应速度分别归入对应标签。
- 标注结果保留标签体系、Prompt、模型和原始语料版本，便于追溯与重新标注。

## 相关文档

- [需求说明](需求说明.md)
- [运行手册](运行手册.md)
- [历史审计材料](docs/audits/2026-09-13/审核报告.md)
