# GitHub Rust 社区情感分析

面向 GitHub Rust 社区的研究数据流水线，采集指定仓库的 Issue、PR 及评论，
生成 13 个方面的情感标签，为模型训练与社区讨论分析提供可追溯语料。

```text
GitHub 采集 → 清洗去重 → 普通/词典采样 → LLM 标注 → 抽检与训练数据准备
                                                     → 方面内 BERTopic 分析
```

已实现采集、清洗、采样、GLM/DeepSeek 标注、运行查询及 BERTopic 批次试验。
已提供 `sentiment_facts` 查询表及只读 HTTP API，供并列的 `Developer-Voice` 前端使用。
BERT/ONNX 目前仅预留接口，训练与自动推理尚未接入。
使用 Python ≥ 3.11、MySQL；部署目标为 Ubuntu 服务器。

- [运行手册](docs/运行手册.md)：服务器运行与查询命令。
- [项目元数据](docs/项目元数据.md)：版本、配置、仓库选取、标签口径与数据记录。
- [需求说明](docs/需求说明.md)：项目范围、数据约束与实现定位。
- [模型离线准备](docs/模型离线准备.md)：本地下载和传送主题编码器。
- [前端 Ubuntu 部署](https://github.com/Wangyuhan29/Developer-Voice/blob/main/docs/ubuntu-deployment.md)：API、systemd 与 Nginx 配置。
- [历史审计材料](docs/audits/)：采样、标注与清洗复核记录。
- [文档维护约定](docs/AGENTS.md)。
