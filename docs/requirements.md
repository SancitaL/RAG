# Technical Research Knowledge Assistant — 需求说明

## 正式依据

本项目执行 **Advanced RAG Engineering — 40-Day Project Version 1**，每天 5–8 小时，总投入约 200–320 小时，Day 40 完成 Version 1。

本说明同步正式计划。原有 EvidenceRAG 企业中文知识库、PostgreSQL + pgvector 和另一版日程不再作为当前实现依据。Python 安装依赖统一保存在根目录 `requirements.txt`。

## 项目定位与用户

为学习者和技术研究人员提供基于证据的技术文档与论文问答。知识库包括 PyTorch/深度学习文档、Transformer/Attention 论文、量化研究、机器学习报告及个人 Markdown 笔记。

用户可以导入资料、提问、核对来源；开发者可以检查各阶段输出、运行对照实验并分析失败案例。

## 功能需求

### 文档与索引

- 支持 PDF、TXT、Markdown 的加载、解析、清洗和切分。
- 保存 Document ID、Chunk ID、正文、来源和可用页码等元数据。
- 检查空正文、解析失败与 Embedding 输入超限。
- 使用 Qdrant 保存向量及 Payload，支持持久化和元数据过滤。
- 后续实现文档更新、Upsert、Delete、版本与索引同步。

### 检索

- 实现并比较 BM25、Dense、Hybrid 和 Cross-Encoder Reranking。
- 手写基础精确检索、BM25 核心算法和 RRF。
- 使用 FAISS 比较 Exact Search 与 ANN，学习 Flat、IVF、HNSW。
- 支持来源范围、元数据过滤、去重和 Parent-Child Retrieval。
- 通过实际结果评估召回、排序、延迟及资源开销。

### 生成

- 支持可配置的 API 或本地 LLM 后端。
- 实现上下文构造、预算控制和基于证据的生成。
- 引用可定位到实际片段，证据不足时拒答或说明不确定性。
- 实现基础 Query Rewriting 和多轮问答。
- 外部文档作为证据处理，不能覆盖系统指令。

### 评测与工程

- 建立 80–120 条经过人工检查的评测问题，包含正负问题。
- 实现 Precision@K、Recall@K、Hit Rate、MRR、nDCG 等指标。
- 评估正确性、Faithfulness、Answer Relevancy 和引用支持情况。
- 使用 Ragas/LLM 裁判时记录配置并人工复核。
- 完成 FastAPI、Streamlit、缓存、安全检查、性能记录与 Docker。
- 维护必要测试、依赖、配置、README、实验报告和 GitHub 仓库。

## 实验要求

至少比较 Chunk Size、BM25/Dense/Hybrid、是否重排、Top-K、查询改写、是否使用 RAG、Exact/ANN，以及 Fixed/Adaptive Retrieval。

使用真实数据，保存配置、指标、时间和可获得的成本；适用时固定随机种子。实验采用可比较的问题与语料，分析误差及质量/性能取舍，不虚构数值。

Day 3 的字符切分结构实验与 Day 26 的 token 长度实验分别记录，不混用长度单位。

具体后续评测数据集尚待 Day 22 结合技术知识库设计；MIRACL/HotpotQA 不作为本版本已经确定的强制主数据集。

## 技术约束与范围

- Qdrant 为主要向量数据库；FAISS 主要用于 ANN 实验。
- 先理解和手写核心算法，Day 21 再进行 LangChain 集成。
- 不同时深入多个 RAG 框架。
- 不要求从零训练或微调 LLM。
- Advanced RAG 只实现一个轻量 Adaptive Retrieval 或失败重试实验。
- 不擅自增加 GraphRAG、Multi-Agent 或大型模型训练。
- 数据来源、许可和获取方式需要记录；密钥和本地数据库不提交 Git。
- 当前本地模式用于开发和小规模实验，不能据此宣称生产性能或完整访问控制。

## 最终验收

只有实际满足以下条件，才标记 Version 1 完成：

1. 文档导入、Metadata、BM25/Dense/Hybrid 和 Reranking 可运行。
2. LLM 问答可返回可验证引用，并合理处理无答案问题。
3. 基础查询改写、多轮对话、增量索引、缓存和安全检查已实现。
4. 人工检查的评测集、检索与生成指标、实际对照实验和报告齐全。
5. FastAPI、Streamlit、Docker 和关键测试可运行。
6. README、依赖、配置、使用示例与限制完整，代码上传 GitHub。
7. 从新环境按文档完成复现。

Recall@5 0.85、引用支持率 90% 只可作为结合数据难度的优化目标，不是预先保证。

## 进度管理

以正式路线的 Day 7、14、21、28、35、38、39、40 为里程碑。每天区分静态检查、实际运行、实验结论及 Git/笔记收尾；不提前标记完成。

