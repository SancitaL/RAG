# Advanced RAG Engineering — 40 天任务清单

正式依据：Advanced RAG Engineering — 40-Day Project Version 1。每天 5–8 小时，Day 40 完成 Version 1。后文保留原正式路线的每日内容与文件路径。

## 当前进度（2026-10-2）

| Day  | 功能与实验                                                                       | 收尾状态                                  |
| ---- | -------------------------------------------------------------------------------- | ----------------------------------------- |
| 1    | RAG 控制流、关键词检索、上下文与占位回答已验证                                   | 以 Git 记录核对                           |
| 2    | TXT/Markdown/PDF 加载、清洗与来源；异常场景部分由学习者确认                      | 以 Git 记录核对                           |
| 3    | 固定、重叠、递归切分及真实 PDF 结构比较已验证                                    | 以 Git 记录核对                           |
| 4    | Pooling、归一化、真实 Embedding 与截断实验已验证                                 | 以 Git 记录核对                           |
| 5    | 相似度、Top-K 与真实论文单问题检索已验证                                         | 以 Git 记录核对                           |
| 6    | Qdrant 存储、读取、排序、过滤、重新打开、8 组 Point 异常输入及真实论文检索已验证 | 结果文件已存在；依赖/文档同步与提交待确认 |
| 7–40 | 尚未开始                                                                         | 按下方正式路线执行                        |

Day 6 尚未逐项展示批次重复 ID、已有 Collection 配置不匹配和查询异常输入分支的运行结果。新环境完整安装复现尚未验证。

不将“代码静态正确”“运行通过”“单个实验有效”“最终交付完成”视为同一状态。

## 每日教学与收尾

1. 今日目标：系统位置、功能、文件、验收标准。
2. 理论与数学：基础定义、变量、推导、数值例子、限制与工程用途。
3. 编程实践：2–4 个 Stage，框架/TODO → 学习者实现 → 检查代码和输出 → 下一阶段。
4. 代码检查：明确原因、最小修改及预期结果。
5. 实验与验收：真实运行、必要测试、指标与限制。
6. Git：检查实际文件、暂存、简洁英文提交信息。
7. 进度总结：掌握内容、实际结果、未确认任务与下一天。

代码按学习者要求在聊天中提供，核心实现由学习者完成；不会时补充解释和局部示例。每天包含少量面试问题，每周复盘。结果与配置保存，适用时固定随机种子。

## 硬性里程碑

| Day | 交付                                 |
| --- | ------------------------------------ |
| 7   | 基础文档向量检索可运行               |
| 14  | Hybrid + Rerank 完成                 |
| 21  | End-to-End RAG 可回答并引用          |
| 28  | Evaluation Pipeline 完成             |
| 35  | API、UI、Docker 可运行               |
| 38  | 核心测试与报告完成                   |
| 39  | 功能冻结，文档齐全                   |
| 40  | 新环境验收、GitHub 与 Version 1 交付 |

工期紧张时降低高级实验复杂度，保留核心检索、生成、评测、服务与交付。不得自行延期或增加计划外大型功能。

---

# 6. 固定 40-Day Roadmap

以下为 Version 1 的正式路线。后续不得随意改变每日主要任务。

## Week 1 — RAG Foundations & Text Processing

**目标：不依赖高级 RAG 框架，独立完成基础文档向量检索。**

### Day 1 — RAG Fundamentals
- RAG 原理与系统结构。
- RAG vs Fine-tuning。
- 为什么 LLM 需要外部知识。
- Indexing Pipeline 和 Query Pipeline。
- 创建项目、基础架构说明。
- 文件：`fundamentals/rag_basics.py`、`docs/architecture.md`。

### Day 2 — Document Loading & Parsing
- Tokenization、Context Window。
- PDF、TXT、Markdown 结构。
- 文档清洗、Metadata、Source ID。
- PDF 页码、解析错误。
- 文件：`ingestion/document_loader.py`。

### Day 3 — Chunking Strategies
- Fixed-size Chunking。
- Overlap。
- Recursive Chunking。
- 语义边界和 Chunk 粒度。
- Chunk Size 对准确率、成本、上下文完整性的影响。
- 文件：`ingestion/chunking.py`。

### Day 4 — Embedding Fundamentals
- Embedding 的原理。
- 向量空间与语义表示。
- Embedding Dimensionality。
- Pooling、Normalization。
- Sentence Transformers。
- 文件：`fundamentals/embedding_basics.py`。

### Day 5 — Similarity Search
- Dot Product。
- Cosine Similarity。
- Euclidean Distance。
- Top-K Search。
- 使用 NumPy 手写精确向量检索。
- 文件：`retrieval/vector_search.py`。

### Day 6 — Vector Store
- 向量索引和向量数据库基础。
- Document ID、Chunk ID。
- Metadata。
- 持久化和过滤。
- 文件：`retrieval/vector_store.py`。

### Day 7 — Baseline Retrieval System
- Document → Chunk → Embed → Retrieve。
- 集成测试与错误分析。
- 第一版不带 LLM 的检索系统。
- 文件：`experiments/baseline_retrieval.py`。

**Week 1 交付：可以加载文档并返回相关 Top-K Chunks 的基础检索系统。**

---

## Week 2 — Retrieval Engineering

**目标：实现 BM25、Dense Retrieval、Hybrid Search 和 Reranking。**

### Day 8 — Sparse Retrieval & BM25
- TF、IDF、倒排索引。
- BM25 数学原理。
- 参数 k1、b 和文档长度归一化。
- 手写 BM25 核心算法。
- 文件：`retrieval/bm25_retriever.py`。

### Day 9 — Dense Retrieval
- Bi-Encoder。
- Semantic Retrieval。
- Embedding 模型选择。
- Dense 与 Sparse 的差异。
- 文件：`retrieval/dense_retriever.py`。

### Day 10 — ANN & FAISS
- Exact KNN vs ANN。
- FAISS。
- Flat、IVF、HNSW 的基本机制。
- Recall、Latency、Memory Trade-off。
- 文件：`experiments/index_benchmark.py`。

### Day 11 — Hybrid Retrieval
- Sparse + Dense。
- 分数归一化与排名融合。
- Reciprocal Rank Fusion。
- 手写 RRF。
- 文件：`retrieval/hybrid_retriever.py`。

### Day 12 — Reranking
- Bi-Encoder vs Cross-Encoder。
- Candidate Retrieval 和 Reranking。
- Top-K 与候选数量的选择。
- 质量与延迟权衡。
- 文件：`retrieval/reranker.py`。

### Day 13 — Retrieval Pipeline
- Metadata Filtering。
- Duplicate Removal。
- Parent-Child Retrieval。
- 来源和文档范围。
- 文件：`retrieval/retrieval_pipeline.py`。

### Day 14 — Retrieval Benchmark
比较四组系统：

1. BM25。
2. Dense。
3. Hybrid。
4. Hybrid + Reranking。

评估 Recall@K、MRR、Latency 等指标。

文件：`experiments/retrieval_comparison.py`。

**Week 2 交付：可配置的高质量混合检索系统及第一份检索对比实验。**

---

## Week 3 — End-to-End RAG

**目标：连接 LLM，完成带引用的真实文档问答。**

### Day 15 — LLM Integration
- LLM Inference。
- Token、Temperature、Top-P。
- API 与 Local Model。
- 统一 LLM 调用接口。
- 文件：`generation/llm_client.py`。

### Day 16 — RAG Prompt Engineering
- Context Injection。
- Prompt Template。
- Context Budget。
- 指令与外部证据分离。
- 文件：`generation/prompt_builder.py`。

### Day 17 — Basic RAG Pipeline
- Retrieval → Context → Generation。
- 基础 RAG Chain。
- End-to-End QA。
- 文件：`rag_pipeline.py`。

### Day 18 — Citations & Grounding
- Grounded Generation。
- Source Attribution。
- 引用粒度。
- 证据不足时拒答。
- 文件：`generation/citation.py`。

### Day 19 — Query Transformation
- Query Rewriting。
- Multi-Query Retrieval。
- Query Decomposition。
- 文件：`retrieval/query_transform.py`。

### Day 20 — Conversational RAG
- 对话历史管理。
- 指代消解。
- Multi-turn Retrieval。
- Context Contamination。
- 文件：`generation/conversation.py`。

### Day 21 — LangChain Integration
- LangChain Retriever Interface。
- Chain Integration。
- 与手写 RAG 实现比较。
- 端到端集成验收。
- 文件：`integrations/langchain_pipeline.py`。

**Week 3 交付：能够从文档中检索证据、通过 LLM 回答并返回来源引用的 RAG 系统。**

---

## Week 4 — Evaluation & Optimization

**目标：建立可量化的评测体系，并通过实验改进系统。**

### Day 22 — Evaluation Dataset
- QA Benchmark 设计。
- Ground Truth。
- 正负问题。
- 数据泄漏。
- 建立至少 80–120 条经过人工检查的测试问题。
- 文件：`evaluation/dataset_builder.py`。

### Day 23 — Retrieval Metrics
- Precision@K。
- Recall@K。
- Hit Rate。
- MRR。
- nDCG。
- 手写核心评价指标。
- 文件：`evaluation/retrieval_metrics.py`。

### Day 24 — Generation Evaluation
- Answer Correctness。
- Faithfulness。
- Answer Relevancy。
- LLM-as-Judge。
- 文件：`evaluation/generation_metrics.py`。

### Day 25 — Ragas
- Ragas 的使用。
- 自动评测结果解释。
- LLM-as-Judge 偏差。
- 人工复核。
- 文件：`evaluation/ragas_eval.py`。

### Day 26 — Chunking Experiments
- 256 / 512 / 1024 Tokens。
- Chunk Overlap。
- Chunking Strategy Ablation。
- 文件：`experiments/chunking_experiment.py`。

### Day 27 — Retrieval Tuning
- Top-K。
- Candidate Pool。
- Rerank Top-N。
- Context Budget。
- Latency / Quality Trade-off。
- 文件：`experiments/retrieval_tuning.py`。

### Day 28 — End-to-End Evaluation
- 完整 RAG 评测。
- Retrieval Error vs Generation Error。
- Failure Taxonomy。
- 错误分析与评测报告。
- 文件：`evaluation/evaluate_pipeline.py`、`docs/evaluation_report.md`。

**Week 4 交付：完整评测集、自动化评测代码和可复现的质量优化实验报告。**

---

## Week 5 — Production RAG Engineering

**目标：构建可以实际使用的服务。**

### Day 29 — FastAPI Backend
- REST API。
- Request / Response。
- 错误处理。
- `/health`、`/query`、`/sources`。
- 文件：`api/main.py`。

### Day 30 — Streamlit UI
- 文档上传。
- 问题输入。
- 回答显示。
- 来源展示。
- 文件：`app/streamlit_app.py`。

### Day 31 — Incremental Indexing
- Document ID。
- Upsert、Delete。
- 文档版本。
- 索引同步。
- 文件：`ingestion/index_manager.py`。

### Day 32 — RAG Security
- Prompt Injection。
- Retrieved Documents 的信任边界。
- 文档级访问控制。
- Sensitive Data。
- Security Tests。
- 文件：`security/guards.py`。

### Day 33 — Caching
- Embedding Cache。
- Retrieval Cache。
- Cache Invalidation。
- 数据更新后的缓存一致性。
- 文件：`core/cache.py`。

### Day 34 — Performance Engineering
- Profiling。
- Batching。
- Async I/O。
- Throughput。
- P50、P95 Latency。
- Load Testing。
- 文件：`benchmark/load_test.py`。

### Day 35 — Docker & Deployment
- Docker。
- Environment Variables。
- Dependency Management。
- Cost / Token Tracking。
- 服务启动与复现。
- 文件：`Dockerfile`、`docker-compose.yml`。

**Week 5 交付：具有 API、Web UI、安全检查、基础性能优化和部署说明的完整 RAG 应用。**

---

## Final Stage — Advanced RAG & Delivery

### Day 36 — Advanced RAG
- Self-RAG 原理。
- Corrective RAG 原理。
- Adaptive RAG。
- Agentic RAG 的适用场景。
- 只实现一个轻量 Adaptive Retrieval 或检索失败重试实验。
- 文件：`experiments/adaptive_retrieval.py`。

### Day 37 — Final Ablation Study
- 固定实验配置。
- 对比检索策略。
- 评估生成质量。
- 分析 Accuracy、Latency、Token Cost。
- 文件：`experiments/final_ablation.py`。

### Day 38 — Testing & Final Optimization
- 单元测试。
- Integration Tests。
- API Tests。
- Failure Cases。
- Bug Fixes。
- 文件：`tests/`、`docs/final_report.md`。

### Day 39 — Documentation & Reproduction
- README。
- Architecture。
- Setup Instructions。
- CLI / API Examples。
- 实验结果。
- 面试讲解和项目演示。
- 功能冻结。

### Day 40 — Final Acceptance & Delivery
- 从新环境复现。
- 完整系统测试。
- 核验所有实验结果。
- GitHub 代码交付。
- 最终报告。
- Version 1 完成。

**Day 40 不允许增加新的核心功能。**

---



## Day 6 工作记录

- 已验证：Point 检查与转换、Collection 创建与配置检查的正常路径、Upsert、按 ID 读取、人工向量排序、来源过滤、关闭后重新打开。
- 非法 Point：维度、形状、长度、NaN、Infinity、零向量、空 ID、空正文均拒绝。
- 真实论文：12 个 Chunk，最长 128 tokens，向量形状 (12, 384)，数据库记录数 12。
- Top-3：chunk 1/0/3，分数 0.6976/0.5200/0.5029，与 Day 5 一致。
- 局限：单页、单问题，不作为整体检索准确率；主入口仍执行重新编码和写入。
- 结果：results/day06_vector_store.json。
- 待完成：依赖和文档更新的本地保存、Git 提交与剩余分支测试确认。
- 下一步：Day 7 基础检索系统集成。



## 历史初始化记录（旧计划，已被正式 Version 1 取代）

以下原文仅保留历史，不作为当前路线或完成状态。涉及 MIRACL、pgvector、旧日期和旧日程的决定不适用于本版本。

## Day 1 工作记录

- **日期**：2026-10-26
- **完成**：明确项目用户、核心流程、文档类型、功能范围、边界与验收标准；建立项目目录和文档。
- **待完成**：选择并固定具体 Python 依赖、模型/API 与解析方案；按计划在实现对应阶段前决定。
- **阻塞**：暂无。
- **下一步（Day 2）**：梳理 RAG 端到端架构和主要失败模式，产出架构图及失败模式清单。

## Day 2 工作记录

- **日期**：2026-10-26
- **完成**：整理离线文档处理与在线问答流程、关键数据、系统边界、主要失败模式及排查方式；列出待实验确定的设计项。
- **产物**：[架构与失败模式](docs/architecture.md)
- **待完成**：后续实现时用实际样例和检索记录验证失败模式；模型、切分和排序参数留待实验。
- **阻塞**：暂无。
- **下一步（Day 3）**：准备 MIRACL 中文 topics/qrels、匹配 corpus 与小规模 HotpotQA 样本，并记录来源和许可。
