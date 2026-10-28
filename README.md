# Advanced RAG Engineering

Technical Research Knowledge Assistant：基于技术文档、研究论文和 Markdown 学习笔记的 RAG 知识助手。

项目按 **Advanced RAG Engineering — 40-Day Project Version 1** 推进，每天 5–8 小时，Day 40 完成可运行、测试、复现并上传 GitHub 的 Version 1。

## 当前状态

Day 1–5 的核心功能和已展示实验已通过检查。Day 6 已验证 Qdrant 写入、重复 ID 更新、按 ID 读取、向量排序、来源过滤、本地数据库重新打开及一组非法输入检查，并完成单页论文检索。

每日笔记、结果文件和 Git 提交分别核对，不因功能运行成功自动视为全部收尾完成。

当前演示入口仍会重新编码并写入论文片段。Day 7 将组织导入与查询流程，形成第一版基础检索系统。当前尚未实现 LLM 问答。

## 环境

已运行实验的环境：

- macOS，Python 3.13.1。
- CPU 编码。
- Python 依赖版本见 `requirements.txt`。
- Embedding：`sentence-transformers/all-MiniLM-L6-v2`。
- 输出：384 维，归一化；当前模型输入上限为 256 tokens。
- Qdrant：Python 客户端本地持久化模式，目录为 `indexes/qdrant_day06/`。

环境版本来自当前虚拟环境；新环境安装与完整复现仍待后续验证。

## 安装

在项目根目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

首次加载 Embedding 模型需要获取模型文件。Qdrant 本地模式无需单独启动数据库服务，也无需数据库 API Key。

## 运行 Day 6 实验

准备支持文本提取的论文 PDF，放到：

```text
data/samples/example.pdf
```

当前实验使用 Transformer 论文 *Attention Is All You Need* 的物理第 2 页。更换 PDF 后，片段数量、token 数和检索结果可能不同。

```bash
python -m retrieval.vector_store
```

当前脚本完成解析、递归切分、编码、写入和检索，并保存结果：

```text
results/day06_vector_store.json
```

本地数据库属于生成数据，不提交 Git。运行时需先关闭使用同一路径的其他本地 Qdrant 客户端。

## Day 6 实际结果

- 12 个片段，最长 128 tokens，向量形状为 `(12, 384)`。
- Collection：`day06_paper_minilm`；度量：Cosine。
- 问题：`Why are recurrent models difficult to parallelize?`
- Top-3 片段索引：1、0、3；分数：0.6976、0.5200、0.5029。
- 第一名包含隐藏状态顺序依赖限制样本内部并行化的直接依据。

这是单页、单问题实验，尚不能代表整体检索质量。完整 Payload 和实际配置见结果 JSON。

## Version 1 技术路线

- 文档处理：PDF、TXT、Markdown；PyMuPDF。
- 基础计算：NumPy、PyTorch、Sentence Transformers。
- 检索：BM25、Dense、Hybrid、RRF、Cross-Encoder Reranking。
- 向量数据库：Qdrant；FAISS 用于 ANN 原理与性能实验。
- 生成：可配置的 API 或本地 LLM，引用、拒答、查询改写和多轮问答。
- 工程集成：LangChain 安排在基础实现之后。
- 评测：检索与生成指标、Ragas、人工复核、真实对照实验。
- 服务与交付：FastAPI、Streamlit、pytest、Docker、GitHub。

后续功能按日程逐步实现，上述清单不代表当前已全部完成。

## 文档

- [需求与验收](docs/requirements.md)
- [架构与当前实现](docs/architecture.md)
- [40 天正式任务路线](TASKS.md)
- [学习笔记](docs/learning_notes.md)

## 限制

当前只验证了英文论文样例，尚未验证中文或跨领域检索质量。当前来源过滤属于检索条件，不等于已经完成文档级访问控制。增量索引、部署、安全、系统评测和 LLM 生成将按正式计划推进。

