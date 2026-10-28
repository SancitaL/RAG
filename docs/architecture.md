# Technical Research Knowledge Assistant — 架构

## 系统位置

索引阶段在文档导入或更新时执行；查询阶段在用户提问时执行。文档索引不是模型训练。

```text
索引阶段
PDF / TXT / Markdown
→ 加载、清洗并保留来源
→ Chunk
→ 检查实际 token 长度
→ Embedding
→ Point：ID + vector + payload
→ Qdrant

查询阶段
用户问题与过滤条件
→ 使用相容模型编码问题
→ 检索候选
→ 后续：融合、重排、上下文构造
→ 后续：LLM 生成、引用与拒答
```

## 当前已实现的边界

- 文档加载：`ingestion/document_loader.py`。
- 字符切分与 Chunk：`ingestion/chunking.py`。
- Embedding：`fundamentals/embedding_basics.py`。
- NumPy 精确检索：`retrieval/vector_search.py`。
- Qdrant 存储与查询：`retrieval/vector_store.py`。

当前 Day 6 演示主入口同时执行导入、编码、写入和查询；尚未形成独立的导入与查询入口。Day 7 集成基础检索系统。后续 LLM、融合、重排等节点属于计划功能。

## 数据结构

### Document 与 Chunk

Document 保存正文、来源、格式和页码。PDF 当前按页返回 Document，因此其 `doc_id` 包含物理页码。

Chunk 保存：

- `chunk_id`
- `document_id`
- `text`
- `source`
- `file_type`
- `page_number`
- `chunk_index`

当前 Chunk ID 根据页面 Document ID 与片段位置生成。它在相同标识及切分配置下可重复生成，但不能自动识别内容变化；切分变化后的旧片段清理留待 Day 31。

### Qdrant Point

- `id`：通过 `uuid5(NAMESPACE_URL, chunk_id)` 生成。
- `vector`：片段 Embedding 的 Python 数值列表。
- `payload`：通过 `asdict(chunk)` 保存全部 Chunk 字段。

相似度分数属于具体查询结果，不作为片段的固定属性。数据库按 ID 定位记录；返回正文时从同一 Point 的 Payload 读取。

## 存储配置

当前客户端：

```python
QdrantClient(path="indexes/qdrant_day06")
```

| Collection         | 用途                 | 维度 | 度量   |
| ------------------ | -------------------- | ---: | ------ |
| day06_demo         | 单记录写入与读取     |    2 | Cosine |
| day06_ranking      | 排序、过滤与重新打开 |    2 | Cosine |
| day06_paper_minilm | 真实论文片段         |  384 | Cosine |

二维实验与真实 Embedding 使用不同 Collection。向量维度、编码模型及配置必须相容，仅检查维度不能保证语义空间相容。

本地数据库生成目录由 Git 忽略，实验配置和结果 JSON 保留。客户端通过 `try/finally` 关闭后再重新打开同一路径。

## 函数职责

- `make_point()`：验证维度、形状、有限数值、非零向量和非空正文/ID，再创建记录。
- `ensure_collection()`：创建或验证单个未命名向量字段及 Cosine 配置。
- `write_points()`：拒绝同一批次内部重复 ID，执行 Upsert。
- `search_points()`：验证问题向量，调用 `query_points()`，返回 `response.points`。
- `retrieve()`：Qdrant 按已知 ID 读取记录的接口，与相似度检索区别开。

Upsert 更新相同 ID 的记录，但不自动删除已经不再属于当前文档的旧 Chunk。

## 过滤与来源

`source` 位于 Payload 顶层。查询通过 `FieldCondition` 和 `MatchValue` 限定来源，再在满足条件的记录中返回 Top-K。

来源过滤可以限制搜索范围；用户身份、可信权限规则及文档级访问控制尚未实现，不能把客户端传入的来源参数当作完整授权机制。

## Day 6 实验依据

- 人工向量得到 b/a/c = 1.0/0.8/0.0。
- 过滤到 Markdown 来源后得到 a/c。
- 关闭重新打开后仍有 3 条记录，并可过滤检索。
- 8 组 Point 非法输入被拒绝。
- 论文第 2 页得到 12 个片段，最长 128 tokens，384 维。
- Top-3 索引 1/0/3，分数 0.6976/0.5200/0.5029，与 Day 5 一致。
- 实际输出保存到 `results/day06_vector_store.json`。

这些结果支持当前样例的行为判断，不代表大规模性能或总体语义检索准确率。当前实验未测量 ANN 性能。

## 失败排查

| 现象                 | 优先检查                                |
| -------------------- | --------------------------------------- |
| 原文缺失或乱码       | PDF 解析与清洗输出，对照原文            |
| 片段缺少完整条件     | 切分边界、大小与相邻片段                |
| 末尾信息未影响向量   | 实际 token 数及模型截断                 |
| 数据库拒绝向量       | 形状、维度、非有限值及 Collection 配置  |
| 分数对应错误正文     | Chunk、向量和 Point 的顺序及 ID/Payload |
| 正确证据未进入 Top-K | 索引内容、编码模型、过滤和排序          |
| 重启后没有数据       | 是否使用同一路径、是否先读取再写入      |
| 文档更新后出现旧内容 | Chunk 身份、版本与旧记录删除策略        |

## 后续正式路线

Day 7 集成基础检索；Day 8–14 加入 BM25、Dense、FAISS 实验、Hybrid 和 Reranking；Day 15–21 加入 LLM、上下文、引用、多轮与框架集成。

Day 22–28 建立评测，Day 29–35 服务化、安全、缓存、性能和部署，Day 36–40 完成有限高级实验、测试、报告和复现交付。

