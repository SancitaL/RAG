# Learning Notes

## Day 1 — RAG Fundamentals

- RAG 先检索外部证据，再结合问题与证据生成回答。
- 索引阶段在文档导入或更新时执行；查询阶段在用户提问后执行。
- 今天使用不同词集合的交集大小作为检索分数。
- 关键词完全匹配不能处理同义表达、词形变化和标点差异。
- 检索结果保留文档 ID、正文和来源。
- build_context 将检索结果整理成带来源的上下文字符串。
- 当前回答函数只列出证据 ID，尚未调用 LLM。
- 没有匹配证据时返回证据不足提示。

### 运行结果

- 优化器问题：返回 d2。
- model：返回 d2、d3，同分时保持原顺序。
- banana 和空问题：返回空列表。
- top_k 为 0：返回空列表。

### Python 知识

- set 的 & 运算计算交集。
- sort 是稳定排序。
- join 将字符串列表连接成一个字符串。
- repr 将换行等字符显示为可见的转义形式。
- print 的返回值是 None。

## Day 2 — Document Loading & Parsing

### 今日目标

实现 TXT、Markdown 和 PDF 加载，将不同格式转换为统一的 `Document`，保留正文、来源和 PDF 页码，为后续 Chunking 做准备。

### 1. 文档加载在 RAG 中的位置

索引阶段的基本流程：

```text
文档 → 加载与解析 → 清洗 → 切分 → 建立检索表示 → 保存索引
```

加载器负责提取文本与元数据，切分器负责将文本拆成适合检索的片段。今天返回的 PDF 页面还不是最终的检索 Chunk。

### 2. 统一文档结构

```python
@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    source: str
    file_type: str
    page_number: int | None = None
```

| 字段          | 用途                                         |
| ------------- | -------------------------------------------- |
| `doc_id`      | 标识加载得到的文本单元                       |
| `text`        | 保存正文                                     |
| `source`      | 保存来源，用于追踪与引用                     |
| `file_type`   | 标识 `txt`、`markdown` 或 `pdf`              |
| `page_number` | PDF 物理页序号，从 1 开始；其他格式为 `None` |

当前 ID 约定：

```text
samples/example.txt::document
samples/example.md::document
samples/example.pdf::page:1
```

同一知识库中的来源标识需要唯一并保持一致。版本管理和增量更新留到后续实现。

### 3. TXT 与 Markdown

使用 UTF-8 读取：

```python
text = path.read_text(encoding="utf-8")
```

Markdown 本身也是文本，但标题、列表、代码围栏和缩进具有结构意义，需要保留。

不能用下面的方式清洗整篇文档：

```python
" ".join(text.split())
```

它会丢失段落换行和代码缩进。

### 4. 基础文本清洗

```python
def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    return text.strip()
```

先处理 `\r\n`，再处理单独的 `\r`，避免把一个换行错误地变成两个。

Python 字符串不可变，`replace()` 和 `strip()` 返回新字符串，需要保存返回值。

当前清洗规则只统一换行、去掉全文首尾空白，不删除正文内部的空格和换行。更复杂的清洗需要结合文档结构设计，避免破坏内容。

### 5. PDF 按页解析

使用 PyMuPDF：

```python
with pymupdf.open(path) as pdf:
    for page_number, page in enumerate(pdf, start=1):
        text = page.get_text("text", sort=True)
```

- `with` 在处理结束后关闭文档。
- `enumerate(..., start=1)` 生成从 1 开始的物理页序号。
- `sort=True` 尝试根据位置整理阅读顺序，但不能保证复杂排版完全正确。
- 跳过空页后，保留后续页面原来的页码。

PDF 主要保存页面布局，提取文本可能出现多余空格、换行、词间空格丢失或阅读顺序问题。需要对照原文检查，不能仅以“没有报错”判断解析正确。

扫描型 PDF 可能只有图片，没有可直接提取的文本，需要 OCR。当前加载器不实现 OCR。

### 6. 错误处理

| 情况                     | 当前处理                            |
| ------------------------ | ----------------------------------- |
| TXT 或 Markdown 正文为空 | 抛出 `ValueError`                   |
| 文件不存在               | 保留 `FileNotFoundError`            |
| 文件编码不符合 UTF-8     | 保留解码错误                        |
| PDF 需要密码             | 抛出包含路径的 `ValueError`         |
| PDF 某页没有可提取文本   | 发出警告并跳过                      |
| 整份 PDF 没有可提取文本  | 抛出 `ValueError`，提示可能需要 OCR |
| PDF 损坏或无法解析       | 保留解析异常                        |

“没有提取到文本”不等于“文件为空”。加载失败也不应伪装成正常的空结果。

### 7. Tokenization 与 Context Window

Tokenization 将文本转换成模型使用的 token 序列。Token 不一定等于一个单词，不能用 `split()` 精确计算模型 token 数。

上下文预算需要考虑：

```text
回答规则 + 对话历史 + 当前问题 + 检索证据 + 输出预留
```

因此不能把整个知识库直接交给模型，需要通过切分和检索选择有限证据。

### 8. 编程中遇到的问题

曾使用：

```python
docs.append([doc])
```

这会形成嵌套列表：

```python
[[Document(...)], [Document(...)]]
```

正确写法是：

```python
docs.append(doc)
```

目标返回类型是 `list[Document]`。

使用 `repr(text)` 查看文本时，换行会显示成 `\n`；使用 `print(text)` 才会按实际换行排版。

### 9. 实验记录

- TXT：中英文读取正确，内部段落换行保留。
- 空白 TXT：抛出 `ValueError`。
- 不存在的文件：抛出 `FileNotFoundError`。
- 清洗函数：`\r\n`、`\r` 正确归一化，纯空白文本变为 `""`。
- Markdown：标题、列表、代码围栏和代码缩进保留。
- 论文 PDF：返回 15 个非空页面对象，来源、页码和格式字段正确；已对照原文确认内容。
- 空白 PDF、损坏 PDF：已自行确认完成，异常输出未在聊天中逐项核对。

### 10. 当前状态与下一步

TXT、Markdown 和普通文本 PDF 的加载功能已完成。依赖版本记录和 Git 提交尚待确认。

下一步学习 Chunking：比较片段大小、重叠和切分边界对检索及上下文完整性的影响。





