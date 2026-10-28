# Learning Notes

## Day 1 — RAG Fundamentals

### 1. 核心知识

- RAG 先检索外部证据，再结合问题与证据生成回答。
- 索引阶段在文档导入或更新时执行；查询阶段在用户提问后执行。
- 今天使用不同词集合的交集大小作为检索分数。
- 关键词完全匹配不能处理同义表达、词形变化和标点差异。
- 检索结果保留文档 ID、正文和来源。
- build_context 将检索结果整理成带来源的上下文字符串。
- 当前回答函数只列出证据 ID，尚未调用 LLM。
- 没有匹配证据时返回证据不足提示。

### 2. 运行结果

- 优化器问题：返回 d2。
- model：返回 d2、d3，同分时保持原顺序。
- banana 和空问题：返回空列表。
- top_k 为 0：返回空列表。

### 3. Python 知识

- set 的 & 运算计算交集。
- sort 是稳定排序。
- join 将字符串列表连接成一个字符串。
- repr 将换行等字符显示为可见的转义形式。
- print 的返回值是 None。

## Day 2 — Document Loading & Parsing

### 1. 今日目标

实现 TXT、Markdown 和 PDF 加载，将不同格式转换为统一的 `Document`，保留正文、来源和 PDF 页码，为后续 Chunking 做准备。

### 2. 文档加载在 RAG 中的位置

索引阶段的基本流程：

```text
文档 → 加载与解析 → 清洗 → 切分 → 建立检索表示 → 保存索引
```

加载器负责提取文本与元数据，切分器负责将文本拆成适合检索的片段。今天返回的 PDF 页面还不是最终的检索 Chunk。

### 3. 统一文档结构

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

### 4. TXT 与 Markdown

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

### 5. 基础文本清洗

```python
def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    return text.strip()
```

先处理 `\r\n`，再处理单独的 `\r`，避免把一个换行错误地变成两个。

Python 字符串不可变，`replace()` 和 `strip()` 返回新字符串，需要保存返回值。

当前清洗规则只统一换行、去掉全文首尾空白，不删除正文内部的空格和换行。更复杂的清洗需要结合文档结构设计，避免破坏内容。

### 6. PDF 按页解析

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

### 7. 错误处理

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

### 8. Tokenization 与 Context Window

Tokenization 将文本转换成模型使用的 token 序列。Token 不一定等于一个单词，不能用 `split()` 精确计算模型 token 数。

上下文预算需要考虑：

```text
回答规则 + 对话历史 + 当前问题 + 检索证据 + 输出预留
```

因此不能把整个知识库直接交给模型，需要通过切分和检索选择有限证据。

### 9. 编程中遇到的问题

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

### 10. 实验记录

- TXT：中英文读取正确，内部段落换行保留。
- 空白 TXT：抛出 `ValueError`。
- 不存在的文件：抛出 `FileNotFoundError`。
- 清洗函数：`\r\n`、`\r` 正确归一化，纯空白文本变为 `""`。
- Markdown：标题、列表、代码围栏和代码缩进保留。
- 论文 PDF：返回 15 个非空页面对象，来源、页码和格式字段正确；已对照原文确认内容。
- 空白 PDF、损坏 PDF：已自行确认完成，异常输出未在聊天中逐项核对。

### 11. 当前状态与下一步

TXT、Markdown 和普通文本 PDF 的加载功能已完成。依赖版本记录和 Git 提交尚待确认。

下一步学习 Chunking：比较片段大小、重叠和切分边界对检索及上下文完整性的影响。

## Day 3 — Chunking Strategies

### 1. Chunking 的作用

将加载后的文档切成较小片段，便于检索相关证据、控制上下文长度和定位引用。

片段过大可能混入无关内容；片段过小可能丢失条件、指代或完整定义。切分大小需要通过后续检索实验选择。

今天所有长度的单位都是 Python 字符数，不是 token 数。

### 2. 固定长度切分

每次截取最多 `chunk_size` 个字符：

```python
text[start:start + chunk_size]
```

无重叠时，非空文本的片段数为：

$$
N=\left\lceil\frac{L}{C}\right\rceil
$$

其中 $L$ 为文本字符数，$C$ 为片段大小。

优点是简单、长度可控；缺点是可能切断单词、句子或代码。

### 3. Overlap

相邻片段共享部分内容，步长为：

$$
step=chunk\_size-overlap
$$

参数必须满足：

$$
chunk\_size>0,\qquad 0\le overlap<chunk\_size
$$

当前片段覆盖文本末尾后就停止，避免生成完全包含在上一片段中的多余尾段。

Overlap 可以减少边界附近的信息缺失，但会增加重复存储、Embedding 计算和潜在的上下文重复。

### 4. 递归切分

当前按以下顺序尝试边界：

```text
段落 "\n\n" → 换行 "\n" → 空格 " " → 固定字符切分
```

处理过程：

1. 按当前分隔符拆出单元，保留分隔符。
2. 单元本身超长时，使用剩余分隔符递归处理。
3. 单元不超长时，尝试与暂存内容 `current` 合并。
4. 合并将超长时，保存 `current`，再开始新的片段。
5. 循环结束后保存剩余内容。

`current` 保存正在组合的片段，`chunks` 保存已经确定的片段。

递归切分更倾向于保留自然边界，因为它优先组合完整的段落、行或单词单元。它不理解语义；PDF 排版换行也不一定对应完整句意。

当前递归版本不包含 overlap。

### 5. 正确性检查

无重叠切分验证了：

```python
"".join(chunks) == text
all(len(chunk) <= chunk_size for chunk in chunks)
all(chunk != "" for chunk in chunks)
```

空白字符串如 `" "`、`"\n"` 不是空字符串，仍可能成为片段。是否过滤纯空白片段，需要在后续使用时明确处理。

有 overlap 时，直接拼接片段通常不等于原文，因为部分字符重复保存。

### 6. 片段来源信息

`Chunk` 保存：

- `chunk_id`
- `document_id`
- `text`
- `source`
- `file_type`
- `page_number`
- `chunk_index`

片段继承加载结果中的来源和页码，便于后续引用。

当前 ID 示例：

```text
samples/demo.pdf::page:2::chunk:0
```

`document_id` 当前指向加载器返回的文本单元；PDF 的文本单元是一页。切分配置改变后，片段内容和编号可能变化，增量更新阶段需要进一步管理。

### 7. Python 知识与错误修正

- Python 切片终点超出字符串长度时，会自动截到末尾。
- 比较片段大小要使用 `len(current) + len(part)`。
- `append(x)` 将 `x` 整体加入列表。
- `extend(items)` 将其中元素逐个加入列表。
- 每条递归返回路径都应保持 `list[str]` 类型。
- 递归调用使用更少的分隔符，最终通过短文本或固定切分停止。
- 创建 `Chunk` 后需要加入结果列表；构造参数名必须与数据类字段一致。

### 8. 真实 PDF 实验

输入为同一份论文中加载结果的第二个非空页面对象，文本共 4,315 个字符。

| 策略     | 最大字符数 | 重叠字符数 | 片段数 | 平均长度 | 保存字符总数 |
| -------- | ---------: | ---------: | -----: | -------: | -----------: |
| 固定切分 |        256 |          0 |     17 |    253.8 |        4,315 |
| 固定切分 |        512 |          0 |      9 |    479.4 |        4,315 |
| 固定切分 |      1,024 |          0 |      5 |    863.0 |        4,315 |
| 重叠切分 |        512 |         64 |     10 |    489.1 |        4,891 |
| 递归切分 |        512 |          0 |     12 |    359.6 |        4,315 |

观察：

- 固定切分大小增加，片段数减少。
- 重叠切分重复保存了 `9 × 64 = 576` 个字符，约增加 13.3%；这不是实际 token 成本增幅。
- 固定 256 字符切分将 `modeling` 拆成 `mode` 和 `ling`。
- 递归切分的首段停在 `15].\n\n`，保留了段落边界。
- 另一个递归片段停在 `achieved\n`，说明行边界不保证句意完整。
- 同为 512 字符上限，递归切分产生更多、更短的片段。
- 本次只验证数量和结构差异，没有测量检索准确率。

### 9. 当前状态

固定切分、重叠切分、递归切分、来源继承及真实文档结构实验已通过检查。

## Day 4 — Embedding Fundamentals

### 1. 今日目标

理解文本如何转换为固定维度向量，手写 Masked Mean Pooling 和 L2 归一化，再使用 Sentence Transformers 生成真实 Embedding。

当前流程：

```text
文档 → 加载与清洗 → 切分 → Embedding
```

相似度计算与 Top-K 检索安排在 Day 5。

### 2. Embedding 的作用

Embedding 模型将文本映射为向量：

$$
f_\theta(text)=\mathbf{v},\qquad \mathbf{v}\in\mathbb{R}^{D}
$$

其中：

- $f_\theta$：参数为 $\theta$ 的编码模型。
- $\mathbf{v}$：文本向量。
- $D$：向量维度。

训练目标使相关文本有机会获得相近的向量表示，从而支持语义检索。随机生成数字不能得到有用的语义表示。

单个维度通常没有明确、独立的语义名称。模型效果受语言、领域和训练数据影响，需要实际评测。

### 3. 从 Token 到文本向量

```text
文本
→ Tokenizer
→ Token ID
→ Transformer 上下文向量
→ Pooling
→ 可选归一化
→ 文本向量
```

Token ID 是编号，不能直接用于计算语义距离。

Token 向量表示各个位置的信息；Pooling 将多个位置的向量聚合成一个文本向量。不同模型可能采用不同的 Pooling 策略，今天学习 Mean Pooling。

### 4. Padding 与 Mask

批量文本长度不同，需要补齐为相同长度。Padding 位置不属于正文，不应参与平均。

Mask 示例：

```text
文本 A：[token 1, token 2, PAD]
Mask A：[1,       1,       0]
```

有效位置为 1，Padding 为 0。

带 Mask 的 Mean Pooling：

$$
\mathbf{v}
=
\frac{\sum_{t=1}^{T}m_t\mathbf{h}_t}
{\sum_{t=1}^{T}m_t}
$$

其中：

- $T$：补齐后的序列长度。
- $\mathbf{h}_t$：第 $t$ 个位置的向量。
- $m_t$：对应位置的 Mask。
- 分母：有效位置数量。

例子：

```text
向量：[1, 2]、[3, 4]、[100, 100]
Mask：   1       1          0
```

结果：

```text
([1, 2] + [3, 4]) / 2 = [2, 3]
```

即使 Padding 向量为零，也必须除以有效位置数，而不是补齐后的总位置数。

### 5. Pooling 的 Tensor Shape

| 对象或操作                     | Shape       |
| ------------------------------ | ----------- |
| `token_embeddings`             | `[B, T, D]` |
| `attention_mask`               | `[B, T]`    |
| `attention_mask.unsqueeze(-1)` | `[B, T, 1]` |
| 向量乘以扩展后的 Mask          | `[B, T, D]` |
| 沿 T 求和                      | `[B, D]`    |
| Mask 沿 T 求和                 | `[B, 1]`    |
| 除以有效位置数                 | `[B, D]`    |

- `B`：文本条数。
- `T`：补齐后的 token 位置数。
- `D`：每个 token 的向量维度。

`sum(dim=1)` 沿 T 聚合，默认移除这一维。全零 Mask 没有可计算的平均值，当前实现抛出 `ValueError`。

### 6. L2 归一化

向量长度：

$$
\|\mathbf{v}\|_2=\sqrt{\sum_{j=1}^{D}v_j^2}
$$

归一化：

$$
\hat{\mathbf{v}}
=
\frac{\mathbf{v}}{\max(\|\mathbf{v}\|_2,\varepsilon)}
$$

例如：

```text
[3, 4] 的长度为 5
归一化后为 [0.6, 0.8]
新长度为 1
```

归一化保留非零向量的方向，缩放其长度。是否归一化需要遵循模型和检索度量的要求。

`eps` 是人为指定的很小的正数，当前使用 `1e-12`，用于限制分母、避免除零：

```python
denominator = norms.clamp_min(eps)
```

零向量保持为零，长度仍为 0。长度小于 `eps` 的非零向量也不保证归一化后为单位长度。

### 7. keepdim 与广播

归一化输入为 `[B, D]`。沿 D 计算长度时使用：

```python
sum_squared = embeddings.square().sum(dim=1, keepdim=True)
```

`keepdim=True` 将结果保持为 `[B, 1]`，让每条向量的所有分量除以自己的长度。

如果分母变成 `[B]`，广播会从最后一维对齐，可能报错，也可能在 B 与 D 恰好相等时产生错误结果。

### 8. 编程中修正的问题

- `ndim` 是整数属性，使用 `tensor.ndim`，不加括号。
- `shape` 表示各维大小；检查 B、T 应比较：

  ```python
  token_embeddings.shape[:2] == attention_mask.shape
  ```

- `.sum()` 返回新 Tensor，不会修改原变量，必须接住返回值。
- 多元素布尔 Tensor 不能直接用于 `if`；检查是否存在零值使用：

  ```python
  (valid_counts == 0).any()
  ```

- 模型在函数外加载一次，再传入编码函数，避免重复初始化。
- `normalize_embeddings=True` 已要求归一化，无需再次手动归一化。

### 9. 数学验证结果

Masked Mean Pooling：

```text
Pooled:
[[2, 3],
 [4, 2]]

Shape: [2, 2]
Correct: True
```

全零 Mask 和 B/T 不匹配均正确抛出 `ValueError`。

L2 归一化：

```text
输入：[3, 4]、[0, 0]、[-3, 4]
输出：[0.6, 0.8]、[0, 0]、[-0.6, 0.8]

Shape: [3, 2]
Correct: True
Finite: True
Lengths: [1, 0, 1]
```

### 10. 真实模型编码

模型：

```text
sentence-transformers/all-MiniLM-L6-v2
```

本次配置：

```text
设备：CPU
输出维度：384
最大输入长度：256 tokens
convert_to_tensor：True
normalize_embeddings：True
```

三条英文文本的运行结果：

```text
Shape: [3, 384]
Lengths: [1, 1, 1]
Finite: True
Unit lengths: True
```

每一行对应一条输入文本，行顺序与输入顺序一致。

模型主要面向英文句子和短段落，中文及专业领域效果需要另外评测。向量形状和长度正确，不代表检索质量已经通过验证。

### 11. Token 数与截断实验

原始长度通过关闭截断的 tokenizer 测量，包含特殊 token；实际输入通过 `model.tokenize()` 检查。

| 输入   | 字符数 | 原始 token 数 | 实际输入 token 数 | 截断 |
| ------ | -----: | ------------: | ----------------: | ---- |
| 短文本 |     54 |            13 |                13 | 否   |
| 长文本 |  3,900 |           802 |               256 | 是   |

结论：

- 字符数不等于 token 数。
- 模型输出固定维度，不代表读取了完整输入。
- 超长文本尾部没有参与本次向量生成。
- 后续切分需要结合实际 tokenizer 和模型输入限制。

实验中的超长序列警告来自关闭截断后的长度测量。实际模型预处理将输入截到 256 tokens，本次没有发生索引错误。

### 12. 环境与下载问题

实际使用环境：

```text
Python：3.13 系列
torch==2.14.1
sentence-transformers==6.2.0
transformers==5.21.0
```

首次加载时，访问 Hugging Face 出现 HTTPS 握手超时。排查了终端代理连接后，模型下载、加载和编码成功。

`200 Connection established` 只说明代理接受了隧道请求，还需要继续完成 TLS 握手并收到网站响应。

公开模型的未登录提示没有阻止本次下载。顶部的 macOS 输出重定向提示也没有阻止计算。

### 13. 当前状态与下一步

Day 4 的核心功能与实验验收通过：

- Masked Mean Pooling。
- L2 归一化和零向量处理。
- 真实模型编码。
- 输出形状、有限数值和单位长度检查。
- Token 数与截断实验。

## Day 5 — Similarity Search

### 1. 今日目标

理解并实现点积、余弦相似度、欧氏距离，以及基于 NumPy 的 Top-K 精确向量检索。

完成流程：

```text
PDF 页面 → Chunk → Embedding → 相似度计算 → Top-K → 正文与来源
```

当前实现遍历全部候选向量，属于精确检索，不使用 ANN 索引。

### 2. 点积

两个同维向量的对应分量相乘，再求和：

$$
\mathbf{a}\cdot\mathbf{b}
=\sum_{j=1}^{D}a_jb_j
$$

其中 $D$ 为向量维度。

例如：

$$
[1,2]\cdot[3,4]=1\times3+2\times4=11
$$

点积同时受方向和长度影响。将一个向量扩大两倍，点积也会扩大两倍。

是否使用原始点积，应遵循 Embedding 模型的训练方式和使用要求。

### 3. 余弦相似度

$$
\operatorname{cosine}(\mathbf{a},\mathbf{b})
=
\frac{\mathbf{a}\cdot\mathbf{b}}
{\|\mathbf{a}\|_2\|\mathbf{b}\|_2}
$$

分母消除向量长度的影响，用于比较方向。

从距离平方展开：

$$
\|\mathbf{a}-\mathbf{b}\|_2^2
=
\|\mathbf{a}\|_2^2+\|\mathbf{b}\|_2^2
-2\mathbf{a}\cdot\mathbf{b}
$$

结合余弦定理，可以得到：

$$
\mathbf{a}\cdot\mathbf{b}
=
\|\mathbf{a}\|_2\|\mathbf{b}\|_2\cos\theta
$$

余弦值的几何含义：

- 1：方向相同。
- 0：相互垂直。
- -1：方向相反。

零向量没有方向，余弦相似度未定义。当前实现明确抛出 `ValueError`。

文本向量的余弦分数不是相关概率，也不是回答正确率。阈值需要结合模型与评测数据选择。

### 4. 归一化后的点积

两条非零向量归一化后：

$$
\hat{\mathbf{a}}\cdot\hat{\mathbf{b}}
=
\frac{\mathbf{a}\cdot\mathbf{b}}
{\|\mathbf{a}\|_2\|\mathbf{b}\|_2}
=
\operatorname{cosine}(\mathbf{a},\mathbf{b})
$$

因此问题和文档向量均为单位向量时，点积等于余弦相似度。

### 5. 欧氏距离

$$
d(\mathbf{a},\mathbf{b})
=
\sqrt{\sum_{j=1}^{D}(a_j-b_j)^2}
$$

例如：

$$
d([1,2],[4,6])=\sqrt{9+16}=5
$$

排序方向：

| 方法       | 更靠前的候选 |
| ---------- | ------------ |
| 点积       | 分数更大     |
| 余弦相似度 | 分数更大     |
| 欧氏距离   | 距离更小     |

单位向量满足：

$$
\|\hat{\mathbf{a}}-\hat{\mathbf{b}}\|_2^2
=
2-2\operatorname{cosine}(\mathbf{a},\mathbf{b})
$$

所以在单位向量上，最大余弦与最小欧氏距离的排序一致。原始向量上不一定一致。

### 6. 输入检查

标量比较函数检查：

- 两个输入均为一维数组。
- 形状相同且非空。
- 不包含 NaN 或正负无穷大。
- 计算余弦时，额外拒绝零向量。

需要区分：

```text
维度数量必须大于 0。
向量分量允许为负数或零。

[0, 1] 是合法非零向量。
[0, 0] 是零向量。
```

NumPy 数组的 `.size` 返回元素数量，空数组的 `.size` 为 0，而不是 `None`。

### 7. 批量余弦计算

形状关系：

```text
query：   [D]
vectors： [N, D]
scores：  [N]
```

批量点积：

```python
dots = vectors @ query
```

等价于：

```python
dots = (vectors * query).sum(axis=1)
```

`vectors * query` 只做逐元素乘法，结果仍为 `[N, D]`；必须沿 D 求和，才能得到每个候选的一条分数。

候选长度：

```python
vector_norms = np.sqrt((vectors ** 2).sum(axis=1))
```

余弦分数：

```python
scores = dots / (query_norm * vector_norms)
```

NumPy 使用 `axis` 指定求和维度。这里输出需要为 `[N]`，不保留被求和的维度。

### 8. Top-K 与索引关联

使用：

```python
ranked_indices = np.argsort(-scores, kind="stable")
top_indices = ranked_indices[:top_k]
```

- 对负分数升序排序，实现原分数降序。
- `argsort()` 返回原候选索引，不直接重排分数。
- 稳定排序让同分候选保持输入顺序。
- `top_k <= 0` 返回空列表。
- `top_k` 超过候选数时，返回全部候选。

不能先排序分数，再用 `enumerate()` 当作原候选索引，否则会破坏对应关系。

返回形式：

```python
[(index, score), ...]
```

通过 `chunks[index]` 找回完整 Chunk，保留正文、来源和页码。

### 9. 数学与排序验证

标量计算：

| a        | b         | 点积 |     余弦 |   欧氏距离 |
| -------- | --------- | ---: | -------: | ---------: |
| `[3, 4]` | `[6, 8]`  |   50 |        1 |          5 |
| `[1, 0]` | `[0, 1]`  |    0 |        0 | $\sqrt{2}$ |
| `[1, 0]` | `[-1, 0]` |   -1 |       -1 |          2 |
| `[0, 0]` | `[3, 4]`  |    0 | 抛出异常 |          5 |

运行结果均符合预期。

批量验证结果：

```text
Scores: [0.8, 0.6, -1.0, 1.0]

Top 3:
[(3, 1.0), (0, 0.8), (1, 0.6)]

Top 0:
[]

Top 10:
[(3, 1.0), (0, 0.8), (1, 0.6), (2, -1.0)]

同分候选取前两个：
[(0, 1.0), (1, 1.0)]
```

### 10. 真实文档检索实验

配置：

```text
来源：samples/example.pdf
范围：物理第 2 页
切分：递归切分，无重叠
片段上限：512 个字符
候选数量：12
最大候选长度：128 tokens
模型：sentence-transformers/all-MiniLM-L6-v2
输出归一化：开启
检索方式：精确余弦检索
Top-K：3
```

问题：

```text
Why are recurrent models difficult to parallelize?
```

实际结果：

| 排名 | Chunk 编号 | 余弦分数 | 内容                                           |
| ---: | ---------: | -------: | ---------------------------------------------- |
|    1 |          1 |   0.6976 | 隐藏状态依赖前一状态，顺序计算限制样本内部并行 |
|    2 |          0 |   0.5200 | 循环模型及其应用背景                           |
|    3 |          3 |   0.5029 | Attention 与循环网络的关系                     |

第一名包含回答问题的直接依据，后两名主要提供相关背景。

所有片段均未超过模型的 256 token 输入限制。返回的正文、Chunk ID、来源和页码正确对应。

本次仅验证单页、单个问题，不能据此推断整个系统的检索准确率。

### 11. 编码与来源映射

保持以下顺序关系：

```text
texts[i] → vectors[i] → chunks[i]
```

Tensor 转 NumPy：

```python
tensor.detach().cpu().numpy()
```

编码函数接收字符串列表，因此单个问题也传入 `[query]`。其输出为 `[1, D]`，取第 0 行后得到检索所需的 `[D]`。

避免使用与函数相同的局部变量名，例如：

```python
encode_texts = encode_texts(...)
```

### 12. 修复导入副作用

最初，`ingestion.chunking` 中部分 PDF 实验位于主入口之外，导致检索模块导入它时自动运行 Day 3 实验。

修复结构：

```python
# 类和函数定义放在模块顶层

if __name__ == "__main__":
    # 样例、文件读取和实验输出放在这里
    ...
```

修复后重新运行检索脚本，Day 3 实验不再自动执行，检索结果保持一致。

### 13. 当前限制与下一步

当前每次查询都会重新编码文档片段。后续需要在索引阶段预先编码并保存，查询时复用。

向量评分遍历所有候选，计算量约为 $O(ND)$；当前完整排序约为 $O(N\log N)$。大规模候选下的索引与性能权衡将在后续学习。



## Day 6 — Vector Store

### 1. 今日目标与系统位置

使用 Qdrant 保存片段向量、正文和来源信息，实现持久化、向量检索和来源过滤。

```text
导入：Document → Chunk → Embedding → Point → Qdrant
查询：问题 → 问题向量 → 搜索 Collection → 分数、正文和来源
```

文档向量在导入时生成，查询时只需要编码问题。目前演示主入口仍会重新编码并写入文档；Day 7 将进一步组织导入与查询流程。

### 2. 向量存储的基本概念

- 向量：文本的数值表示。
- 向量索引：组织向量并支持搜索的数据结构。
- 向量数据库：管理向量、记录身份、元数据、持久化及查询。
- Collection：采用统一向量配置的一组记录。
- Point：一条记录，包含 `id`、`vector` 和 `payload`。
- Payload：保存片段正文、Chunk ID、Document ID、来源、页码等信息。

Collection 的 `size` 表示每条向量的维度，不是记录数量。问题和文档必须使用相容的编码模型与配置，仅维度相同还不够。

### 3. 实现的函数

| 函数                       | 职责                                     |
| -------------------------- | ---------------------------------------- |
| `point_id_from_chunk_id()` | 使用 UUID5 将 Chunk ID 映射为确定性 UUID |
| `make_point()`             | 检查向量和片段，并组装 Point             |
| `ensure_collection()`      | 不存在时创建，存在时检查维度和余弦度量   |
| `write_points()`           | 检查批次内部重复 ID，再执行 Upsert       |
| `search_points()`          | 根据问题向量检索，可按来源过滤           |

UUID5 在命名空间和 Chunk ID 相同时生成相同 ID。相同 ID 的 Upsert 会更新已有记录，但完整的文档版本、旧片段清理和增量索引仍待后续实现。

`make_point()` 接收已有向量，不负责生成 Embedding。

### 4. 分数与过滤

相似度取决于问题，因此保存的是片段向量，分数在查询时计算。

对于单位问题向量 `[1, 0]`：

| 候选 | 向量         | 余弦分数 |
| ---- | ------------ | -------: |
| b    | `[1, 0]`     |      1.0 |
| a    | `[0.8, 0.6]` |      0.8 |
| c    | `[0, 1]`     |      0.0 |

来源过滤通过 Payload 的 `source` 字段限制搜索范围。只搜索 `notes/training.md` 时，返回 a、c。

先取全库 Top-K 再删除不符合条件的记录，可能漏掉过滤范围内本应返回的候选。

### 5. 实际验证结果

人工向量实验已验证：

- 首次写入 1 条记录，重复写入相同 ID 后仍为 1 条。
- 按 ID 读取后，ID、Payload 和向量与原记录一致。
- 三候选排序为 b、a、c。
- 来源过滤后返回 a、c。
- 关闭客户端并重新打开后，3 条记录仍可计数和检索。

`make_point()` 的 8 个异常案例均被拒绝：非法维度、二维输入、长度不符、NaN、无穷大、零向量、空白 ID、空白正文。

批次重复 ID、已有 Collection 配置不匹配、查询异常输入等分支已实现，但本轮尚未逐项展示运行验证。

### 6. 真实论文实验

配置：

- 输入：`samples/example.pdf`，物理第 2 页。
- 切分：递归切分，最大 512 字符。
- 模型：`sentence-transformers/all-MiniLM-L6-v2`。
- Collection：`day06_paper_minilm`。
- 度量：Cosine。
- 来源过滤：`samples/example.pdf`。
- Top-K：3。

实际输出：

```text
Real chunks: 12
Max chunk tokens: 128
Vector shape: (12, 384)
Stored points: 12
Count matches: True
```

问题：

```text
Why are recurrent models difficult to parallelize?
```

检索结果：

| 排名 | 片段索引 |   分数 |
| ---- | -------: | -----: |
| 1    |        1 | 0.6976 |
| 2    |        0 | 0.5200 |
| 3    |        3 | 0.5029 |

前三名与 Day 5 的 NumPy 检索结果一致。第一名直接说明隐藏状态之间的顺序依赖限制了同一训练样本内部的并行化。

本次是单页、单问题验证，尚不能说明系统整体检索准确率。余弦分数也不是答案正确的概率。

### 7. Python 与接口易错点

- `vector.ndim` 是维数，`vector.shape[0]` 是一维向量的分量数量。
- `.all()` 需要调用，`.all` 本身是方法对象。
- 字典使用 `{"id": value}`；创建对象使用 `PointStruct(id=value)`。
- `query_points()` 使用 `query` 参数，结果列表为 `.points`。
- `retrieve()` 按已知 ID 读取；`query_points()` 按向量检索。
- 路径类使用 `from pathlib import Path`。
- Tokenizer 字段名是 `input_ids`。
- 使用 `try/finally` 确保客户端关闭。

