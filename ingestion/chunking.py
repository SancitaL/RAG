from dataclasses import dataclass
from pathlib import Path
from ingestion.document_loader import Document


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    source: str
    file_type: str
    page_number: int | None
    chunk_index: int


def chunk_document(
    document: Document,
    chunk_size: int,
) -> list[Chunk]:
    """Split one document and retain its provenance."""
    texts = split_recursive(document.text, chunk_size)
    chunks = []

    for index, text in enumerate(texts):
        # TODO: 创建 Chunk 并加入 chunks
        # chunk_id: f"{document.doc_id}::chunk:{index}"
        # document_id: document.doc_id
        # text: 当前 text
        # source、file_type、page_number: 从 document 继承
        # chunk_index: index
        new_chunk = Chunk(
            chunk_id = f"{document.doc_id}::chunk:{index}",
            document_id = document.doc_id,
            text = text,
            source = document.source,
            file_type = document.file_type,
            page_number = document.page_number,
            chunk_index=index
        )

        chunks.append(new_chunk)

    return chunks


def split_fixed(text: str, chunk_size: int) -> list[str]:
    """Split text into non-overlapping character-based chunks."""
    # TODO 1: chunk_size <= 0 时抛出 ValueError
    # TODO 2: 按 chunk_size 移动起点
    # TODO 3: 截取并保存每个片段
    # TODO 4: 返回片段列表
    
    if chunk_size <= 0:
        raise ValueError(
            "chunk_size less or equal than 0"
        )
    if len(text) == 0:
        return []
    split_text = []

    for start in range(0, len(text), chunk_size):
        if (start + chunk_size) > len(text):
            split_text.append(text[start:(len(text))])
        else:
            split_text.append(text[start:start+chunk_size])

    return split_text


def split_with_overlap(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split text into character-based chunks with overlap."""
    # TODO 1: 验证 chunk_size > 0
    # TODO 2: 验证 0 <= overlap < chunk_size
    # TODO 3: 计算 step
    # TODO 4: 按 step 遍历起点，保存切片
    # TODO 5: 当前片段已到达文本末尾时，结束循环
    # TODO 6: 返回片段列表
    if chunk_size <= 0:
        raise ValueError(
            "chun size <= 0"
        )

    if (overlap < 0) or (overlap >= chunk_size) :
        raise ValueError(
            "invalid overlap value"
        )

    split_texts = []

    for start in range(0,len(text),chunk_size-overlap):
        split_texts.append(text[start:start+chunk_size])
        if start + chunk_size >= len(text):
            break
    return split_texts


def split_keep_separator(text: str, separator: str) -> list[str]:
    """Split text while attaching each separator to the preceding part."""
    parts = text.split(separator)

    for index in range(len(parts) - 1):
        parts[index] += separator

    return [part for part in parts if part]


def split_recursive(
    text: str,
    chunk_size: int,
    separators: tuple[str, ...] = ("\n\n", "\n", " ", ""),
) -> list[str]:
    """Split by progressively finer boundaries, preserving the original text."""
    # TODO 1: chunk_size <= 0 时抛出 ValueError
    if chunk_size <= 0:
        raise ValueError(
            "Invalid input chun size"
        )


    # TODO 2: 空字符串返回 []
    if not text:
        return []

    # TODO 3: 如果全文长度已经 <= chunk_size，返回 [text]
    if len(text) <= chunk_size:
        return [text]
    # TODO 4: 如果 separators 用完，调用 split_fixed 作为兜底
    if not separators:
        return split_fixed(text,chunk_size)
    # TODO 5: 取第一个 separator，以及剩余的 separators
    separator = separators[0]
    remaining = separators[1:]
    # TODO 6: separator 为 "" 时，调用 split_fixed 作为兜底
    if separator == "":
        return split_fixed(text,chunk_size)
    # TODO 7: 使用 split_keep_separator 得到 parts
    parts = split_keep_separator(text, separator)
    chunks = []
    current = ""

    for part in parts:
        # TODO 8: 如果 part 本身超过 chunk_size：
        #   a. 若 current 非空，先保存并清空
        #   b. 用剩余 separators 递归切分 part
        #   c. 将递归结果逐个加入 chunks
        #
        # 否则：
        #   a. 如果 current + part 超长，先保存 current 并清空
        #   b. 将 part 加入 current
        if len(part) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            smaller_chunks = split_recursive(part, chunk_size, remaining)
            chunks.extend(smaller_chunks)
        else:
            if len(current) + len(part) > chunk_size:
                chunks.append(current)
                current = ""
            current += part

    # TODO 9: 保存最后一个非空 current
    if current:
        chunks.append(current)

    return chunks


if __name__ == "__main__":
    cases = [
        ("AAAA\n\nBBBB\n\nCCCC", 10),
        ("AA BB CC DD", 5),
        ("ABCDEFGHIJK", 4),
        ("ABC", 5),
        ("", 5),
        ("A\n\nB\nC D", 1),
    ]

    for text, size in cases:
        chunks = split_recursive(text, size)

        print("Input:", repr(text), "Size:", size)
        print("Chunks:", chunks)
        print("Preserved:", "".join(chunks) == text)
        print("Within limit:", all(len(c) <= size for c in chunks))
        print("No empty chunks:", all(c != "" for c in chunks))
        print()

    document = Document(
        doc_id="samples/demo.pdf::page:2",
        text="AA BB CC DD",
        source="samples/demo.pdf",
        file_type="pdf",
        page_number=2,
    )

    chunks = chunk_document(document, chunk_size=5)

    for chunk in chunks:
        print(chunk)

    print("Preserved:", "".join(c.text for c in chunks) == document.text)





from ingestion.document_loader import load_pdf

documents = load_pdf(
    Path("data/samples/example.pdf"),
    source="samples/example.pdf",
)

# 使用同一页比较，避免输入不同导致结果不可比
text = documents[1].text

configs = [
    ("fixed", 256, 0),
    ("fixed", 512, 0),
    ("fixed", 1024, 0),
    ("overlap", 512, 64),
    ("recursive", 512, 0),
]

print("Input characters:", len(text))

for strategy, size, overlap in configs:
    if strategy == "fixed":
        parts = split_fixed(text, size)
    elif strategy == "overlap":
        parts = split_with_overlap(text, size, overlap)
    else:
        parts = split_recursive(text, size)

    total_chars = sum(len(part) for part in parts)
    average_length = total_chars / len(parts) if parts else 0

    print(
        strategy,
        "size:", size,
        "overlap:", overlap,
        "chunks:", len(parts),
        "average:", round(average_length, 1),
        "stored characters:", total_chars,
    )

    for index, part in enumerate(parts[:2]):
        print(
            "Chunk:", index,
            "start:", repr(part[:60]),
            "end:", repr(part[-60:]),
        )

    print()