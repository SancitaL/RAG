from dataclasses import dataclass
from pathlib import Path

from ingestion.document_loader import Document, load_pdf


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
        new_chunk = Chunk(
            chunk_id=f"{document.doc_id}::chunk:{index}",
            document_id=document.doc_id,
            text=text,
            source=document.source,
            file_type=document.file_type,
            page_number=document.page_number,
            chunk_index=index,
        )
        chunks.append(new_chunk)

    return chunks


def split_fixed(text: str, chunk_size: int) -> list[str]:
    """Split text into non-overlapping character-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if not text:
        return []

    chunks = []

    for start in range(0, len(text), chunk_size):
        chunks.append(text[start:start + chunk_size])

    return chunks


def split_with_overlap(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """Split text into character-based chunks with overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(text), step):
        chunks.append(text[start:start + chunk_size])

        if start + chunk_size >= len(text):
            break

    return chunks


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
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    if not separators:
        return split_fixed(text, chunk_size)

    separator = separators[0]
    remaining = separators[1:]

    if separator == "":
        return split_fixed(text, chunk_size)

    parts = split_keep_separator(text, separator)
    chunks = []
    current = ""

    for part in parts:
        if len(part) > chunk_size:
            if current:
                chunks.append(current)
                current = ""

            smaller_chunks = split_recursive(
                part,
                chunk_size,
                remaining,
            )
            chunks.extend(smaller_chunks)

        else:
            if len(current) + len(part) > chunk_size:
                chunks.append(current)
                current = ""

            current += part

    if current:
        chunks.append(current)

    return chunks


if __name__ == "__main__":
    # 1. 递归切分样例
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

    # 2. Chunk 来源信息验证
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

    print(
        "Preserved:",
        "".join(chunk.text for chunk in chunks) == document.text,
    )
    print()

    # 3. 真实 PDF 的切分策略比较
    documents = load_pdf(
        Path("data/samples/example.pdf"),
        source="samples/example.pdf",
    )

    # 保持与之前实验一致，使用第二个非空页面对象
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