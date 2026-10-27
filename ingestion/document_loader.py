import warnings
import pymupdf
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    source: str
    file_type: str
    page_number: int | None = None

def clean_text(text: str) -> str:
    """Normalize line endings and remove surrounding whitespace."""    
    # 1. replave \r\n by \n
    text = text.replace("\r\n","\n")

    # 2.\r by \n
    text = text.replace("\r","\n")

    # 3.cut off empty at the start and end
    text = text.strip()

    return text


def load_txt(path: Path, source: str) -> list[Document]:
    """Load one UTF-8 TXT file into a single document."""    
    # 1. use UTF-8 read file
    text = path.read_text(encoding="utf-8")

    # 2.cut off empty at the start and end of the passage, keep new line inside
    text = clean_text(text)

    # invalid input
    if not text:
        raise ValueError(f"Empty document:{path}")

    # 3.create document
    doc = Document(
        doc_id = f"{source}::document",
        text = text,
        source = source,
        file_type = "txt",
        page_number = None
    )
    return [doc]


def load_markdown(path: Path, source: str) -> list[Document]:
    """Load one UTF-8 Markdown file while preserving its structure."""
    # TODO: UTF-8 读取
    # TODO: 调用 clean_text
    # TODO: 空正文时抛出包含 path 的 ValueError
    # TODO: 返回一个 Document，file_type 使用 "markdown"
    #       doc_id 仍使用 f"{source}::document"
    
    # 1.load the file
    text = path.read_text(
        encoding="utf-8"
    )

    # 2.clean text
    text = clean_text(text)

    # error
    if not text:
        raise ValueError(f"Empty document in {path}")

    doc = Document(
        doc_id=f"{source}::document",
        text = text,
        source = source,
        file_type = "markdown",
        page_number = None
    )

    return [doc]


def load_pdf(path:Path, source: str) -> list[Document]:
    docs = []
    with pymupdf.open(path) as pdf:
        if pdf.needs_pass:
            raise ValueError(f"Password-protected PDF: {path}")

        for page_number, page in enumerate(pdf, start = 1):
            text = page.get_text("text", sort = True)
            text = clean_text(text)
            if not text:
                warnings.warn(
                    f"No extractable text: {path}, page {page_number}"
                )
            else: 
                doc = Document(
                    doc_id = f"{source}::page:{page_number}",
                    text = text,
                    source = source,
                    file_type = "pdf",
                    page_number = page_number
                )
                docs.append(doc)

    if not docs:
        raise ValueError(
            f"No extractable text in PDF: {path}; OCR may be required."
        )

    return docs


if __name__ == "__main__":
    # documents = load_txt(
    #     Path("data/samples/example.txt"),
    #     source="samples/example.txt",
    # )

    # print("Count:", len(documents))

    # doc = documents[0]
    # print("ID:", doc.doc_id)
    # print("Text:", repr(doc.text))
    # print("Source:", doc.source)
    # print("Type:", doc.file_type)
    # print("Page:", doc.page_number)

    # try:
    #     load_txt(Path("data/samples/empty.txt"), source="samples/empty.txt")
    # except ValueError as error:
    #     print(type(error).__name__, str(error))

    # try:
    #     load_txt(Path("data/samples/missing.txt"), source="samples/missing.txt")
    # except FileNotFoundError as error:
    #     print(type(error).__name__, str(error))

    # raw_text = " \r\n# Training\r\n\r\n    optimizer.step()\r "
    # cleaned = clean_text(raw_text)

    # print("Cleaned:", repr(cleaned))
    # print("Whitespace only:", repr(clean_text(" \r\n\t ")))
    # documents = load_markdown(
    #     Path("data/samples/example.md"),
    #     source="samples/example.md",
    # )

    # doc = documents[0]
    # print("ID:", doc.doc_id)
    # print("Type:", doc.file_type)
    # print("Text:", repr(doc.text))



    documents = load_pdf(
        Path("data/samples/example.pdf"),
        source="samples/example.pdf",
    )

    print("Extracted pages:", len(documents))

    for doc in documents[:3]:
        print("ID:", doc.doc_id)
        print("Page:", doc.page_number)
        print("Type:", doc.file_type)
        print("Preview:", repr(doc.text[:200]))
        print()


