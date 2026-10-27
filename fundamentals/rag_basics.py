"""A tiny, dependency-free sketch of the RAG query flow.

Day 1 scope: make the control flow visible. This is not a real search engine
and does not call an LLM; later days will replace the TODOs with real systems.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """A small evidence unit with enough provenance to cite its source."""

    doc_id: str
    text: str
    source: str

def retrieve(query: str, documents: list[Document], top_k: int = 2) -> list[Document]:
    """Return up to top_k relevant documents for query.    """
    if top_k <= 0:
        return []

    query_tokens = set(query.lower().split())
    scored_documents = []

    for doc in documents:
        document_tokens = set(doc.text.lower().split())
        score = len(query_tokens & document_tokens)

        if score > 0:
            scored_documents.append((doc,score))

    scored_documents.sort(key=lambda item:item[1], reverse=True)
    return [doc for doc, score in scored_documents[:top_k]]

def build_context(retrieved: list[Document]) -> str:
    """Format retrieved evidence while keeping each source visible."""
    if not retrieved:
        return ""

    blocks = []
    for doc in retrieved:
        block = f"[{doc.doc_id}] Source: {doc.source}\n{doc.text}"
        blocks.append(block)

    return "\n".join(blocks)

def answer_from_evidence(query: str, retrieved: list[Document]) -> str:
    """A deterministic stand-in for an LLM, not natural-language generation."""
    if not retrieved:
        return "Insufficient evidence to answer"

    source_ids = []

    for doc in retrieved:
        source_ids.append(f"[{doc.doc_id}]")

    return "Evidence found in: " + ", ".join(source_ids)

def run_rag(query: str, documents: list[Document], top_k: int = 2) -> dict[str, object]:
    """Run retrieval, context assembly, and evidence-based answer placeholder."""    

    retrieved = retrieve(query,documents,top_k)
    context = build_context(retrieved)
    answer = answer_from_evidence(query,retrieved)
    return {
        "retrieved" : retrieved,
        "context" : context,
        "answer" : answer
    }

if __name__ == "__main__":
    demo_documents = [
        Document("d1", "PyTorch tensors store numerical data.", "notes/tensors.md"),
        Document("d2", "An optimizer updates model parameters from gradients.", "notes/training.md"),
        Document("d3", "Attention lets a model weight information from tokens.", "papers/attention.txt"),
    ]
    queries = [
        ("How does an optimizer update parameters?", 2),
        ("model", 2),
        ("banana", 2),
        ("", 2),
        ("model", 0),
    ]

    # for query, top_k in queries:
    #     retrieved = retrieve(query, demo_documents, top_k)
    #     print(repr(query), top_k, [doc.doc_id for doc in retrieved])

    # retrieved = retrieve("model", demo_documents)
    # print(build_context(retrieved))

    # print("Empty context:", repr(build_context([])))

    for query in [
        "How does an optimizer update parameters?",
        "model",
        "banana",
    ]:
        result = run_rag(query, demo_documents)

        print("Query:", repr(query))
        print("Retrieved:", [doc.doc_id for doc in result["retrieved"]])
        print("Context:", repr(result["context"]))
        print("Answer:", result["answer"])
        print()