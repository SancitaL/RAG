from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from experiments.baseline_retrieval import retrieve


def dense_search(
    query: str,
    model: SentenceTransformer,
    client: QdrantClient,
    collection_name: str,
    top_k: int = 3,
    source: str | None = None,
) -> list[models.ScoredPoint]:
    """Retrieve document chunks using dense vector similarity."""
    # TODO 1: 调用已有的 retrieve，并返回结果
    # 参数逐项传入：
    # query、model、client、collection_name、top_k、source
    rs = retrieve(query,model,client,collection_name,top_k=top_k,source=source)
    return rs



if __name__ == "__main__":
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )

    client = QdrantClient(path="indexes/day07_baseline")

    try:
        # TODO 2: 调用 dense_search
        # collection_name="day07_minilm"
        # query="Why are recurrent models difficult to parallelize?"
        # top_k=3
        results = dense_search(
            collection_name="day07_minilm",
            model=model,
            client=client,
            top_k=1,
            query="Why are recurrent models difficult to parallelize?"
            )

        print("Retrieved:", len(results))

        for rank, point in enumerate(results, start=1):
            payload = point.payload or {}

            print(f"\nRank: {rank}, Score: {point.score:.4f}")
            print("ID:", payload.get("chunk_id"))
            print("Source:", payload.get("source"))
            print("Page:", payload.get("page_number"))
            print("Text:", payload.get("text"))
    finally:
        client.close()