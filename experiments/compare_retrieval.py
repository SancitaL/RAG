from pathlib import Path
import json
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from experiments.baseline_retrieval import prepare_chunks
from fundamentals.embedding_basics import encode_texts
from retrieval.bm25_retriever import (
    build_inverted_index,
    bm25_search,
)
from retrieval.dense_retriever import dense_search
from retrieval.vector_store import (
    make_point,
    ensure_collection,
    write_points,
)


if __name__ == "__main__":
    paths = [
        Path("data/samples/example.txt"),
        Path("data/samples/example.md"),
        Path("data/samples/example.pdf"),
    ]

    chunks = prepare_chunks(paths, chunk_size=512)
    texts = [chunk.text for chunk in chunks]

    inverted_index, document_lengths = build_inverted_index(texts)

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )

    # 防止模型静默截断文档
    tokenized = model.tokenizer(
        texts,
        truncation=False,
        padding=False,
    )
    for chunk, ids in zip(chunks, tokenized["input_ids"]):
        if len(ids) > model.max_seq_length:
            raise ValueError(
                f"Chunk {chunk.chunk_id} has {len(ids)} tokens; "
                f"limit is {model.max_seq_length}"
            )

    # TODO 1: 编码 texts，并转换为 NumPy 数组 [N, D]
    vectors = encode_texts(model, texts).detach().cpu().numpy()

    dimension = vectors.shape[1]

    # TODO 2: 将 chunks 与 vectors 按顺序转换为 points
    points = []
    for chunk, vector in zip(chunks, vectors):
        point = make_point(chunk,vector,dimension)
        points.append(point)

    client = QdrantClient(location=":memory:")
    collection_name = "day09_comparison"

    queries = [
        "Why are recurrent models difficult to parallelize?",
        "recurrent models parallelization",
        "What do PyTorch tensors store?",
        "How are model parameters updated?",
    ]

    try:
        ensure_collection(client, collection_name, dimension)
        write_points(client, collection_name, points)

        print("Chunks:", len(chunks))
        print(
            "Stored points:",
            client.count(
                collection_name=collection_name,
                exact=True,
            ).count,
        )
        experiment_rows = []
        for query in queries:
            print(f"\nQuery: {query}")

            # TODO 3: 调用 bm25_search，top_k=3
            sparse_results = bm25_search(
                query, 
                inverted_index,
                document_lengths,
                top_k=3
            )

            # TODO 4: 调用 dense_search，top_k=3
            # 不设置 source，让两种方法搜索全部相同语料
            dense_results = dense_search(
                query, 
                model, 
                client, 
                collection_name,
                top_k=3
            )
            experiment_rows.append({
                "query" : query,
                "bm25" : [
                    {
                        "rank" : rank,
                        "score": float(score),
                        "chunk_id": chunks[doc_index].chunk_id,
                        "source": chunks[doc_index].source,
                        "page_number": chunks[doc_index].page_number,
                        "text": chunks[doc_index].text
                    }
                    for rank, (doc_index, score)
                    in enumerate(sparse_results, start = 1)
                ],
                "dense": [
                    {
                        "rank": rank,
                        "score": float(point.score),
                        "payload": point.payload
                    }
                    for rank, point
                    in enumerate(dense_results, start=1)
                ]
            })

            print("BM25:")
            for rank, (doc_index, score) in enumerate(
                sparse_results, start=1
            ):
                chunk = chunks[doc_index]
                print(
                    f"{rank}. {score:.4f} "
                    f"{chunk.chunk_id}"
                )

            print("Dense:")
            for rank, point in enumerate(dense_results, start=1):
                payload = point.payload or {}
                print(
                    f"{rank}. {point.score:.4f} "
                    f"{payload.get('chunk_id')}"
                )
        output_path = Path(
            "results/day09_retrieval_comparison.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "chunk_count": len(chunks),
            "chunk_size": 512,
            "dimension": dimension,
            "normalize_embeddings": True,
            "distance": "cosine",
            "bm25_k1": 1.2,
            "bm25_b": 0.75,
            "top_k": 3,
            "source_filter": None,
            "experiments": experiment_rows,
        }
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent = 2)+ "\n",
            encoding='utf-8'
        )
        print("Saved:",output_path)
        print("\nDense boundary checks:")

        zero_results = dense_search(
            "What do PyTorch tensors store?",
            model,
            client,
            collection_name,
            top_k=0,
        )
        print("Zero top_k:", zero_results == [])

        absent_results = dense_search(
            "What do PyTorch tensors store?",
            model,
            client,
            collection_name,
            source="data/samples/not_indexed.txt",
        )
        print("Absent source:", absent_results == [])

        filtered_results = dense_search(
            "What do PyTorch tensors store?",
            model,
            client,
            collection_name,
            source="data/samples/example.txt",
        )
        print(
            "Source filter:",
            len(filtered_results) == 1
            and filtered_results[0].payload["source"]
            == "data/samples/example.txt",
        )

        invalid_cases = [
            (
                "Blank query",
                lambda: dense_search(
                    "   ", model, client, collection_name
                ),
            ),
            (
                "Missing collection",
                lambda: dense_search(
                    "test", model, client,
                    "day09_missing_collection",
                ),
            ),
            (
                "Overlong query",
                lambda: dense_search(
                    "hello " * (model.max_seq_length + 1),
                    model,
                    client,
                    collection_name,
                ),
            ),
        ]

        for label, operation in invalid_cases:
            try:
                operation()
            except ValueError as error:
                print(f"{label}: REJECTED ({error})")
            else:
                print(f"{label}: NOT REJECTED")
    finally:
        client.close()