import numpy as np
from pathlib import Path
import json
from sentence_transformers import SentenceTransformer

from fundamentals.embedding_basics import encode_texts
from ingestion.chunking import Chunk, chunk_document
from ingestion.document_loader import load_pdf

def validate_pair(a: np.ndarray, b: np.ndarray) -> None:
    """Validate two non-empty vectors with matching shapes."""
    # TODO 1: 两者必须都是一维
    # TODO 2: shape 必须相同
    # TODO 3: 不能是空向量
    # TODO 4: 不能包含 NaN 或无穷大
    # 不符合条件时抛出 ValueError
    if a.ndim !=1 or b.ndim != 1:
        raise ValueError("Invalid input")

    if a.shape != b.shape:
        raise ValueError("Shape different")

    if (a.size == 0) or (b.size == 0):
        raise ValueError("Empty input")

    if (not np.isfinite(a).all()) or (not np.isfinite(b).all()):
        raise ValueError("infinite or NaN include")

def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    validate_pair(a, b)
    # TODO: 对应分量相乘，再求和
    mult = a * b
    rs = mult.sum()
    return float(rs)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    validate_pair(a, b)
    # TODO: 分别计算 a、b 的 L2 长度
    # TODO: 任意长度为 0 时抛出 ValueError
    # TODO: 点积除以两个长度的乘积
    squared_a = a ** 2
    squared_b = b ** 2

    l_a = np.sqrt(squared_a.sum())
    l_b = np.sqrt(squared_b.sum())

    if l_a == 0 or l_b == 0:
        raise ValueError(
            "length error"
        )

    return float(dot_product(a,b) / (l_a * l_b))

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    validate_pair(a, b)
    # TODO: 差值 → 平方 → 求和 → 开平方
    diff = a - b
    squared_diff = diff ** 2
    sum_squared_diff = squared_diff.sum()
    return float(np.sqrt(sum_squared_diff))

def cosine_scores(
    query: np.ndarray,
    vectors: np.ndarray,
) -> np.ndarray:
    """Compute one cosine score per candidate vector."""
    # TODO 1: query 必须是一维，vectors 必须是二维
    # TODO 2: D 必须一致且大于 0，所有输入值必须有限
    # TODO 3: 计算 query 的长度，以及 vectors 每一行的长度
    # TODO 4: query 或任意候选为零向量时抛出 ValueError
    # TODO 5: 批量计算点积，再除以对应长度乘积
    if query.ndim != 1 or vectors.ndim != 2:
        raise ValueError("Invalid input")
    dimension = query.shape[0]

    if dimension == 0 or vectors.shape[1] != dimension:
        raise ValueError("Vector dimensions must match and be non-empty")

    if not np.isfinite(query).all() or not np.isfinite(vectors).all():
        raise ValueError("Inputs must contain only finite values")

    query_norm = np.sqrt((query ** 2).sum())
    vectors_norm = np.sqrt((vectors ** 2).sum(axis = 1))

    if query_norm == 0 or (vectors_norm == 0).any():
        raise ValueError("Cosine similarity is undefined for zero vectors")
    
    dots = vectors @ query
    return dots / (query_norm * vectors_norm)

def search_top_k(
    query: np.ndarray,
    vectors: np.ndarray,
    top_k: int = 3,
) -> list[tuple[int, float]]:
    """Return candidate indices and cosine scores in descending order."""
    if top_k <= 0:
        return []

    scores = cosine_scores(query, vectors)

    # TODO: 得到分数从高到低的索引，同分保持输入顺序
    # TODO: 截取前 top_k 个
    # TODO: 返回 [(候选索引, 分数), ...]
    ranked_indices = np.argsort(-scores, kind="stable")
    top_indices = ranked_indices[:top_k]
    rs = []
    for index in top_indices:
        rs.append((int(index),float(scores[index])))

    return rs

def search_chunks(
    query: str,
    chunks: list[Chunk],
    model: SentenceTransformer,
    top_k: int = 3,
) -> list[tuple[Chunk, float]]:
    """Retrieve chunks while retaining their text and provenance."""
    if not query.strip():
        raise ValueError("Query must not be empty")

    if not chunks or top_k <= 0:
        return []

    texts = []
    # TODO 1: 按顺序取出所有 chunk.text，得到 texts
    for chunk in chunks:
        texts.append(chunk.text)
    # TODO 2: 将 [query] 和 texts 分别传入 encode_texts
    # 问题结果：[1, D]，取第 0 行得到 [D]
    # 文档结果：[N, D]
    document_embeddings = encode_texts(model,texts)
    query_embeddings = encode_texts(model,[query])
    # TODO 3: 将两个 Tensor 转成 NumPy 数组
    query_vector = query_embeddings[0].detach().cpu().numpy()
    vectors = document_embeddings.detach().cpu().numpy()
    # TODO 4: 调用 search_top_k，得到 [(index, score), ...]
    topk = search_top_k(query_vector, vectors,top_k)
    # TODO 5: 通过 chunks[index] 找回对应对象
    # 返回 [(Chunk对象, score), ...]
    rs = []
    for index, scores in topk:
        rs.append((chunks[index], scores))
    return rs

if __name__ == "__main__":
    documents = load_pdf(
        Path("data/samples/example.pdf"),
        source="samples/example.pdf",
    )

    page = next(
        (doc for doc in documents if doc.page_number == 2),
        None,
    )
    if page is None:
        raise ValueError("No extractable text on physical page 2")

    chunks = [
        chunk
        for chunk in chunk_document(page, chunk_size=512)
        if chunk.text.strip()
    ]

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )

    # 编码前检查，避免片段被静默截断
    tokenized = model.tokenizer(
        [chunk.text for chunk in chunks],
        truncation=False,
        padding=False,
    )
    token_counts = [len(ids) for ids in tokenized["input_ids"]]

    print("Chunks:", len(chunks))
    print("Max chunk tokens:", max(token_counts, default=0))

    if any(count > model.max_seq_length for count in token_counts):
        raise ValueError("A chunk exceeds the embedding input limit")

    query = "Why are recurrent models difficult to parallelize?"
    results = search_chunks(query, chunks, model, top_k=3)




    output = {
        "query": query,
        "source": page.source,
        "page_number": page.page_number,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "strategy": "recursive",
        "chunk_size_characters": 512,
        "candidate_count": len(chunks),
        "max_chunk_tokens": max(token_counts, default=0),
        "metric": "cosine",
        "top_k": 3,
        "results": [
            {
                "rank": rank,
                "chunk_id": chunk.chunk_id,
                "score": score,
                "text": chunk.text,
            }
            for rank, (chunk, score) in enumerate(results, start=1)
        ],
    }

    output_path = Path("results/day05_vector_search.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"\nRank: {rank}, Score: {score:.4f}")
        print("ID:", chunk.chunk_id)
        print("Source:", chunk.source)
        print("Page:", chunk.page_number)
        print("Text:", chunk.text)


    # cases = [
    #     ([3., 4.], [6., 8.], 50., 1., 5.),
    #     ([1., 0.], [0., 1.], 0., 0., np.sqrt(2)),
    #     ([1., 0.], [-1., 0.], -1., -1., 2.),
    # ]

    # for values_a, values_b, expected_dot, expected_cos, expected_dist in cases:
    #     a = np.array(values_a)
    #     b = np.array(values_b)

    #     dot = dot_product(a, b)
    #     cosine = cosine_similarity(a, b)
    #     distance = euclidean_distance(a, b)

    #     print("Vectors:", a, b)
    #     print("Dot:", dot, "Correct:", np.isclose(dot, expected_dot))
    #     print("Cosine:", cosine, "Correct:", np.isclose(cosine, expected_cos))
    #     print("Distance:", distance, "Correct:", np.isclose(distance, expected_dist))
    #     print()

    # zero = np.array([0., 0.])
    # other = np.array([3., 4.])

    # print("Zero dot:", dot_product(zero, other))
    # print("Zero distance:", euclidean_distance(zero, other))

    # try:
    #     cosine_similarity(zero, other)
    # except ValueError as error:
    #     print("Zero cosine:", str(error))


    # query = np.array([1., 0.])

    # vectors = np.array([
    #     [0.8, 0.6],
    #     [3.0, 4.0],
    #     [-1., 0.0],
    #     [1.0, 0.0],
    # ])

    # print("Scores:", cosine_scores(query, vectors))
    # print("Top 3:", search_top_k(query, vectors, 3))
    # print("Top 0:", search_top_k(query, vectors, 0))
    # print("Top 10:", search_top_k(query, vectors, 10))

    # # 三个候选方向相同，余弦都为 1，检查同分顺序
    # ties = np.array([
    #     [1., 0.],
    #     [2., 0.],
    #     [3., 0.],
    # ])
    # print("Ties:", search_top_k(query, ties, 2))