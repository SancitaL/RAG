from dataclasses import asdict, replace
from uuid import NAMESPACE_URL, uuid5
from pathlib import Path
from sentence_transformers import SentenceTransformer
from fundamentals.embedding_basics import encode_texts
from ingestion.chunking import chunk_document
from ingestion.document_loader import load_pdf
import numpy as np
from qdrant_client import models
from qdrant_client import QdrantClient
from ingestion.chunking import Chunk
import json

def point_id_from_chunk_id(chunk_id: str) -> str:
    """Map a readable chunk ID to a deterministic UUID."""
    return str(uuid5(NAMESPACE_URL, chunk_id))

def make_point(
    chunk: Chunk,
    vector: np.ndarray,
    dimension: int,
) -> models.PointStruct:
    """Validate one embedding and attach its text and provenance."""
    # TODO 1: dimension 必须大于 0
    if dimension <= 0:
        raise ValueError("Invalid dimension size")
    # TODO 2: vector 必须是一维，长度必须等于 dimension
    if vector.ndim != 1 or vector.shape[0] != dimension:
        raise ValueError("Invalid vector dimension or size")
    # TODO 3: vector 不能包含 NaN 或无穷大
    if not np.isfinite(vector).all():
        raise ValueError("Invalid data")
    # TODO 4: vector 不能是零向量，因为我们使用 Cosine
    if np.linalg.norm(vector) == 0:
        raise ValueError("Cosine similarity is undefined for zero vectors")
    # TODO 5: chunk_id 和正文不能是空字符串或纯空白
    if not chunk.chunk_id.strip() or not chunk.text.strip():
        raise ValueError("Invalid chunk id or empty text")
    

    # TODO 6: 创建并返回 PointStruct
    return models.PointStruct(
        id = point_id_from_chunk_id(chunk.chunk_id),
        vector = vector.tolist(),
        payload=asdict(chunk)
    )
    # id：调用 point_id_from_chunk_id
    # vector：使用 vector.tolist()
    # payload：使用 asdict(chunk)

def ensure_collection(
    client: QdrantClient,
    name: str,
    dimension: int,
) -> None:
    """Create a cosine collection or validate its existing configuration."""
    # TODO 1: dimension <= 0 时抛出 ValueError
    if dimension <= 0:
        raise ValueError("dimension must > 0")
        
    # TODO 2: 检查 Collection 是否存在
    # 不存在：创建 Collection，然后 return
    # 提示：client.collection_exists(collection_name=name)
    if not client.collection_exists(collection_name=name):
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size = dimension,
                distance=models.Distance.COSINE
            )
        )
        return

    # TODO 3: 已存在时，读取向量配置
    info = client.get_collection(collection_name=name)
    config = info.config.params.vectors

    # 本阶段只处理一个未命名的稠密向量字段
    if not isinstance(config, models.VectorParams):
        raise ValueError("Expected a single unnamed vector configuration")

    # TODO 4: config.size 必须等于 dimension
    if config.size != dimension:
        raise ValueError("size is differ to dimension")
    # TODO 5: config.distance 必须是 models.Distance.COSINE
    if config.distance != models.Distance.COSINE:
        raise ValueError("distance inconsistant")

def write_points(
    client: QdrantClient,
    name: str,
    points: list[models.PointStruct],
) -> None:
    """Write prepared points to an existing collection."""
    # TODO 1: 空列表直接 return
    if not points:
        return
    # TODO 2: 检查本批次的 Point ID 是否重复
    # 提示：提取 ID 后，比较列表长度与 set 的长度
    # 重复时抛出 ValueError，避免一批数据内相互覆盖
    point_ids = []
    for point in points:
        point_ids.append(point.id)

    if len(point_ids) != len(set(point_ids)):
        raise ValueError("repeated ids exist")
        
    # TODO 3: 调用 client.upsert，等待写入完成
    client.upsert(
        collection_name=name,
        points=points,
        wait=True,
    )

def search_points(
    client: QdrantClient,
    name: str,
    query_vector: np.ndarray,
    dimension: int,
    top_k: int = 3,
    source: str | None = None
) -> list[models.ScoredPoint]:
    """Search stored vectors and return scores with their payloads."""
    # TODO 1: dimension 必须大于 0
    if dimension <= 0:
        raise ValueError("Invalid input dimension")

    # TODO 2: query_vector 必须是一维，长度等于 dimension
    if query_vector.ndim != 1 or query_vector.shape[0] != dimension:
        raise ValueError("Invalid size or dimension")

    # TODO 3: 所有分量必须有限，且不能是零向量
    if not np.isfinite(query_vector).all() or np.linalg.norm(query_vector) == 0:
        raise ValueError("invalid vector")

    # TODO 4: top_k <= 0 时返回 []
    if top_k <= 0:
        return []

    query_filter = None
    if source is not None:
        # TODO crate filter conditions
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key = "source",
                    match = models.MatchValue(value=source)
                )
            ]
        )
    # TODO 5: 调用 client.query_points
    results = client.query_points(
        collection_name=name,
        query = query_vector.tolist(),
        limit =top_k,
        query_filter=query_filter,
        with_payload=True
    )


    # TODO 6: 返回 response.points
    return results.points
    
if __name__ == "__main__":
# PDF 第 2 页 → Chunk → 编码一次 → 写入 Qdrant
# 问题 → 问题向量 → 检索 → 正文、分数和出处
    documents = load_pdf(
        Path("data/samples/example.pdf"),
        source = "samples/example.pdf"
    )

    page = next(
        (doc for doc in documents if doc.page_number == 2),
        None
    )

    if page is None:
        raise ValueError("No extractable text on physical page 2")

    chunks = [
        chunk
        for chunk in chunk_document(page, chunk_size=512)
        if chunk.text.strip()
    ]
    if not chunks:
        raise ValueError("No non-empty chunks")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu"
    ) 

    tokenized = model.tokenizer(
        [chunk.text for chunk in chunks],
        truncation = False,
        padding = False
    )

    token_counts = [len(ids) for ids in tokenized["input_ids"]]

    if any(count > model.max_seq_length for count in token_counts):
        raise ValueError("A chunk exceeds the embedding input limit")
    vectors = (
        encode_texts(model, [chunk.text for chunk in chunks])
        .detach().cpu().numpy()
    )
    print("\nReal chunks:", len(chunks))
    print("Max chunk tokens:", max(token_counts))
    print("Vector shape:", vectors.shape)

    # TODO 1: 按相同顺序，将每个 Chunk 和对应向量
    # 传入 make_point，组成 points 列表
    # dimension 使用 vectors.shape[1]
    points = []
    for chunk, vector in zip(chunks, vectors):
        point = make_point(chunk, vector, vectors.shape[1])
        points.append(point)

    real_client = QdrantClient(path="indexes/qdrant_day06")

    try:
        # TODO 2: 使用独立 Collection，名称为 "day06_paper_minilm"
        # 调用 ensure_collection，维度使用 vectors.shape[1]
        collection_name = "day06_paper_minilm"
        ensure_collection(real_client, collection_name, dimension=vectors.shape[1])
        # TODO 3: 调用 write_points，写入真实片段
        write_points(real_client, collection_name, points)

        # TODO 4: 使用 client.count 检查记录数量并打印
        count = real_client.count(
            collection_name=collection_name,
            exact=True
        ).count

        print("Stored points:", count)
        print("Count matches:", count == len(points))


        query = "Why are recurrent models difficult to parallelize?"

        query_vector = (
            encode_texts(model, [query])[0]
            .detach()
            .cpu()
            .numpy()
        )

        hits = search_points(
            real_client,
            collection_name,
            query_vector=query_vector,
            dimension=vectors.shape[1],
            top_k=3,
            source="samples/example.pdf",
        )

        for rank, hit in enumerate(hits, start=1):
            print(f"\nRank: {rank}, Score: {hit.score:.4f}")
            print("ID:", hit.payload["chunk_id"])
            print("Source:", hit.payload["source"])
            print("Page:", hit.payload["page_number"])
            print("Text:", hit.payload["text"])

            
        output = {
            "query": query,
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "collection": collection_name,
            "metric": "cosine",
            "chunk_size_characters": 512,
            "candidate_count": len(chunks),
            "dimension": int(vectors.shape[1]),
            "max_chunk_tokens": max(token_counts),
            "source_filter": "samples/example.pdf",
            "top_k": 3,
            "results": [
                {
                    "rank": rank,
                    "point_id": str(hit.id),
                    "score": float(hit.score),
                    "payload": hit.payload,
                }
                for rank, hit in enumerate(hits, start=1)
            ],
        }

        output_path = Path("results/day06_vector_store.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("\nSaved:", output_path)

    finally:
        real_client.close()

