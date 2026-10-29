from pathlib import Path
import json
import argparse
from qdrant_client import models
from ingestion.document_loader import (
    load_txt,
    load_markdown,
    load_pdf,
)
from ingestion.chunking import Chunk, chunk_document
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from fundamentals.embedding_basics import encode_texts
from retrieval.vector_store import (
    make_point,
    ensure_collection,
    write_points,
    search_points,
)

def prepare_chunks(
    paths: list[Path],
    chunk_size: int = 512,
) -> list[Chunk]:
    """Load supported files and produce chunks with provenance."""
    # TODO 1: paths 为空时抛出 ValueError
    if not paths:
        raise ValueError("Invalid path")
    # TODO 2: chunk_size <= 0 时抛出 ValueError
    if chunk_size <= 0:
        raise ValueError("Invalid chunk size")

    loaders = {
        ".txt": load_txt,
        ".md": load_markdown,
        ".pdf": load_pdf,
    }

    chunks = []

    for path in paths:
        # TODO 3: 获取小写后缀
        # 提示：path.suffix.lower()
        suffix = path.suffix.lower()
        
        # TODO 4: 后缀不在 loaders 中时，抛出 ValueError
        # 错误消息中包含 path
        if suffix not in loaders:
            raise ValueError(f"Invalid suffix in {path}")

        # TODO 5: 取出对应加载函数
        # 调用 loader(path, source=path.as_posix())
        # 得到 documents 列表
        loader = loaders[suffix]
        documents = loader(path,source=path.as_posix())

        # TODO 6: 遍历 documents
        # 对每个 Document 调用 chunk_document
        # 将非空白 Chunk 逐个加入 chunks
        for doc in documents:
            doc_chunks = chunk_document(doc,chunk_size)
            for chunk in doc_chunks:
                if chunk.text.strip():
                    chunks.append(chunk)

    # TODO 7: 没有可用 Chunk 时抛出 ValueError
    if not chunks:
        raise ValueError("No valid chunk")

    # TODO 8: 检查 chunk_id 是否重复
    # 重复时抛出 ValueError，避免不同输入相互覆盖
    chunk_ids=[]
    for chunk in chunks:
        chunk_ids.append(chunk.chunk_id)
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("Duplicate chunk IDs")

    return chunks

def build_index(
    paths: list[Path],
    model: SentenceTransformer,
    client: QdrantClient,
    collection_name: str,
    chunk_size: int = 512,
) -> dict[str, int]:
    """Prepare, encode and store document chunks."""
    # TODO 1: 调用 prepare_chunks，得到 chunks
    chunks = prepare_chunks(paths,chunk_size)

    # TODO 2: 按顺序提取正文，得到 texts
    texts = []
    for chunk in chunks:
        texts.append(chunk.text)

    # TODO 3: 使用 tokenizer 测量完整 token 数
    # truncation=False，padding=False
    # 从 input_ids 得到 token_counts
    tokenized = model.tokenizer(
        texts,
        truncation = False,
        padding = False,
    )
    token_counts = []
    for ids in tokenized["input_ids"]:
        token_counts.append(len(ids))

    # TODO 4: 检查每个片段是否超过 model.max_seq_length
    # 超过时抛出 ValueError
    # 消息包含 chunk_id、实际 token 数和模型上限
    for chunk, token_count in zip(chunks, token_counts):
        if token_count > model.max_seq_length:
            raise ValueError(
                f"Chunk {chunk.chunk_id} has {token_count} tokens;"
                f"limit is {model.max_seq_length}"
            )
    # TODO 5: 调用 encode_texts，将结果转换为 NumPy 数组
    # vectors 的形状为 [N, D]
    vectors = encode_texts(
        model,
        texts
    ).detach().cpu().numpy()

    # TODO 6: 使用 vectors.shape[1] 得到 dimension
    # 按顺序将 chunks 和 vectors 转换为 points
    dimension = vectors.shape[1]
    points = []
    for chunk, vector in zip(chunks, vectors):
        point = make_point(chunk, vector, dimension)
        points.append(point)
    
    # TODO 7: 调用 ensure_collection 和 write_points
    ensure_collection(
        client,
        name = collection_name,
        dimension = dimension
    )

    write_points(
        client,
        name = collection_name,
        points = points
    )

    # TODO 8: 返回本次索引摘要
    # {
    #     "chunk_count": len(chunks),
    #     "dimension": dimension,
    #     "max_chunk_tokens": max(token_counts),
    # }
    return {
        "chunk_count" : len(chunks),
        "dimension" : dimension,
        "max_chunk_tokens" : max(token_counts)
    }

def retrieve(
    query: str,
    model: SentenceTransformer,
    client: QdrantClient,
    collection_name: str,
    top_k: int = 3,
    source: str | None = None,
) -> list[models.ScoredPoint]:
    """Encode a query and search the stored document vectors."""
    # TODO 1: 问题为空字符串或纯空白时抛出 ValueError
    if not query.strip():
        raise ValueError("Empty query")
    # TODO 2: top_k <= 0 时返回 []
    if top_k <= 0:
        return []

    # TODO 3: 检查 Collection 是否存在
    # 不存在时抛出 ValueError，提示先建立索引
    # 使用 client.collection_exists(collection_name=collection_name)
    if not client.collection_exists(collection_name=collection_name):
        raise ValueError("collection DNE, create Collection first")

    # TODO 4: 检查问题的完整 token 数，超过模型上限时报错
    # model.tokenizer(
    #     query, truncation=False, padding=False
    # )["input_ids"]
    # 注意：输入单个字符串时，input_ids 是一条 ID 列表
    token_ids = model.tokenizer(
        query,
        truncation = False,
        padding = False
    )["input_ids"]

    if len(token_ids) > model.max_seq_length:
        raise ValueError(
            f"Query has {len(token_ids)} tokens; "
            f"limit is {model.max_seq_length}"
        )

    # TODO 5: 使用 encode_texts(model, [query]) 编码问题
    # 结果为 [1, D]，取第 0 行，再转成 NumPy，得到 [D]
    query_vector = encode_texts(model,[query])[0].detach().cpu().numpy()

    # TODO 6: 调用已有的 search_points 并返回结果
    # 传入 client、collection_name、问题向量、
    # dimension、top_k 和 source
    return search_points(
        client,
        collection_name,
        query_vector,
        dimension = query_vector.shape[0],
        top_k = top_k,
        source = source
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build or query a baseline document index."
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["index", "query"],
    )
    parser.add_argument("--query")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--source")

    args = parser.parse_args()

    if args.mode == "query" and (
        args.query is None or not args.query.strip()
    ):
        parser.error("--query is required in query mode")

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
    client = QdrantClient(path="indexes/day07_baseline")
    collection_name = "day07_minilm"

    try:
        if args.mode == "index":
            # TODO 1: 准备原来的三个文件路径
            # TODO 2: 调用 build_index
            # TODO 3: 打印索引摘要和数据库记录数量
            paths = [
                Path("data/samples/example.txt"),
                Path("data/samples/example.md"),
                Path("data/samples/example.pdf"),
            ]
            summary = build_index(paths,model,client,collection_name=collection_name)
            print(
                f"Index summary: {summary}"
            )
            stored_count = client.count(
                collection_name=collection_name,
                exact=True
            ).count
            print("Stored points: ", stored_count)
            print(
                "Count matches:",
                stored_count == summary["chunk_count"],
            )


        else:
            # TODO 4: 调用 retrieve
            # query=args.query
            # top_k=args.top_k
            # source=args.source
            hits = retrieve(
                query=args.query,
                top_k=args.top_k,
                source=args.source,
                model=model,
                client=client,
                collection_name=collection_name
            )
            # TODO 5: 打印结果数量
            # 遍历结果，打印排名、分数、ID、来源、页码和正文

            print("Retrieved: ",len(hits))
            for rank, hit in enumerate(hits, start = 1):
                print(f"\nRank: {rank}, Score: {hit.score:.4f}")
                print("ID:", hit.payload["chunk_id"])
                print("Source:",hit.payload["source"])
                print("Page:",hit.payload["page_number"])
                print("Text:", hit.payload["text"])

            output = {
                "query": args.query,
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "collection": collection_name,
                "source_filter": args.source,
                "top_k": args.top_k,
                "result_count": len(hits),
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

            output_path = Path("results/day07_queries.jsonl")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with output_path.open("a", encoding="utf-8") as file:
                file.write(
                    json.dumps(output, ensure_ascii=False) + "\n"
                )

            print("Saved:", output_path)

    finally:
        client.close()