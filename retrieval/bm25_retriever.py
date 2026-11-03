import re, math, json
from pathlib import Path
from experiments.baseline_retrieval import prepare_chunks
from collections import Counter

def tokenize(text: str) -> list[str]:
    """Extract lowercase English words and numbers."""
    # TODO: 转小写，再提取连续的英文字母或数字
    # 提示：re.findall(r"[a-z0-9]+", ...)
    lower = text.lower()
    repeat = re.findall(r"[a-z0-9]+", lower)
    return repeat

def build_inverted_index(
    texts: list[str],
) -> tuple[dict[str, dict[int, int]], list[int]]:
    """Return term postings and document token lengths."""
    # TODO 1: texts 为空时抛出 ValueError
    if not texts:
        raise ValueError("Invalid Input")

    inverted_index = {}
    document_lengths = []

    for doc_index, text in enumerate(texts):
        # TODO 2: 调用 tokenize
        after_t = tokenize(text)

        # TODO 3: 将 token 数量加入 document_lengths
        document_lengths.append(len(after_t))

        # TODO 4: 使用 Counter 统计当前片段的词频
        term_counts = Counter(after_t)
        
        # TODO 5: 遍历“词、词频”
        # 为每个词保存：
        # inverted_index[term][doc_index] = frequency
        for term, frequency in term_counts.items():
            if term not in inverted_index:
                inverted_index[term] = {}
            inverted_index[term][doc_index] = frequency

    return inverted_index, document_lengths

def bm25_scores(
    query: str,
    inverted_index: dict[str, dict[int, int]],
    document_lengths: list[int],
    k1: float = 1.2,
    b: float = 0.75,
) -> list[float]:
    """Compute one BM25 score per document."""
    # TODO 1: k1 必须大于 0，b 必须在 [0, 1] 内
    # 不满足时抛出 ValueError
    if k1 <= 0:
        raise ValueError("Invalid k1")
    if not 0<=b<=1:
        raise ValueError("Invalid b")

    # TODO 2: 为每个片段准备一个初始分数 0.0
    # 提示：[0.0] * len(document_lengths)
    scores = [0.0] * len(document_lengths)

    # TODO 3: 没有片段，或所有片段长度都是 0 时，
    # 直接返回 scores，避免后面除以 0
    if not document_lengths or sum(document_lengths) == 0:
        return scores

    # TODO 4: 计算片段总数和平均长度
    n_documents = len(document_lengths)
    avgdl = sum(document_lengths) / n_documents

    # TODO 5: 分词并去重，本版本每个查询词只计算一次
    # 提示：dict.fromkeys(tokenize(query))
    for term in dict.fromkeys(tokenize(query)):
        # TODO 6: 取出这个词的 postings
        # 查询词不在索引中时，使用空字典

        postings = inverted_index.get(term, {})

        if not postings:
            continue

        # TODO 7: 计算 df 和 idf
        # math.log(...) 表示自然对数
        df = len(postings)
        idf = math.log(1 + (n_documents - df + 0.5)/(df + 0.5))

        # 只遍历包含当前查询词的片段
        for doc_index, frequency in postings.items():
            # TODO 8: 获取当前片段长度
            document_length = document_lengths[doc_index]

            # TODO 9: 计算长度修正
            length_factor = 1-b+b*(document_length / avgdl)

            # TODO 10: 计算词频权重
            tf_weight = frequency*(k1+1) / (frequency + k1*length_factor)

            # TODO 11: 将当前词的贡献累加到当前片段
            scores[doc_index] += tf_weight * idf

    return scores

def bm25_search(
    query: str,
    inverted_index: dict[str, dict[int, int]],
    document_lengths: list[int],
    top_k: int = 3,
) -> list[tuple[int, float]]:
    """Return matching document indices and scores."""
    # TODO 1: 查询为空字符串或纯空白时，抛出 ValueError
    if not query or not query.strip():
        raise ValueError("Invalid query")

    # TODO 2: top_k <= 0 时返回 []
    if top_k <= 0:
        return []

    # TODO 3: 调用 bm25_scores
    scores = bm25_scores(
        query,
        inverted_index,
        document_lengths
    )

    # TODO 4: 收集分数大于 0 的片段
    # 每一项为 (doc_index, score)
    candidates = []
    for doc_index, score in enumerate(scores):
        if score > 0:
            candidates.append([doc_index,score])

    # TODO 5: 按分数降序排列；同分时按片段编号升序
    # 提示：列表.sort(key=lambda item: (-item[1], item[0]))
    candidates.sort(key=lambda item: (-item[1], item[0]))

    # TODO 6: 返回前 top_k 项
    return candidates[:top_k]

if __name__ == "__main__":
    paths = [
        Path("data/samples/example.txt"),
        Path("data/samples/example.md"),
        Path("data/samples/example.pdf"),
    ]

    # TODO 1: 使用 Day 7 的 prepare_chunks
    # chunk_size=512
    chunks = prepare_chunks(paths,512)

    # TODO 2: 按 chunks 的顺序提取正文
    texts = []
    for chunk in chunks:
        texts.append(chunk.text)

    # TODO 3: 建立倒排索引，得到索引和片段长度
    inverted_index, document_lengths = build_inverted_index(texts)

    print("Chunks:", len(chunks))
    print("Vocabulary:", len(inverted_index))

    queries = [
        "Why are recurrent models difficult to parallelize?",
        "recurrent models parallelization",
        "previous hidden state sequential",
    ]
    experiment_rows = []
    for query in queries:
        results = bm25_search(
            query,
            inverted_index,
            document_lengths,
            top_k=3,
        )
        experiment_rows.append({
            "query": query,
            "result_count": len(results),
            "results": [
                {
                    "rank": rank,
                    "score": score,
                    "chunk_id": chunks[doc_index].chunk_id,
                    "source": chunks[doc_index].source,
                    "page_number": chunks[doc_index].page_number,
                    "text": chunks[doc_index].text,
                }
                for rank, (doc_index, score)
                in enumerate(results, start=1)
            ],
        })

        print(f"\nQuery: {query}")
        print("Retrieved:", len(results))

        for rank, (doc_index, score) in enumerate(results, start=1):
            chunk = chunks[doc_index]

            print(f"Rank: {rank}, Score: {score:.4f}")
            print("ID:", chunk.chunk_id)
            print("Text:", chunk.text)

    print("Collected queries:", len(experiment_rows))
    print("\nBoundary checks:")

    checks = [
        (
            "Unknown term",
            bm25_search(
                "zzzznotinvocabulary",
                inverted_index,
                document_lengths,
            ) == [],
        ),
        (
            "Zero top_k",
            bm25_search(
                "recurrent",
                inverted_index,
                document_lengths,
                top_k=0,
            ) == [],
        ),
        (
            "Empty corpus",
            bm25_scores("optimizer", {}, []) == [],
        ),
        (
            "Zero-length documents",
            bm25_scores("optimizer", {}, [0, 0]) == [0.0, 0.0],
        ),
        (
            "Repeated query term",
            bm25_scores(
                "optimizer optimizer",
                inverted_index,
                document_lengths,
            )
            == bm25_scores(
                "optimizer",
                inverted_index,
                document_lengths,
            ),
        ),
    ]


    output_path = Path("results/day08_bm25.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "chunk_count": len(chunks),
        "vocabulary_size": len(inverted_index),
        "chunk_size": 512,
        "k1": 1.2,
        "b": 0.75,
        "experiments": experiment_rows,
    }

    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Saved:", output_path)

    for label, passed in checks:
        print(f"{label}: {passed}")

    tie_results = bm25_search(
        "optimizer",
        {"optimizer": {0: 1, 1: 1}},
        [1, 1],
    )



    print(
        "Tie ordering:",
        [doc_index for doc_index, _ in tie_results] == [0, 1],
    )
    print(
        "Tie scores:",
        len(tie_results) == 2
        and all(
            math.isclose(
                score,
                math.log(1.2),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
            for _, score in tie_results
        ),
    )

    invalid_cases = [
        (
            "Blank query",
            lambda: bm25_search(
                "   ", inverted_index, document_lengths
            ),
        ),
        (
            "Invalid k1",
            lambda: bm25_scores(
                "optimizer", inverted_index, document_lengths,
                k1=0,
            ),
        ),
        (
            "Invalid b",
            lambda: bm25_scores(
                "optimizer", inverted_index, document_lengths,
                b=1.1,
            ),
        ),
        (
            "Empty index input",
            lambda: build_inverted_index([]),
        ),
    ]

    for label, operation in invalid_cases:
        try:
            operation()
        except ValueError:
            print(f"{label}: REJECTED")
        else:
            print(f"{label}: NOT REJECTED")