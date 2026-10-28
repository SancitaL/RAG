import torch
from sentence_transformers import SentenceTransformer

def masked_mean_pool(
    token_embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Average valid token vectors for each text in a batch."""
    # 输入：
    # token_embeddings: [B, T, D]，浮点 Tensor
    # attention_mask: [B, T]，只包含 0/1，与 embeddings 在同一设备

    # TODO 1: 检查 embeddings 是三维、mask 是二维
    # TODO 2: 检查两者的 B、T 一致
    # TODO 3: 将 mask 扩展为 [B, T, 1]，并匹配 embeddings 的 dtype
    # TODO 4: 沿 T 维计算有效 token 数；存在全零 mask 时抛出 ValueError
    # TODO 5: embeddings 乘以 mask，再沿 T 维求和
    # TODO 6: 除以每条文本的有效 token 数，返回 [B, D]

    if token_embeddings.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError(
            "token embedding dim or attention mask dim error"
        )

    if token_embeddings.shape[:2] != attention_mask.shape:
        raise ValueError(
            "dim B or T not same"
        )

    after_mask = attention_mask.unsqueeze(-1).to(
        dtype=token_embeddings.dtype
    )

    valid_counts = after_mask.sum(dim=1)    #valid_counts.shape = [B,1]
    if (valid_counts == 0).any():
        raise ValueError("Each text must have at least one valid token")

    summed = (token_embeddings * after_mask).sum(dim=1) 
    #token_embedding = [B,T,D] 
    # after_mask = [B,T,1]
    # sum = [B, D]
    return summed / valid_counts


def l2_normalize(
    embeddings: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Normalize each row, keeping zero vectors unchanged."""
    # 输入：浮点 Tensor，shape [B, D]

    # TODO 1: 检查 embeddings 是二维，eps > 0
    # TODO 2: 每个分量平方
    # TODO 3: 沿 D 维求和，使用 keepdim=True
    # TODO 4: 开平方，得到每条向量的长度 [B, 1]
    # TODO 5: 将分母限制为至少 eps，避免除零
    # TODO 6: 原向量除以分母，返回 [B, D]
    if embeddings.ndim != 2 or eps <= 0:
        raise ValueError(
            "Invalid input value"
        )

    squared = embeddings.square()
    squared = squared.sum(dim=1,keepdim=True)
    norms = torch.sqrt(squared)
    denominator = norms.clamp_min(eps)
    return embeddings / denominator


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
) -> torch.Tensor:
    """Encode non-empty texts into normalized sentence vectors."""
    # TODO 1: texts 为空列表时抛出 ValueError

    # TODO 2: 如果存在空字符串或纯空白文本，抛出 ValueError
    # 提示：any(not text.strip() for text in texts)

    # TODO 3: 调用 model.encode，返回 PyTorch Tensor，
    #         并要求对输出向量归一化
    if not texts:
        raise ValueError(
            "empty texts"
        )

    if any(not text.strip() for text in texts):
        raise ValueError(
            "empty text"
        )

    embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    return embeddings


if __name__ == "__main__":
    embeddings = torch.tensor([
        [[1., 2.], [3., 4.], [100., 100.]],
        [[2., 0.], [4., 2.], [6., 4.]],
    ])
    mask = torch.tensor([
        [1, 1, 0],
        [1, 1, 1],
    ])

    pooled = masked_mean_pool(embeddings, mask)
    expected = torch.tensor([[2., 3.], [4., 2.]])

    print("Pooled:", pooled)
    print("Shape:", pooled.shape)
    print("Correct:", torch.allclose(pooled, expected))

    # 第一条文本没有任何有效 token
    zero_mask = torch.tensor([
        [0, 0, 0],
        [1, 1, 1],
    ])

    try:
        masked_mean_pool(embeddings, zero_mask)
    except ValueError as error:
        print("Zero mask:", str(error))

    # Mask 的 T=2，与 embeddings 的 T=3 不一致
    wrong_shape_mask = torch.ones((2, 2), dtype=torch.long)

    try:
        masked_mean_pool(embeddings, wrong_shape_mask)
    except ValueError as error:
        print("Shape mismatch:", str(error))

    vectors = torch.tensor([
        [3., 4.],
        [0., 0.],
        [-3., 4.],
    ])

    normalized = l2_normalize(vectors)
    expected = torch.tensor([
        [0.6, 0.8],
        [0.0, 0.0],
        [-0.6, 0.8],
    ])

    print("Normalized:", normalized)
    print("Shape:", normalized.shape)
    print("Correct:", torch.allclose(normalized, expected))
    print("Finite:", torch.isfinite(normalized).all().item())
    print("Lengths:", torch.sqrt(normalized.square().sum(dim=1)))
    print("Loading model...", flush=True)

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )

    print("Model loaded.", flush=True)

    texts = [
        "An optimizer updates model parameters using gradients.",
        "Quantization reduces the memory required to store model weights.",
        "Attention combines information from different token positions.",
    ]

    embeddings = encode_texts(model, texts)
    lengths = embeddings.square().sum(dim=1).sqrt()
    print("Shape:", embeddings.shape)
    print("First 5 values:", embeddings[0, :5])
    print("Lengths:", lengths)
    print("Finite:", torch.isfinite(embeddings).all().item())
    print(
        "Unit lengths:",
        torch.allclose(lengths, torch.ones_like(lengths), atol=1e-6),
    )
    print("Max sequence length:", model.max_seq_length)


    short_text = "An optimizer updates model parameters using gradients."
    long_text = "An optimizer updates model parameters. " * 100

    for name, text in [
        ("short", short_text),
        ("long", long_text),
    ]:
        # 不截断，测量原始 token 数，包含特殊 token
        full_ids = model.tokenizer(
            text,
            truncation=False,
            add_special_tokens=True,
        )["input_ids"]

        # 使用模型实际的预处理流程
        model_inputs = model.tokenize([text])
        used_tokens = model_inputs["attention_mask"].sum().item()

        print(
            name,
            "characters:", len(text),
            "original tokens:", len(full_ids),
            "used tokens:", used_tokens,
            "truncated:", len(full_ids) > used_tokens,
        )