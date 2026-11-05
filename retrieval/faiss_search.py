import faiss
import numpy as np


def build_flat_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """Build an exact inner-product index from unit vectors."""
    # TODO 1: vectors 必须是二维数组 [N, D]
    # N 和 D 都必须大于 0，否则抛出 ValueError
    if vectors.ndim != 2 or any(n <= 0 for n in vectors.shape[0]) or any(d <=0 for d in vectors.shape[1]):
        raise ValueError("Invalid vector")

    # TODO 2: 所有分量必须有限
    if any((n,v) for (n,v))


    # TODO 3: 转成连续的 float32 数组
    # 提示：np.ascontiguousarray(vectors, dtype=np.float32)
    matrix = 

    # TODO 4: 检查转换后仍然全部有限
    # 较大的有限数转换为 float32 后可能溢出
    ...

    # TODO 5: 检查每一行都是单位向量
    # 提示：
    # norms = np.linalg.norm(matrix, axis=1)
    # np.allclose(norms, 1.0, rtol=1e-4, atol=1e-6)
    ...

    # TODO 6: 从 matrix.shape[1] 获取维度并创建索引
    # 提示：faiss.IndexFlatIP(dimension)
    index = ...

    # TODO 7: 调用 index.add(matrix)，返回 index
    ...