import numpy as np
from typing import List, Tuple, Any


class VectorStore:
    """
    In‐memory vector store for concept embeddings.
    - .add(ids, embeddings)
    - .query(query_emb, top_k) → List[(id, score)]
    """
    def __init__(self):
        # list of external IDs (e.g. concept codes)
        self.ids: List[Any] = []
        # will become a (N × D) array once data is added
        self._matrix: np.ndarray = np.zeros((0, 0), dtype='float32')

    def add(self, ext_ids: List[Any], embeddings: List[List[float]]) -> None:
        """
        Add a batch of embeddings.
        Args:
          ext_ids: list of same length as embeddings; your concept identifiers
          embeddings: list of float vectors (all same length D)
        """
        embs = np.vstack(embeddings).astype('float32')  # shape = (n, D)
        if self._matrix.size == 0:
            # first batch
            self._matrix = embs
        else:
            # stack on existing
            self._matrix = np.vstack([self._matrix, embs])
        self.ids.extend(ext_ids)

    def query(self, query_emb: List[float], top_k: int = 5) -> List[Tuple[Any, float]]:
        """
        Return top_k nearest neighbors under cosine similarity.
        """
        if self._matrix.size == 0:
            return []
        q = np.array(query_emb, dtype='float32').reshape(1, -1)  # (1, D)
        # normalize
        q_norm = q / np.linalg.norm(q, axis=1, keepdims=True)
        M_norm = self._matrix / np.linalg.norm(self._matrix, axis=1, keepdims=True)
        # cosine similarities
        sims = (M_norm @ q_norm.T).reshape(-1)  # shape = (N,)
        # get top_k indices
        idxs = np.argsort(-sims)[:top_k]
        return [(self.ids[i], float(sims[i])) for i in idxs]

    def get_text(self, ext_id: Any) -> str:
        """
        If you also stored a mapping ext_id → text elsewhere,
        override or extend this method. By default returns str(ext_id).
        """
        return str(ext_id)
