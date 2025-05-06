import numpy as np
import faiss                    # make sure `faiss-cpu` is installed
from typing import List, Tuple, Any

class VectorStore:
    """
    A simple FAISS‐backed vector store for RAG:
      - call .add(texts, embeddings, ids)
      - call .query(query_embedding, top_k) to get (id, score) pairs
    """

    def __init__(self, dim: int, index_factory: str = "Flat"):
        """
        Args:
          dim: the dimensionality of your embedding vectors
          index_factory: FAISS index type, e.g. "Flat", "IVF100,Flat", "HNSW32"
        """
        # build FAISS index
        self.index = faiss.index_factory(dim, index_factory)
        self.id_to_text = {}
        self.next_id = 0

    def add(self, texts: List[str], embeddings: List[List[float]], ids: List[Any] = None):
        """
        Index a batch of documents.

        Args:
          texts: original text snippets
          embeddings: list of float vectors (length == len(texts))
          ids: optional external identifiers for each text; if None, auto‐assigned
        """
        embeddings_np = np.vstack(embeddings).astype('float32')
        n = embeddings_np.shape[0]

        if ids is None:
            ids = list(range(self.next_id, self.next_id + n))
        ids_np = np.array(ids, dtype='int64')

        # add to FAISS and to mapping
        self.index.add_with_ids(embeddings_np, ids_np)
        for _id, text in zip(ids, texts):
            self.id_to_text[_id] = text

        self.next_id += n

    def query(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Any, float]]:
        """
        Retrieve the top_k nearest neighbors for a query vector.

        Returns:
          List of (id, score) sorted by score descending.
        """
        q = np.array(query_embedding, dtype='float32').reshape(1, -1)
        D, I = self.index.search(q, top_k)
        # FAISS returns distances; for inner product indexes higher is better, for L2 lower is better.
        # Here we assume Flat (L2) so we invert sign for “score.”
        results = [(int(I[0,i]), -float(D[0,i])) for i in range(I.shape[1])]
        return results

    def get_text(self, _id: Any) -> str:
        return self.id_to_text.get(_id, "")
