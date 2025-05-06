from abc import ABC, abstractmethod
from typing import List

class EmbeddingProvider(ABC):
    """abstract interface for all embedding backends."""
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """encode a list of strings into embeddings."""
        pass
