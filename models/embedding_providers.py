import torch
from typing import List
from transformers import AutoTokenizer, AutoModel
from .providers import EmbeddingProvider

class HFEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    def embed(self, texts: List[str]) -> List[List[float]]:
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            last_hidden = self.model(**enc).last_hidden_state
        # mean pooling on the [CLS] token
        embeddings = last_hidden[:, 0, :].cpu().numpy()
        return embeddings.tolist()