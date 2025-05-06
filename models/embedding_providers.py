import torch
from typing import List
from transformers import AutoTokenizer, AutoModel

from .providers import EmbeddingProvider

class HFEmbeddingProvider(EmbeddingProvider):
    def __init__(self, model_name: str):
        self.device   = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name).to(self.device)

    def embed(self, texts: List[str]) -> List[List[float]]:
        # simple mean pooling on the CLS token
        enc = self.tokenizer(texts, padding=True, truncation=True,
                             return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**enc).last_hidden_state[:,0,:].cpu().numpy()
        return out.tolist()