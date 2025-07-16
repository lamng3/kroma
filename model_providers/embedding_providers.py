import torch
from typing import List
from transformers import AutoTokenizer, BertTokenizerFast, AutoModel
from sentence_transformers import SentenceTransformer, models
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

    
class STEmbeddingProvider(EmbeddingProvider):
    """sentence-transformers"""
    def __init__(self, model_name: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        word_embedding_model = models.Transformer(
            model_name,
            max_seq_length=512,
            model_args={
                "do_lower_case": "uncased" in model_name
            }
        )
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True,
            pooling_mode_cls_token=False,
            pooling_mode_max_tokens=False
        )
        self.model = SentenceTransformer(
            modules=[word_embedding_model, pooling_model],
            device=self.device
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return embeddings.tolist()