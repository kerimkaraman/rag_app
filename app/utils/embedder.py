# app/utils/embedder.py
from sentence_transformers import SentenceTransformer
import numpy as np

class LocalEmbedder():
    """
    Basit yerel embedder (sentence-transformers kullanıyor).
    Neden: Ollama embedding desteği varsa onu da kullanabiliriz; fakat
    sentence-transformers hem offline hem kolayca kurulup çalışır.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]):
        """
        texts -> numpy array of embeddings
        Neden: Milvus'a eklemek için vektörleri numpy olarak almak istiyoruz.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings

    def embed_text(self, text: str):
        return self.embed([text])[0]
