"""
Génération d'embeddings avec modèle multilingue.
"""
import os
from sentence_transformers import SentenceTransformer
import numpy as np


MODEL_EMBEDDING = os.getenv("MODEL_EMBEDDING", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


class Embedder:
    """Wrapper pour générer des embeddings vectoriels."""

    def __init__(self, model_name: str = None):
        model_name = model_name or MODEL_EMBEDDING
        print(f"⏳ Chargement du modèle {model_name}...")
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        print(f"✅ Modèle chargé. Dimension : {self.model.get_embedding_dimension()}")

    def encode(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings

    @property
    def dimension(self) -> int:
        return self.model.get_embedding_dimension()