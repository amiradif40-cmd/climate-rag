"""
Reranking avec CrossEncoder.
Le bi-encoder (embeddings) fait un premier filtre rapide.
Le cross-encoder relit chaque paire [question, chunk] pour un score plus précis.
"""
import os
from sentence_transformers import CrossEncoder

# Modele multilingue (FR/EN/...) - remplace l'ancien cross-encoder anglais
# uniquement, necessaire depuis que le corpus melange des documents FR et EN.
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")


class Reranker:
    """
    Reranker basé sur CrossEncoder.
    Plus précis que le bi-encoder mais plus lent,
    donc on ne l'utilise que sur les top-k résultats.
    """

    def __init__(self, model_name: str = None):
        model_name = model_name or RERANKER_MODEL
        print(f"⏳ Chargement du reranker {model_name}...")
        self.model = CrossEncoder(model_name)
        print(f"✅ Reranker chargé")

    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        if not documents:
            return []

        pairs = [[query, doc["text"]] for doc in documents]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True
        )

        results = []
        for doc, score in ranked[:top_k]:
            results.append({
                **doc,
                "rerank_score": float(score),
                "metadata": {
                    **doc.get("metadata", {}),
                    "rerank_score": float(score)
                }
            })

        return results