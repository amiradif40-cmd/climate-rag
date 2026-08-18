"""
Reranking avec CrossEncoder multilingue.
Le bi-encoder (embeddings) fait un premier filtre rapide.
Le cross-encoder relit chaque paire [question, chunk] pour un score plus précis.
"""

import math
from sentence_transformers import CrossEncoder


class Reranker:
    """
    Reranker basé sur CrossEncoder multilingue (FR/EN).
    Plus précis que le bi-encoder mais plus lent,
    donc on ne l'utilise que sur les top-k résultats.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-m3"):
        print(f"⏳ Chargement du reranker multilingue {model_name}...")
        self.model = CrossEncoder(model_name)
        print(f"✅ Reranker chargé")

    @staticmethod
    def _sigmoid(x: float) -> float:
        """
        Normalise les logits bruts du CrossEncoder entre 0.0 et 1.0.
        """
        return 1.0 / (1.0 + math.exp(-x))

    @staticmethod
    def _get_document_text(doc: dict) -> str:
        """
        Récupère le texte du document sans planter si la clé varie.
        """
        return (
            doc.get("text")
            or doc.get("content")
            or doc.get("page_content")
            or ""
        )

    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        if not documents:
            return []

        # Construction des paires [question, texte]
        pairs = [[query, self._get_document_text(doc)] for doc in documents]

        # Prédiction des logits bruts
        raw_scores = self.model.predict(pairs)

        # Transformation en score normalisé (0 à 1)
        ranked = []
        for doc, score in zip(documents, raw_scores):
            norm_score = self._sigmoid(float(score))
            ranked.append((doc, norm_score))

        # Tri décroissant par score
        ranked.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc, score in ranked[:top_k]:
            results.append({
                **doc,
                "rerank_score": score,
                "metadata": {
                    **doc.get("metadata", {}),
                    "rerank_score": score
                }
            })

        return results