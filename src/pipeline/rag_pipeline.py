"""
Pipeline RAG complet : Retrieval + Augmentation + Generation.
Orchestre tous les composants pour repondre a une question.
"""
import os

from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.reranker import Reranker
from generation.llm import GroqLLM
from generation.prompt import build_prompt, format_sources


TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "30"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "5"))
RERANK_SCORE_THRESHOLD = float(os.getenv("RERANK_SCORE_THRESHOLD", "0.3"))


class ClimateRAG:
    """
    Pipeline ClimateRAG complet.

    Flux :
    1. Embedding de la question
    2. Retrieval FAISS (top 30)
    3. Reranking CrossEncoder (top 5)
    4. Fallback si reranker trop faible
    5. Construction du prompt avec contexte
    6. Generation LLM
    7. Retour de la reponse + sources
    """

    def __init__(
        self,
        embedder=None,
        vector_store=None,
        reranker=None,
        llm=None,
        top_k_retrieval=None,
        top_k_rerank=None
    ):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store
        self.reranker = reranker or Reranker()
        self.llm = llm or GroqLLM()
        self.top_k_retrieval = top_k_retrieval or TOP_K_RETRIEVAL
        self.top_k_rerank = top_k_rerank or TOP_K_RERANK

    def ask(self, question):
        print(f"\nQuestion : {question}")

        # 1. Embedding de la question
        print("Etape 1 : Embedding de la question...")
        query_embedding = self.embedder.encode([question])[0]

        # 2. Retrieval FAISS
        print(f"Etape 2 : Retrieval FAISS (top {self.top_k_retrieval})...")
        results = self.vector_store.search(query_embedding, k=self.top_k_retrieval)

        if not results:
            return {
                "answer": "Aucun document trouve dans l'index. Veuillez d'abord ingerer des documents.",
                "sources": [],
                "retrieval_results": [],
                "reranked_results": []
            }

        documents = [r["document"] for r in results]

        for doc, result in zip(documents, results):
            doc["metadata"]["faiss_score"] = result["score"]

        # 3. Reranking
        print(f"Etape 3 : Reranking (top {self.top_k_rerank})...")
        ranked = self.reranker.rerank(question, documents, top_k=self.top_k_rerank)

        # 4. Fallback : si le meilleur score de rerank est trop bas, prendre plus de chunks FAISS
        if not ranked or (ranked and ranked[0].get("rerank_score", 0) < RERANK_SCORE_THRESHOLD):
            print("Fallback : scores de reranking trop faibles, utilisation des top 10 FAISS...")
            ranked = documents[:10]
            for r in ranked:
                if "rerank_score" not in r.get("metadata", {}):
                    r["rerank_score"] = r.get("metadata", {}).get("faiss_score", 0.5)

        # 5. Construction du prompt
        print("Etape 4 : Construction du prompt scientifique...")
        prompt = build_prompt(question, ranked)

        # 6. Generation LLM
        print("Etape 5 : Generation de la reponse...")
        answer = self.llm.generate(prompt)

        # 7. Formatage des sources
        sources = format_sources(ranked)

        return {
            "answer": answer,
            "sources": sources,
            "retrieval_results": results,
            "reranked_results": ranked
        }