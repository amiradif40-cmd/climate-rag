"""
Pipeline RAG complet : Retrieval + Augmentation + Generation.
Orchestre tous les composants pour répondre à une question.
"""
import os

from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.reranker import Reranker
from generation.llm import GroqLLM
from generation.prompt import build_prompt, format_sources


TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "20"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "3"))


class ClimateRAG:
    """
    Pipeline ClimateRAG complet.

    Flux :
    1. Embedding de la question
    2. Retrieval FAISS (top 20)
    3. Reranking CrossEncoder (top 3)
    4. Construction du prompt avec contexte
    5. Génération LLM
    6. Retour de la réponse + sources
    """

    def __init__(
        self,
        embedder: Embedder = None,
        vector_store: VectorStore = None,
        reranker: Reranker = None,
        llm: GroqLLM = None,
        top_k_retrieval: int = None,
        top_k_rerank: int = None
    ):
        self.embedder = embedder or Embedder()
        self.vector_store = vector_store
        self.reranker = reranker or Reranker()
        self.llm = llm or GroqLLM()
        self.top_k_retrieval = top_k_retrieval or TOP_K_RETRIEVAL
        self.top_k_rerank = top_k_rerank or TOP_K_RERANK

    def ask(self, question: str) -> dict:
        print(f"\n❓ Question : {question}")

        # 1. Embedding de la question
        print("🔍 Étape 1 : Embedding de la question...")
        query_embedding = self.embedder.encode([question])[0]

        # 2. Retrieval FAISS
        print(f"🔍 Étape 2 : Retrieval FAISS (top {self.top_k_retrieval})...")
        results = self.vector_store.search(query_embedding, k=self.top_k_retrieval)

        if not results:
            return {
                "answer": "Aucun document trouvé dans l'index. Veuillez d'abord ingérer des documents.",
                "sources": [],
                "retrieval_results": [],
                "reranked_results": []
            }

        documents = [r["document"] for r in results]

        for doc, result in zip(documents, results):
            doc["metadata"]["faiss_score"] = result["score"]

        # 3. Reranking
        print(f"🔍 Étape 3 : Reranking (top {self.top_k_rerank})...")
        ranked = self.reranker.rerank(question, documents, top_k=self.top_k_rerank)

        # 4. Construction du prompt
        print("📝 Étape 4 : Construction du prompt scientifique...")
        prompt = build_prompt(question, ranked)

        # 5. Génération LLM
        print("🧠 Étape 5 : Génération de la réponse...")
        answer = self.llm.generate(prompt)

        # 6. Formatage des sources
        sources = format_sources(ranked)

        return {
            "answer": answer,
            "sources": sources,
            "retrieval_results": results,
            "reranked_results": ranked
        }