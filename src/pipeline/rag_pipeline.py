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


TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "30"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "5"))
RERANK_SCORE_THRESHOLD = float(
    os.getenv("RERANK_SCORE_THRESHOLD", "0.3")
)


class ClimateRAG:
    """
    Pipeline ClimateRAG complet.

    Flux :
    1. Embedding de la question
    2. Retrieval FAISS
    3. Reranking
    4. Fallback si le score du reranker est trop faible
    5. Construction du prompt
    6. Génération LLM
    7. Retour de la réponse et des sources
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

        self.top_k_retrieval = (
            top_k_retrieval
            if top_k_retrieval is not None
            else TOP_K_RETRIEVAL
        )

        self.top_k_rerank = (
            top_k_rerank
            if top_k_rerank is not None
            else TOP_K_RERANK
        )

    @staticmethod
    def _get_rerank_score(document):
        """
        Récupère le score de reranking, quel que soit son emplacement.
        """
        return document.get(
            "rerank_score",
            document.get("metadata", {}).get("rerank_score", 0)
        )

    @staticmethod
    def _get_document_text(document):
        """
        Récupère le texte du document sans provoquer d'erreur
        si la clé n'existe pas.
        """
        return (
            document.get("text")
            or document.get("content")
            or document.get("page_content")
            or ""
        )

    @staticmethod
    def _print_faiss_results(results):
        """
        Affiche les résultats FAISS pour le diagnostic.
        """
        print(f"\nNombre de résultats FAISS : {len(results)}")

        for i, result in enumerate(results[:10]):
            document = result.get("document", {})
            metadata = document.get("metadata", {})

            print(f"\n--- Résultat FAISS {i + 1} ---")
            print("Score FAISS :", result.get("score"))
            print("Source :", metadata.get("source"))
            print("Page :", metadata.get("page"))
            print(
                "Texte :",
                ClimateRAG._get_document_text(document)[:500]
            )

    @staticmethod
    def _print_reranked_results(ranked):
        """
        Affiche les résultats après reranking pour le diagnostic.
        """
        print(
            f"\nNombre de résultats après reranking : {len(ranked)}"
        )

        for i, document in enumerate(ranked):
            metadata = document.get("metadata", {})

            print(f"\n--- Résultat reranké {i + 1} ---")
            print(
                "Score rerank :",
                document.get("rerank_score")
            )
            print(
                "Score dans metadata :",
                metadata.get("rerank_score")
            )
            print(
                "Score FAISS :",
                metadata.get("faiss_score")
            )
            print("Source :", metadata.get("source"))
            print("Page :", metadata.get("page"))
            print(
                "Texte :",
                ClimateRAG._get_document_text(document)[:500]
            )

    def ask(self, question):
        """
        Répond à une question à partir des documents indexés.
        """

        print("\n" + "=" * 80)
        print("Question :", question)
        print("=" * 80)

        # 1. Embedding de la question
        print("\nÉtape 1 : Embedding de la question...")
        query_embedding = self.embedder.encode([question])[0]

        # 2. Retrieval FAISS
        print(
            f"Étape 2 : Retrieval FAISS "
            f"(top {self.top_k_retrieval})..."
        )

        results = self.vector_store.search(
            query_embedding,
            k=self.top_k_retrieval
        )

        if not results:
            print("Aucun résultat FAISS trouvé.")

            return {
                "answer": (
                    "Aucun document trouvé dans l'index. "
                    "Veuillez d'abord ingérer des documents."
                ),
                "sources": [],
                "retrieval_results": [],
                "reranked_results": []
            }

        self._print_faiss_results(results)

        documents = [result["document"] for result in results]

        for document, result in zip(documents, results):
            metadata = document.setdefault("metadata", {})
            metadata["faiss_score"] = result.get("score", 0)

        # 3. Reranking
        print(
            f"\nÉtape 3 : Reranking "
            f"(top {self.top_k_rerank})..."
        )

        ranked = self.reranker.rerank(
            question,
            documents,
            top_k=self.top_k_rerank
        )

        ranked = ranked or []

        self._print_reranked_results(ranked)

        # 4. Vérification du score du reranker
        best_score = (
            self._get_rerank_score(ranked[0])
            if ranked
            else 0
        )

        print("\nMeilleur score rerank :", best_score)
        print("Seuil utilisé :", RERANK_SCORE_THRESHOLD)

        # 5. Fallback si le score est trop faible
        if not ranked or best_score < RERANK_SCORE_THRESHOLD:
            print(
                "\nFallback : scores de reranking trop faibles."
            )
            print(
                "Utilisation des 10 premiers résultats FAISS."
            )

            ranked = documents[:10]

            for document in ranked:
                metadata = document.setdefault("metadata", {})

                if "rerank_score" not in metadata:
                    metadata["rerank_score"] = metadata.get(
                        "faiss_score",
                        0.5
                    )

        # 6. Construction du prompt
        print("\nÉtape 4 : Construction du prompt scientifique...")
        prompt = build_prompt(question, ranked)

        # 7. Génération de la réponse
        print("\nÉtape 5 : Génération de la réponse...")
        answer = self.llm.generate(prompt)

        # 8. Formatage des sources
        sources = format_sources(ranked)

        print("\nPipeline terminé.")
        print("=" * 80)

        return {
            "answer": answer,
            "sources": sources,
            "retrieval_results": results,
            "reranked_results": ranked
        }