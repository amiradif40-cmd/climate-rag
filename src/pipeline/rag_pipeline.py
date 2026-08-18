"""
Pipeline RAG complet : Retrieval Dual (FR + EN) + Augmentation + Generation.
Orchestre tous les composants pour répondre à une question sur un corpus mixte (GIEC + Météo-France).
"""

import os

from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.reranker import Reranker
from generation.llm import GroqLLM
from generation.prompt import build_prompt, format_sources


TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "16"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "10"))
RERANK_SCORE_THRESHOLD = float(
    os.getenv("RERANK_SCORE_THRESHOLD", "0.3")
)


class ClimateRAG:
    """
    Pipeline ClimateRAG complet avec support multilingue hybride.

    Flux :
    1. Traduction de la question (FR -> EN) pour les sources anglophones
    2. Double Retrieval FAISS (recherche FR + recherche EN)
    3. Déduplication des résultats FAISS
    4. Reranking unifié sur la question originale en français
    5. Fallback si le score du reranker est trop faible
    6. Construction du prompt
    7. Génération LLM déterministe (temperature=0.0)
    8. Retour de la réponse et des sources
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

        # Modele leger dedie a la traduction (gpt-oss-20b = remplacant officiel de
        # llama-3.1-8b-instant, deprecie par Groq). Plus rapide et moins cher que
        # d'utiliser le gros modele 120b juste pour traduire une phrase.
        translation_api_key = getattr(self.llm, "api_key", None)
        self.translation_llm = GroqLLM(
            api_key=translation_api_key,
            model="openai/gpt-oss-20b"
        ) if translation_api_key else self.llm

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

    def _translate_to_english(self, question: str) -> str:
        """
        Traduite la question en anglais pour maximiser la recherche dans le GIEC.
        """
        prompt = (
            "Translate the following user question into English for a vector search engine. "
            "Output strictly ONLY the English translation, with no extra text or commentary.\n\n"
            f"Question: {question}"
        )
        try:
            # gpt-oss-20b remplace llama-3.1-8b-instant (deprecie par Groq). C'est aussi
            # un modele de raisonnement : on force un effort bas et peu de tokens, une
            # traduction n'a pas besoin de reflexion approfondie.
            translation = self.translation_llm.generate(
                prompt,
                temperature=0.0,
                max_tokens=200,
                reasoning_effort="low",
            )
            clean_translation = translation.strip().strip('"')
            print(f"Traduction pour retrieval : '{question}' -> '{clean_translation}'")
            return clean_translation
        except Exception as e:
            print(f"Échec de la traduction ({e}). Utilisation de la question originale.")
            return question

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
        Récupère le texte du document sans provoquer d'erreur.
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
        print(f"\nNombre de résultats FAISS (après déduplication) : {len(results)}")

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

    def ask(self, question: str):
        """
        Répond à une question à partir des documents indexés (FR + EN).
        """

        print("\n" + "=" * 80)
        print("Question originale :", question)
        print("=" * 80)

        # 1. Traduction pour les sources anglaises
        search_query_en = self._translate_to_english(question)

        # 2. Embedding des deux requêtes (FR et EN)
        print("\nÉtape 1 : Embedding de la question (FR + EN)...")
        emb_fr = self.embedder.encode([question])[0]
        emb_en = self.embedder.encode([search_query_en])[0]

        # 3. Double Retrieval FAISS
        k_half = max(1, self.top_k_retrieval // 2)
        print(
            f"Étape 2 : Retrieval FAISS dual "
            f"({k_half} FR + {k_half} EN)..."
        )

        results_fr = self.vector_store.search(emb_fr, k=k_half) or []
        results_en = self.vector_store.search(emb_en, k=k_half) or []

        # Fusion et déduplication des paires source/page/texte
        seen_keys = set()
        results = []
        for res in results_fr + results_en:
            doc = res.get("document", {})
            metadata = doc.get("metadata", {})

            unique_key = (
                metadata.get("source", ""),
                metadata.get("page", ""),
                self._get_document_text(doc)[:100]
            )

            if unique_key not in seen_keys:
                seen_keys.add(unique_key)
                results.append(res)

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

        # 4. Reranking unifié sur la question originale en français
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

        # 5. Vérification du score de reranking
        best_score = (
            self._get_rerank_score(ranked[0])
            if ranked
            else 0
        )

        print("\nMeilleur score rerank :", best_score)
        print("Seuil utilisé :", RERANK_SCORE_THRESHOLD)

        # 6. Fallback si le score du reranker est insuffisant
        if not ranked or best_score < RERANK_SCORE_THRESHOLD:
            print(
                "\nFallback : scores de reranking trop faibles."
            )
            print(
                f"Utilisation des {self.top_k_rerank} premiers résultats FAISS."
            )

            ranked = documents[:self.top_k_rerank]

            for document in ranked:
                metadata = document.setdefault("metadata", {})

                if "rerank_score" not in metadata:
                    metadata["rerank_score"] = metadata.get(
                        "faiss_score",
                        0.5
                    )

        # 7. Prompt scientifique en français
        print("\nÉtape 4 : Construction du prompt scientifique...")
        prompt = build_prompt(question, ranked)

        # 8. Génération déterministe (temperature=0.0)
        print("\nÉtape 5 : Génération de la réponse...")
        answer = self.llm.generate(prompt, temperature=0.0)

        # 9. Formattage des sources
        sources = format_sources(ranked)

        print("\nPipeline terminé.")
        print("=" * 80)

        return {
            "answer": answer,
            "sources": sources,
            "retrieval_results": results,
            "reranked_results": ranked
        }