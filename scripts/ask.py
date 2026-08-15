#!/usr/bin/env python3
"""
Script de test du pipeline RAG complet (Phase 2) — avec Groq API.
Usage: python scripts/ask.py

Nécessite :
- L'index FAISS déjà construit (python scripts/ingest.py)
- Une clé API Groq (gratuite sur https://console.groq.com)
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Charge les variables d'environnement depuis .env
from dotenv import load_dotenv
load_dotenv()

from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore
from retrieval.reranker import Reranker
from generation.llm import GroqLLM
from pipeline.rag_pipeline import ClimateRAG


FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/index/climate_index")


def main():
    print("=" * 70)
    print("🌍 ClimateRAG — Phase 2 : Pipeline RAG Complet (Groq)")
    print("=" * 70)

    # Vérifie que l'index existe
    index_path = Path(FAISS_INDEX_PATH)
    if not (index_path.with_suffix(".faiss").exists() and index_path.with_suffix(".pkl").exists()):
        print("\n⚠️ Index non trouvé !")
        print(f"Chemin attendu : {FAISS_INDEX_PATH}.faiss / .pkl")
        print("Lance d'abord : python scripts/ingest.py")
        return

    # Vérifie la clé API Groq (depuis les variables d'environnement)
    print("\n🔌 Vérification de Groq...")

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    if not GROQ_API_KEY:
        print("\n❌ Clé API Groq manquante !")
        print("\n1. Va sur https://console.groq.com")
        print("2. Crée une API Key gratuite")
        print("3. Crée un fichier .env à la racine du projet avec :")
        print('   GROQ_API_KEY=gsk_ta_cle_api')
        return

    llm = GroqLLM(api_key=GROQ_API_KEY)

    if not llm.is_available():
        print("\n❌ Clé API Groq invalide !")
        return

    print("✅ Groq est prêt")

    # Charge les composants
    print("\n📦 Chargement des composants...")
    embedder = Embedder()
    vector_store = VectorStore(dimension=embedder.dimension)
    vector_store.load(FAISS_INDEX_PATH)
    reranker = Reranker()

    # Crée le pipeline
    rag = ClimateRAG(
        embedder=embedder,
        vector_store=vector_store,
        reranker=reranker,
        llm=llm
    )

    print("\n" + "=" * 70)
    print("🤖 ClimateRAG est prêt ! Pose ta question.")
    print('Tape "quit" ou "exit" pour quitter.')
    print("=" * 70)

    # Questions de test prédéfinies
    test_questions = [
        "Pourquoi les canicules deviennent-elles plus fréquentes ?",
        "Quel est le rôle du changement climatique dans les vagues de chaleur en Europe ?",
        "Quels sont les impacts des canicules sur la santé humaine ?",
    ]

    print("\n📋 Questions de test disponibles :")
    for i, q in enumerate(test_questions, 1):
        print(f"   [{i}] {q}")
    print("   [0] Pose ta propre question")

    while True:
        print("\n" + "-" * 70)
        choice = input("\nChoix (0-3) ou ta question : ").strip()

        if choice.lower() in ("quit", "exit", "q"):
            print("\n👋 Au revoir !")
            break

        if choice == "1":
            question = test_questions[0]
        elif choice == "2":
            question = test_questions[1]
        elif choice == "3":
            question = test_questions[2]
        elif choice == "0" or choice == "":
            question = input("Ta question : ").strip()
        else:
            question = choice

        if not question:
            continue

        # Lance le RAG
        result = rag.ask(question)

        # Affiche la réponse
        print("\n" + "=" * 70)
        print("📄 RÉPONSE")
        print("=" * 70)
        print(result["answer"])

        # Affiche les sources
        print("\n" + "=" * 70)
        print("📚 SOURCES UTILISÉES")
        print("=" * 70)
        for src in result["sources"]:
            print(f"   [{src['id']}] {src['source']} — Page {src['page']}")
            print(f"       FAISS score: {src['faiss_score']:.3f} | Rerank score: {src['rerank_score']:.3f}")

        # Affiche les scores de retrieval
        print("\n" + "=" * 70)
        print("🔍 SCORES DE RETRIEVAL")
        print("=" * 70)
        for i, r in enumerate(result["retrieval_results"][:5], 1):
            meta = r["document"]["metadata"]
            print(f"   [{i}] {meta['source']} p.{meta['page']} — FAISS: {r['score']:.3f}")


if __name__ == "__main__":
    main()