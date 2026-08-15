#!/usr/bin/env python3
"""
Script de test du pipeline complet Phase 1.
Usage: python scripts/ingest.py

Ce script :
1. Charge les PDF du dossier data/raw/ipcc/
2. Nettoie le texte
3. Découpe en chunks
4. Génère les embeddings
5. Construit l'index FAISS
6. Teste une recherche
"""
import sys
import os
from pathlib import Path

# Ajoute src/ au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Charge les variables d'environnement depuis .env
from dotenv import load_dotenv
load_dotenv()

from ingestion.pdf_loader import load_pdfs_from_directory
from processing.cleaner import clean_documents
from processing.chunker import chunk_documents
from embeddings.embedder import Embedder
from retrieval.vector_store import VectorStore


FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/index/climate_index")


def main():
    print("=" * 60)
    print("🌍 ClimateRAG - Pipeline d'ingestion Phase 1")
    print("=" * 60)

    # --- ÉTAPE 1 : Chargement des PDF ---
    print("\n📚 ÉTAPE 1 : Chargement des PDF")
    raw_dir = "data/raw/ipcc"
    documents = load_pdfs_from_directory(raw_dir)

    if not documents:
        print(f"⚠️ Aucun PDF trouvé dans {raw_dir}")
        print("Place tes PDF du GIEC dans ce dossier et relance.")
        return

    # --- ÉTAPE 2 : Nettoyage ---
    print("\n🧹 ÉTAPE 2 : Nettoyage du texte")
    cleaned = clean_documents(documents)
    print(f"✅ {len(cleaned)} pages après nettoyage")

    # --- ÉTAPE 3 : Chunking ---
    print("\n✂️ ÉTAPE 3 : Découpage en chunks")
    chunks = chunk_documents(cleaned)
    print(f"✅ {len(chunks)} chunks créés")
    print(f"   Exemple de chunk : {chunks[0]['text'][:150]}...")

    # --- ÉTAPE 4 : Embeddings ---
    print("\n🧠 ÉTAPE 4 : Génération des embeddings")
    embedder = Embedder()
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts)
    print(f"✅ Embeddings générés : shape = {embeddings.shape}")

    # --- ÉTAPE 5 : Index FAISS ---
    print("\n📦 ÉTAPE 5 : Construction de l'index FAISS")
    store = VectorStore(dimension=embedder.dimension)
    store.add(embeddings, chunks)

    # Sauvegarde
    store.save(FAISS_INDEX_PATH)

    # --- ÉTAPE 6 : Test de recherche ---
    print("\n🔍 ÉTAPE 6 : Test de recherche")
    test_questions = [
        "Pourquoi les canicules deviennent-elles plus fréquentes ?",
        "Quel est le rôle du changement climatique dans les vagues de chaleur ?",
        "Quels sont les impacts des canicules sur la santé ?",
    ]

    for question in test_questions:
        print(f"\n❓ Question : {question}")
        query_embedding = embedder.encode([question])[0]
        results = store.search(query_embedding, k=3)

        for i, result in enumerate(results, 1):
            doc = result["document"]
            print(f"   [{i}] Score: {result['score']:.3f} | Source: {doc['metadata']['source']} p.{doc['metadata']['page']}")
            print(f"       {doc['text'][:200]}...")

    print("\n" + "=" * 60)
    print("🎉 Pipeline Phase 1 terminé avec succès !")
    print("=" * 60)


if __name__ == "__main__":
    main()