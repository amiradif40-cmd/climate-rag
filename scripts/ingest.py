#!/usr/bin/env python3
"""
Script d'ingestion des PDF.
Usage: python scripts/ingest.py

Ce script :
1. Charge les PDF du dossier data/raw/ipcc/
2. Nettoie le texte
3. Decoupe en chunks
4. Genere les embeddings
5. Construit ou met a jour l'index FAISS
6. Teste une recherche
"""
import sys
import os
import pickle
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
    print("ClimateRAG - Pipeline d'ingestion")
    print("=" * 60)

    # --- ETAPE 1 : Charger l'index existant si present ---
    print("\nEtape 0 : Verification de l'index existant")
    index_path = Path(FAISS_INDEX_PATH)
    existing_chunks = []
    
    if index_path.with_suffix(".faiss").exists() and index_path.with_suffix(".pkl").exists():
        print("Index existant trouve, chargement...")
        try:
            with open(index_path.with_suffix(".pkl"), "rb") as f:
                existing_chunks = pickle.load(f)
            print(f"   {len(existing_chunks)} chunks deja indexes")
        except Exception as e:
            print(f"   Erreur de chargement : {e}")
            existing_chunks = []
    else:
        print("Aucun index existant, creation d'un nouvel index")

    # --- ETAPE 2 : Chargement des PDF ---
    print("\nEtape 1 : Chargement des PDF")
    raw_dir = "data/raw/ipcc"
    documents = load_pdfs_from_directory(raw_dir)

    if not documents:
        print(f"Aucun PDF trouve dans {raw_dir}")
        print("Place tes PDF du GIEC dans ce dossier et relance.")
        return

    # --- ETAPE 3 : Filtrer les PDF deja traites ---
    processed_sources = set()
    for chunk in existing_chunks:
        processed_sources.add(chunk["metadata"].get("source", ""))

    new_documents = []
    for doc in documents:
        source = doc["metadata"].get("source", "")
        if source not in processed_sources:
            new_documents.append(doc)
        else:
            print(f"   Deja traite : {source}")

    if not new_documents:
        print("\nAucun nouveau PDF a traiter. L'index est a jour.")
        return

    print(f"\n{len(new_documents)} nouveaux PDF a traiter")

    # --- ETAPE 4 : Nettoyage ---
    print("\nEtape 2 : Nettoyage du texte")
    cleaned = clean_documents(new_documents)
    print(f"   {len(cleaned)} pages apres nettoyage")

    # --- ETAPE 5 : Chunking ---
    print("\nEtape 3 : Decoupage en chunks")
    new_chunks = chunk_documents(cleaned)
    print(f"   {len(new_chunks)} nouveaux chunks crees")

    # --- ETAPE 6 : Fusion avec l'existant ---
    all_chunks = existing_chunks + new_chunks
    print(f"\nTotal : {len(all_chunks)} chunks ({len(existing_chunks)} existants + {len(new_chunks)} nouveaux)")

    # --- ETAPE 7 : Embeddings ---
    print("\nEtape 4 : Generation des embeddings")
    embedder = Embedder()
    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(texts)
    print(f"   Embeddings generes : shape = {embeddings.shape}")

    # --- ETAPE 8 : Index FAISS ---
    print("\nEtape 5 : Construction de l'index FAISS")
    store = VectorStore(dimension=embedder.dimension)
    store.add(embeddings, all_chunks)

    # Sauvegarde
    store.save(FAISS_INDEX_PATH)
    with open(index_path.with_suffix(".pkl"), "wb") as f:
        pickle.dump(all_chunks, f)

    print(f"\nIndex sauvegarde : {len(all_chunks)} chunks total")

    # --- ETAPE 9 : Test de recherche ---
    print("\nEtape 6 : Test de recherche")
    test_questions = [
        "Pourquoi les canicules deviennent-elles plus frequentes ?",
        "Quel est le role du changement climatique dans les vagues de chaleur ?",
        "Quels sont les impacts des canicules sur la sante ?",
    ]

    for question in test_questions:
        print(f"\nQuestion : {question}")
        query_embedding = embedder.encode([question])[0]
        results = store.search(query_embedding, k=3)

        for i, result in enumerate(results, 1):
            doc = result["document"]
            print(f"   [{i}] Score: {result['score']:.3f} | Source: {doc['metadata']['source']} p.{doc['metadata']['page']}")
            print(f"       {doc['text'][:200]}...")

    print("\n" + "=" * 60)
    print("Pipeline termine avec succes !")
    print("=" * 60)


if __name__ == "__main__":
    main()