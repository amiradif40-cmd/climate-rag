"""
Stockage et recherche vectorielle avec FAISS.
FAISS = Facebook AI Similarity Search, ultra-rapide pour la recherche par similarité.
"""
import faiss
import numpy as np
import pickle
from pathlib import Path


class VectorStore:
    """
    Index FAISS pour stocker et rechercher des embeddings.
    """

    def __init__(self, dimension: int, index_type: str = "flat"):
        """
        Initialise l'index FAISS.

        Args:
            dimension: Dimension des vecteurs d'embedding
            index_type: Type d'index ("flat" = exact, rapide à construire)
        """
        self.dimension = dimension
        self.documents = []  # Stocke les documents originaux

        if index_type == "flat":
            # IndexFlatIP = Inner Product (équivalent à cosinus si normalisé)
            self.index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError(f"Type d'index non supporté : {index_type}")

    def add(self, embeddings: np.ndarray, documents: list[dict]):
        """
        Ajoute des embeddings et leurs documents à l'index.

        Args:
            embeddings: Matrice (N, D) de vecteurs
            documents: Liste de dict avec "text" et "metadata"
        """
        embeddings = np.asarray(embeddings, dtype="float32")
        self.index.add(embeddings)
        self.documents.extend(documents)
        print(f"✅ {len(documents)} documents ajoutés à l'index (total: {len(self.documents)})")

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[dict]:
        """
        Recherche les k documents les plus similaires.

        Args:
            query_embedding: Vecteur de requête (1D array)
            k: Nombre de résultats à retourner

        Returns:
            list[dict]: Résultats avec score et document
        """
        query_embedding = np.asarray([query_embedding], dtype="float32")
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "score": float(score),
                "document": self.documents[idx]
            })

        return results

    def save(self, path: str):
        """
        Sauvegarde l'index FAISS et les documents sur disque.

        Args:
            path: Chemin de sauvegarde (sans extension)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Sauvegarde l'index FAISS
        faiss.write_index(self.index, str(path) + ".faiss")

        # Sauvegarde les documents avec pickle
        with open(str(path) + ".pkl", "wb") as f:
            pickle.dump(self.documents, f)

        print(f"💾 Index sauvegardé : {path}")

    def load(self, path: str):
        """
        Charge un index FAISS et les documents depuis le disque.

        Args:
            path: Chemin de chargement (sans extension)
        """
        path = Path(path)

        self.index = faiss.read_index(str(path) + ".faiss")

        with open(str(path) + ".pkl", "rb") as f:
            self.documents = pickle.load(f)

        print(f"📂 Index chargé : {len(self.documents)} documents")
