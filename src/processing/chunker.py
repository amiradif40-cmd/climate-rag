"""
Découpage des documents en chunks (morceaux) pour le RAG.
Un chunk = un passage de texte de taille contrôlée avec chevauchement.
"""
import os


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    """
    Découpe un texte en chunks avec chevauchement.

    Args:
        text: Texte à découper
        chunk_size: Taille maximale d'un chunk (en caractères)
        overlap: Chevauchement entre chunks consécutifs (en caractères)

    Returns:
        list[str]: Liste des chunks
    """
    chunk_size = chunk_size or CHUNK_SIZE
    overlap = overlap or CHUNK_OVERLAP

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        # Avance avec chevauchement
        start = end - overlap

    return chunks


def chunk_documents(documents, chunk_size: int = None, overlap: int = None):
    """
    Découpe une liste de documents en chunks.

    Args:
        documents: Liste de dict avec "text" et "metadata"
        chunk_size: Taille des chunks
        overlap: Chevauchement

    Returns:
        list[dict]: Chaque chunk avec ses métadonnées enrichies
    """
    chunk_size = chunk_size or CHUNK_SIZE
    overlap = overlap or CHUNK_OVERLAP

    chunks = []

    for doc in documents:
        text_chunks = chunk_text(doc["text"], chunk_size, overlap)

        for i, chunk in enumerate(text_chunks):
            chunks.append({
                "text": chunk,
                "metadata": {
                    **doc["metadata"],
                    "chunk_id": i,
                    "chunk_index": len(chunks)
                }
            })

    return chunks