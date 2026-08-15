"""
Construction du prompt pour le LLM scientifique.
Version allégée pour éviter l'erreur 400 de Groq.
"""


SCIENTIFIC_PROMPT_TEMPLATE = """Tu es un assistant scientifique spécialisé en climatologie.
Tu réponds UNIQUEMENT à partir des extraits fournis ci-dessous.
Si les sources ne permettent pas de répondre, dis-le clairement.
Réponds en français. Cite les sources utilisées.

CONTEXTE :
{context}

QUESTION : {question}

RÉPONSE :"""


def build_prompt(question: str, documents: list[dict]) -> str:
    """
    Construit le prompt avec contexte tronqué.
    """
    context_parts = []
    for i, doc in enumerate(documents, 1):
        meta = doc.get("metadata", {})
        source = meta.get("source", "Document inconnu")
        page = meta.get("page", "N/A")
        # Tronque le texte à 800 caractères pour ne pas dépasser la limite
        text = doc.get("text", "")[:800]
        
        context_parts.append(
            f"[{i}] {source} p.{page}: {text}"
        )

    context = "\n\n".join(context_parts)

    return SCIENTIFIC_PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )


def format_sources(documents: list[dict]) -> list[dict]:
    sources = []
    for i, doc in enumerate(documents, 1):
        meta = doc.get("metadata", {})
        sources.append({
            "id": i,
            "source": meta.get("source", "Inconnu"),
            "page": meta.get("page", "N/A"),
            "chunk_id": meta.get("chunk_id", "N/A"),
            "faiss_score": meta.get("faiss_score", None),
            "rerank_score": doc.get("rerank_score", None)
        })
    return sources