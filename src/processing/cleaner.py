"""
Nettoyage avancé du texte extrait des PDF scientifiques.
Supprime les artefacts de mise en page, numéros de ligne, tableaux brisés, etc.
"""
import re


def clean_text(text: str) -> str:
    """
    Nettoie un texte brut extrait d'un PDF scientifique (IPCC, etc.).
    """
    # Supprime les numéros de ligne (ex: "  12  ", "  345  ")
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # Supprime les numéros de page isolés
    text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE)
    
    # Supprime les headers/footers fréquents dans les rapports IPCC
    text = re.sub(r'\bChapter\s+\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bIPCC\s+AR6\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bClimate\s+Change\s+\d{4}\b', '', text, flags=re.IGNORECASE)
    
    # Supprime les références de type [1.2, 3.4] ou {11.2, 11.3}
    text = re.sub(r'\{[\d\.\s,;]+\}', '', text)
    text = re.sub(r'\[[\d\.\s,;]+\]', '', text)
    
    # Supprime les codes de tableaux (ex: "HOT EXT. ↑ L. ↑ V.")
    text = re.sub(r'[A-Z]{2,4}\s*EXT\.\s*[↑↓]\s*[A-Z]\.\s*[↑↓]\s*[A-Z]\.', '', text)
    
    # Supprime les flèches et symboles isolés
    text = re.sub(r'[↑↓•◦▪▫]', '', text)
    
    # Supprime les lignes trop courtes (souvent des artefacts)
    lines = text.split('\n')
    cleaned_lines = [line for line in lines if len(line.strip()) > 20]
    text = '\n'.join(cleaned_lines)
    
    # Espaces multiples -> simple espace
    text = re.sub(r'\s+', ' ', text)
    
    # Retours à la ligne multiples -> espace
    text = re.sub(r'\n+', ' ', text)
    
    return text.strip()


def clean_documents(documents):
    """
    Nettoie une liste de documents.
    """
    cleaned = []
    for doc in documents:
        text = clean_text(doc["text"])
        if len(text) > 150:  # Augmente le seuil minimum
            cleaned.append({
                "text": text,
                "metadata": doc["metadata"]
            })
    return cleaned