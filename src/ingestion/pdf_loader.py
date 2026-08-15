"""
Extraction de texte depuis des fichiers PDF scientifiques.
Utilise PyMuPDF pour extraire le texte page par page.
"""
from pathlib import Path
import pymupdf


def load_pdf(pdf_path: str):
    """
    Extrait le texte page par page d'un PDF.

    Args:
        pdf_path: Chemin vers le fichier PDF

    Returns:
        list[dict]: Liste de documents avec texte et métadonnées
        Chaque élément contient:
            - text: str, le texte de la page
            - metadata: dict avec source (nom du fichier) et page (numéro)
    """
    documents = []
    pdf_path = Path(pdf_path)

    with pymupdf.open(pdf_path) as doc:
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text("text")
            if text.strip():
                documents.append({
                    "text": text,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_number,
                        "path": str(pdf_path)
                    }
                })

    return documents


def load_pdfs_from_directory(directory: str):
    """
    Charge tous les PDF d'un répertoire.

    Args:
        directory: Chemin du dossier contenant les PDF

    Returns:
        list[dict]: Tous les documents extraits de tous les PDF
    """
    all_documents = []
    directory = Path(directory)

    for pdf_file in directory.glob("*.pdf"):
        print(f"📄 Chargement : {pdf_file.name}")
        docs = load_pdf(str(pdf_file))
        all_documents.extend(docs)

    print(f"✅ Total : {len(all_documents)} pages extraites")
    return all_documents
