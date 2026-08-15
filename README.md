# 🌍 ClimateRAG

Assistant scientifique RAG (Retrieval-Augmented Generation) spécialisé sur le réchauffement climatique, les événements extrêmes et les canicules.

## Architecture Phase 1

```
PDF (GIEC, Copernicus...)
    ↓
PyMuPDF — Extraction page par page
    ↓
Cleaner — Nettoyage (headers, footers, espaces)
    ↓
Chunker — Découpage ~1500 caractères, overlap 200
    ↓
BGE-M3 — Embeddings 1024D, multilingue
    ↓
FAISS — Index de similarité vectorielle
    ↓
Recherche — Top-K chunks par cosinus similarity
```

## Stack technique

| Composant | Technologie |
|-----------|-------------|
| Extraction PDF | PyMuPDF |
| Embeddings | BAAI/bge-m3 (1024D, multilingue) |
| Vector DB | FAISS (CPU) |
| Langage | Python 3.10+ |

## Installation rapide

```bash
# 1. Cloner / créer le projet
cd climate-rag

# 2. Environnement virtuel
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# 3. Dépendances
pip install -r requirements.txt
```

## Utilisation Phase 1

### 1. Placer tes documents

Copie tes PDF scientifiques dans :
```
data/raw/ipcc/
```

Exemples de sources recommandées :
- **IPCC AR6 WGI** (Physical Science Basis)
- **IPCC AR6 Synthesis Report**
- **Copernicus Climate Change Service**

> 💡 Commence avec **3 à 5 PDF maximum** pour tester le pipeline.

### 2. Lancer le pipeline

```bash
python scripts/ingest.py
```

Ce script va :
1. 📚 Charger tous les PDF
2. 🧹 Nettoyer le texte
3. ✂️ Découper en chunks
4. 🧠 Générer les embeddings (télécharge BGE-M3 au 1er lancement ~2GB)
5. 📦 Construire l'index FAISS
6. 🔍 Tester 3 questions de recherche

### 3. Résultat attendu

```
❓ Question : Pourquoi les canicules deviennent-elles plus fréquentes ?
   [1] Score: 0.871 | Source: ipcc_ar6_wgi.pdf p.123
       Les vagues de chaleur ont augmenté en fréquence...
```

## Structure du projet

```
climate-rag/
├── data/
│   ├── raw/ipcc/          # PDF sources
│   ├── processed/         # Données nettoyées (futur)
│   └── index/             # Index FAISS sauvegardé
├── src/
│   ├── ingestion/         # Extraction PDF
│   ├── processing/        # Nettoyage + chunking
│   ├── embeddings/        # BGE-M3
│   ├── retrieval/         # FAISS
│   ├── generation/        # LLM (Phase 2)
│   └── pipeline/          # Orchestration (Phase 2)
├── scripts/
│   └── ingest.py          # Pipeline Phase 1
├── requirements.txt
└── README.md
```

## Prochaines étapes (Phase 2)

- [ ] Reranker (CrossEncoder)
- [ ] LLM local via Ollama
- [ ] Prompt scientifique avec citations
- [ ] API FastAPI
- [ ] Interface Streamlit
- [ ] Évaluation du RAG

---

**Projet pédagogique** — Construit étape par étape pour comprendre chaque brique du RAG.
