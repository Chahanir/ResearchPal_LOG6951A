# ResearchPal v1 — Pipeline RAG de base

**LOG6951A — Travail pratique 1 — Hiver 2026**  
Taha-Chahine HAMITOUCHE — 2118576  
Chargé de cours : Simon Barrette

---

## Description

ResearchPal est un assistant de recherche personnel basé sur une architecture RAG (Retrieval-Augmented Generation). Il permet d'ingérer des documents (PDF, Markdown, pages web) et d'y poser des questions en langage naturel. Le système récupère les passages pertinents et génère des réponses citant explicitement leurs sources.

---

## Prérequis

- Python 3.10+
- [Ollama](https://ollama.com) installé et en cours d'exécution en local
- Le modèle `llama3.2` téléchargé dans Ollama

```bash
# Installer Ollama (macOS / Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Télécharger le modèle llama3.2
ollama pull llama3.2
```

---

## Installation

### 1. Cloner / extraire le projet

```bash
unzip LOG6951A_TP1_HAMITOUCHE.zip
cd LOG6951A_TP1_HAMITOUCHE
```

### 2. Créer et activer un environnement virtuel

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

---

## Structure du projet

```
.
├── README.md
├── requirements.txt
├── data/
│   └── sample_docs/           # Corpus de test (PDF + Markdown)
│       ├── capitals_2025_info.pdf
│       └── WASHINGTON_CAPITALS_DATASET.md
├── scripts/
│   ├── ingest.py               # Script d'ingestion du corpus initial
│   ├── test_retrieval.py       # Tests T2 — comparaison cosinus vs MMR
│   ├── test_rag.py             # Tests T3 — pipeline RAG end-to-end
│   └── test_optimization.py    # Tests T4 — comparaison avant/après multi-query
└── src/
    ├── config.py               # Paramètres globaux (LLM, embeddings, chunking)
    ├── llm_factory.py          # Instanciation du LLM et des embeddings
    ├── ingestion/
    │   ├── loader.py           # Chargement PDF / Markdown / URL + segmentation
    │   └── indexer.py          # Génération d'embeddings et indexation ChromaDB
    ├── retrieval/
    │   └── strategies.py       # Stratégies cosinus et MMR
    ├── query_optimization/
    │   └── optimizer.py        # Multi-query retrieval + RRF
    ├── generation/
    │   └── rag_pipeline.py     # Pipeline RAG complet avec historique
    └── interface/
        └── app.py              # Interface Streamlit (T5)
```

---

## Lancement

### Étape 1 — Ingérer le corpus initial

Ce script charge les trois sources du corpus (PDF, Markdown, Wikipedia via URL) et les indexe dans ChromaDB.

```bash
python scripts/ingest.py
```

Pour repartir d'une base vide (utile lors des tests) :

```bash
python scripts/ingest.py --reset
```

### Étape 2 — Lancer l'interface conversationnelle

```bash
streamlit run src/interface/app.py
```

L'interface s'ouvre automatiquement dans le navigateur à l'adresse `http://localhost:8501`.

---

## Utilisation de l'interface

- **Zone de chat** (gauche) : posez vos questions en langage naturel. Chaque réponse affiche un expander *📎 Sources utilisées*.
- **Sidebar** (droite) :
  - Uploader un fichier PDF ou Markdown pour l'indexer
  - Ajouter une page web par URL
  - Changer la stratégie de retrieval (`cosine` ou `mmr`)
  - Activer/désactiver le **Multi-Query** (T4)
  - Effacer la conversation

---

## Scripts de test

Ces scripts reproduisent les évaluations documentées dans le rapport (sections T2, T3, T4). Ils nécessitent que le corpus ait été ingéré au préalable.

```bash
# T2 — Comparaison cosinus vs MMR sur 5 requêtes
python scripts/test_retrieval.py

# T3 — Pipeline RAG end-to-end (3 tours de dialogue)
python scripts/test_rag.py

# T4 — Impact du multi-query (avant/après sur 3 requêtes)
python scripts/test_optimization.py
```

---

## Configuration

Les principaux paramètres sont centralisés dans `src/config.py` :

| Paramètre | Valeur par défaut | Description |
|---|---|---|
| `LLM_MODEL` | `llama3.2` | Modèle Ollama |
| `LLM_BASE_URL` | `http://localhost:11434` | URL du serveur Ollama |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Modèle Sentence Transformers |
| `CHUNK_SIZE` | `500` | Taille des chunks (caractères) |
| `CHUNK_OVERLAP` | `50` | Chevauchement entre chunks |
| `RETRIEVAL_K` | `4` | Nombre de chunks récupérés |
| `MMR_LAMBDA` | `0.5` | Compromis pertinence/diversité MMR |
| `MULTI_QUERY_N` | `3` | Nombre de variantes générées (T4) |

---

## Corpus de test

Le corpus initial porte sur l'équipe de hockey des Washington Capitals :

| Source | Format | Description |
|---|---|---|
| `capitals_2025_info.pdf` | PDF | Guide du camp de développement 2025 (16 pages) |
| `WASHINGTON_CAPITALS_DATASET.md` | Markdown | Statistiques joueurs 2024-25 et 2025-26 |
| Wikipedia — Alexander Ovechkin | URL | Page Wikipedia récupérée au moment de l'ingestion |

---

## Dépendances principales

| Composante | Technologie |
|---|---|
| Orchestration RAG | LangChain 0.3+ |
| Base vectorielle | ChromaDB (local) |
| LLM | llama3.2 via Ollama |
| Embeddings | all-MiniLM-L6-v2 (Sentence Transformers) |
| Interface | Streamlit |
