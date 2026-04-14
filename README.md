# ResearchPal v2 — Pipeline RAG Agentique

**LOG6951A — Travail pratique 2 — Hiver 2026**  
Taha-Chahine HAMITOUCHE — 2118576  
Félix PAILLÉ DOWELL - 2256243
Chargé de cours : Simon Barrette

---

## Description

ResearchPal v2 est une extension agentique du TP1. Le pipeline RAG linéaire est transformé en une **state machine LangGraph** capable de détecter et corriger ses propres erreurs de retrieval (Corrective RAG). L'agent choisit dynamiquement entre deux outils (`search_corpus` et `search_web`), dispose d'une mémoire à deux niveaux (court terme SQLite + long terme épisodique), et est instrumenté avec Arize Phoenix pour l'observabilité.

---

## Prérequis

- Python 3.10+
- [Ollama](https://ollama.com) installé et en cours d'exécution en local
- Le modèle `llama3.2` téléchargé dans Ollama

```bash
# Installer Ollama (macOS / Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Windows — via winget (Windows 11)
winget install Ollama.Ollama
# ou télécharger le .exe directement sur https://ollama.com/download

# Télécharger le modèle llama3.2
ollama pull llama3.2
```

---

## Installation

### 1. Cloner / extraire le projet

```bash
unzip ResearchPal_LOG6951A.zip
cd ResearchPal_LOG6951A.zip
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
│   └── sample_docs/                    # Corpus de test
│       ├── capitals_2025_info.pdf
│       └── WASHINGTON_CAPITALS_DATASET.md
├── memory/                             # Persistance mémoire (gitignored)
│   ├── checkpoints.sqlite              # Mémoire court terme (LangGraph)
│   └── episodic_memory.json            # Mémoire long terme (Option B)
├── eval/                               # Résultats d'évaluation
│   ├── ragas_results.json
│   └── llm_judge_results.json
├── traces/                             # Captures Phoenix
├── src/
│   ├── config.py                       # Paramètres globaux
│   ├── llm_factory.py                  # Instanciation LLM et embeddings
│   ├── agent/
│   │   ├── graph.py                    # Graphe LangGraph (T1)
│   │   ├── state.py                    # AgentState TypedDict (T1)
│   │   ├── tools.py                    # Outils @tool : search_corpus, search_web (T2)
│   │   └── memory.py                   # Checkpointer SQLite + mémoire épisodique (T3)
│   ├── ingestion/
│   │   ├── loader.py                   # Chargement PDF / Markdown / URL
│   │   └── indexer.py                  # Embeddings et indexation ChromaDB
│   ├── retrieval/
│   │   └── strategies.py              # Stratégies cosinus et MMR
│   ├── observability/
│   │   └── tracing.py                  # Arize Phoenix + décorateur @instrument_node (T4)
│   ├── evaluation/
│   │   ├── dataset.py                  # Dataset 15 paires (T5)
│   │   ├── ragas_eval.py               # Évaluation faithfulness + answer_relevancy (T5)
│   │   └── llm_judge.py                # LLM-as-judge : citation_quality (T5)
│   └── interface/
│       └── app_v2.py                   # Interface Streamlit TP2
├── scripts/
│   ├── ingest.py                       # Ingestion du corpus initial
│   ├── run_eval.py                     # Évaluation complète (RAGAS + LLM-judge)
│   ├── demo_tool_selection.py          # Démonstration sélection dynamique d'outils
│   └── test_corrections.py             # Tests du cycle correctif
```

---

## Lancement

### Étape 1 — Ingérer le corpus initial

```bash
python scripts/ingest.py
```

Pour repartir d'une base vide :

```bash
python scripts/ingest.py --reset
```

### Étape 2 — Lancer l'interface conversationnelle

```bash
streamlit run src/interface/app_v2.py
```

L'interface s'ouvre à `http://localhost:8501`.

### Étape 3 — Démarrer Phoenix (observabilité)

Une fois l'interface ouverte, cliquer sur **"🚀 Démarrer Phoenix"** dans la sidebar. Phoenix démarre localement et est accessible à `http://localhost:6006`. Aucun compte cloud requis.

---

## Utilisation de l'interface

- **Zone de chat** : posez vos questions. L'agent affiche l'outil utilisé, le nombre de retries, le grade et la latence pour chaque réponse.
- **Sidebar** :
  - Démarrer Phoenix pour l'observabilité
  - Uploader un PDF ou Markdown pour l'indexer à la volée
  - Ajouter une page web par URL
  - Voir les exemples mémorisés (mémoire épisodique long terme)
  - Créer une nouvelle session (nouveau thread_id)

---

## Évaluation qualité (T5)

```bash
# Lancer l'évaluation complète (RAGAS + LLM-as-judge) sur 15 paires
python scripts/run_eval.py
```

Les résultats sont sauvegardés dans `eval/ragas_results.json` et `eval/llm_judge_results.json`.

---

## Démonstration sélection d'outils (T2)

```bash
python scripts/demo_tool_selection.py
```

Démontre sur 3 requêtes contrastées (corpus / hors-corpus / mixte) que l'agent sélectionne dynamiquement le bon outil.

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

---

## Architecture agentique (T1)

Le graphe LangGraph implémente le pattern **Corrective RAG** avec les nœuds suivants :

| Nœud | Rôle |
|---|---|
| `route_question` | LLM avec bind_tools — décide quel outil invoquer |
| `execute_tools` | Exécute search_corpus ou search_web via ToolNode |
| `retrieve` | Récupération ChromaDB (fallback si pas de résultat web) |
| `grade_documents` | Évalue la pertinence des documents (LLM) |
| `rewrite_query` | Reformule la question si grade = not_relevant (max 3 retries) |
| `generate` | Génère la réponse finale avec le contexte disponible |

---

## Corpus de test

| Source | Format | Description |
|---|---|---|
| `capitals_2025_info.pdf` | PDF | Guide du camp de développement 2025 |
| `WASHINGTON_CAPITALS_DATASET.md` | Markdown | Statistiques joueurs 2024-25 et 2025-26 |
| Wikipedia — Alexander Ovechkin | URL | Page Wikipedia indexée au moment de l'ingestion |

---

## Stack technologique

| Composante | Technologie |
|---|---|
| Orchestration agentique | LangGraph |
| Framework RAG | LangChain |
| Base vectorielle | ChromaDB (local) |
| LLM | llama3.2 via Ollama |
| Embeddings | all-MiniLM-L6-v2 (Sentence Transformers) |
| Observabilité | Arize Phoenix (local, sans compte cloud) |
| Recherche web | DuckDuckGo (sans clé API) |
| Interface | Streamlit |
| Mémoire court terme | LangGraph SqliteSaver |
| Mémoire long terme | JSON épisodique (Option B) |