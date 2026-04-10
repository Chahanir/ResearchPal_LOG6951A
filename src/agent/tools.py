from __future__ import annotations

import json
from typing import Optional

from langchain_core.tools import tool

# Outil 1 : Recherche dans le corpus indexé (ChromaDB)
@tool
def search_corpus(query: str, k: int = 4) -> str:
    """Recherche des passages pertinents dans le corpus de documents indexés (ChromaDB).

    QUAND utiliser cet outil :
    - La question porte sur le contenu des documents ingérés 
    - L'utilisateur demande une information factuelle tirée du corpus local.
    - La question contient des termes comme « selon le document », « dans les sources »,
      « d'après les fichiers indexés ».

    QUAND NE PAS utiliser cet outil :
    - La question porte sur des événements récents non couverts par le corpus.
    - La question est générale (météo, actualités, calculs mathématiques purs).
    - Un appel précédent a déjà confirmé que le corpus ne contient pas la réponse
      (grade = "not_relevant" après 3 tentatives).

    Format de retour :
    Chaîne de caractères contenant jusqu'à k passages formatés ainsi :
    "[Passage N] (source: <nom_fichier>)\n<contenu>"
    Les passages sont séparés par "---".
    Si aucun passage n'est trouvé, retourne "AUCUN_DOCUMENT_PERTINENT".

    Args:
        query: La requête de recherche à envoyer au vectorstore.
        k: Nombre de passages à récupérer (défaut : 4).
    """
    try:
        # Import local pour éviter les dépendances circulaires
        from src.ingestion.indexer import get_vectorstore

        vs = get_vectorstore()
        docs = vs.similarity_search(query, k=k)

        if not docs:
            return "AUCUN_DOCUMENT_PERTINENT"

        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("filename", doc.metadata.get("source", "inconnu"))
            parts.append(f"[Passage {i}] (source: {source})\n{doc.page_content}")

        return "\n\n---\n\n".join(parts)

    except Exception as e:
        return f"ERREUR_CORPUS: {str(e)}"


# Outil 2 : Recherche web DuckDuckGo

@tool
def search_web(query: str, max_results: int = 3) -> str:
    """Effectue une recherche sur le web via DuckDuckGo (sans clé API).

    QUAND utiliser cet outil :
    - La question porte sur des événements récents, de l'actualité ou des
      informations non couvertes par le corpus local.
    - Le nœud grade_documents a évalué les documents du corpus comme non pertinents
      et la question nécessite tout de même une réponse.
    - L'utilisateur demande explicitement des informations actuelles
      (ex. : « dernières nouvelles », « score d'hier », « résultats récents »).
    - La question est mixte : une partie nécessite le corpus, l'autre le web.

    QUAND NE PAS utiliser cet outil :
    - La réponse est clairement dans le corpus (éviter la sur-sollicitation du web).
    - La question est purement calculatoire ou logique.
    - La connexion Internet n'est pas disponible.

    Format de retour :
    JSON sérialisé contenant une liste de résultats avec les champs :
    [{"title": "...", "href": "...", "body": "..."}, ...]
    Si aucun résultat, retourne "AUCUN_RESULTAT_WEB".
    En cas d'erreur réseau, retourne "ERREUR_WEB: <message>".

    Args:
        query: La requête de recherche.
        max_results: Nombre maximum de résultats (défaut : 3).
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS  # fallback ancien nom

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "href": r.get("href", ""),
                    "body": r.get("body", "")[:500],  # Tronquer pour économiser des tokens
                })

        if not results:
            return "AUCUN_RESULTAT_WEB"

        return json.dumps(results, ensure_ascii=False)

    except ImportError:
        return "ERREUR_WEB: duckduckgo-search non installé. Exécuter : pip install duckduckgo-search"
    except Exception as e:
        return f"ERREUR_WEB: {str(e)}"


# Liste exportée pour le graphe LangGraph
TOOLS = [search_corpus, search_web]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}