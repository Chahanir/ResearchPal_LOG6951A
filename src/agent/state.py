"""
T1 — Définition du State LangGraph.

Contient tous les champs qui circulent entre les nœuds du graphe agentique.
"""
from __future__ import annotations

from typing import Annotated, List, Optional, Sequence
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """
    État partagé entre tous les nœuds du graphe LangGraph.

    Champs obligatoires (selon l'énoncé TP2) :
    - question     : question courante de l'utilisateur
    - documents    : liste des passages récupérés depuis ChromaDB
    - generation   : réponse générée par le LLM
    - retry_count  : nombre de cycles correctifs effectués (garde-fou max 3)

    Champs supplémentaires :
    - rewritten_question : reformulation de la question après échec du grading
    - tool_used          : outil sélectionné par l'agent ("corpus" | "web")
    - web_results        : résultats bruts de la recherche web
    - grade              : résultat du grading ("relevant" | "not_relevant")
    - session_id         : identifiant de la conversation (pour checkpointer)
    - episodic_context   : exemples few-shot injectés depuis la mémoire long terme
    """
    question: str
    documents: List[Document]
    generation: str
    retry_count: int
    rewritten_question: Optional[str]
    tool_used: Optional[str]
    web_results: Optional[str]
    grade: Optional[str]
    session_id: Optional[str]
    episodic_context: Optional[str]
    messages: Optional[List[BaseMessage]]