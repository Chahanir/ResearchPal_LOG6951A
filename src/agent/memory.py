from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

MEMORY_DIR = Path(__file__).parent.parent.parent / "memory"
EPISODIC_FILE = MEMORY_DIR / "episodic_memory.json"
CHECKPOINT_DB = MEMORY_DIR / "checkpoints.sqlite"

MEMORY_DIR.mkdir(parents=True, exist_ok=True)

MAX_EPISODIC_ENTRIES = 5  # Stocker uniquement les N meilleures résolutions


# Niveau 1 : Mémoire à court terme => checkpointer du graphe (Option A)

def get_checkpointer():
    """
    Retourne un checkpointer SQLite pour persister l'état LangGraph.

    Le fichier SQLite est stocké dans memory/checkpoints.sqlite.
    Chaque thread_id (session) a son propre état persistant.
    Après un redémarrage du processus, l'agent peut reprendre là où il en était.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        import sqlite3
        conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
        return SqliteSaver(conn)
    except ImportError:
        # Fallback en mémoire si langgraph.checkpoint.sqlite non disponible
        from langgraph.checkpoint.memory import MemorySaver
        print("⚠️  SqliteSaver non disponible — fallback MemorySaver (non persistant)")
        return MemorySaver()


# Niveau 2 : Mémoire à long terme => mémoire épisodique de résolutions réussies (Option B)

def load_episodic_memory() -> List[Dict[str, Any]]:
    """Charge les exemples épisodiques depuis le fichier JSON persistant."""
    if not EPISODIC_FILE.exists():
        return []
    try:
        with open(EPISODIC_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_episodic_memory(entries: List[Dict[str, Any]]) -> None:
    """Sauvegarde la liste d'exemples épisodiques dans le fichier JSON."""
    with open(EPISODIC_FILE, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def add_episodic_entry(
    question: str,
    answer: str,
    tool_used: str,
    quality_score: float = 1.0,
) -> None:
    """
    Ajoute une résolution réussie à la mémoire épisodique.

    Garde uniquement les MAX_EPISODIC_ENTRIES entrées avec le meilleur score.
    Si le fichier est plein, remplace l'entrée avec le score le plus bas.

    Args:
        question: La question de l'utilisateur.
        answer: La réponse générée par l'agent.
        tool_used: L'outil utilisé ("corpus", "web", "mixte").
        quality_score: Score de qualité estimé [0.0, 1.0] (ex. score RAGAS).
    """
    entries = load_episodic_memory()

    new_entry = {
        "question": question,
        "answer": answer[:800],  # Tronquer pour limiter la taille du contexte
        "tool_used": tool_used,
        "quality_score": quality_score,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    entries.append(new_entry)

    # Trier par score décroissant et garder les N meilleurs
    entries = sorted(entries, key=lambda x: x["quality_score"], reverse=True)
    entries = entries[:MAX_EPISODIC_ENTRIES]

    save_episodic_memory(entries)


def build_episodic_prompt() -> str:
    """
    Construit le bloc few-shot à injecter dans le system prompt.

    Retourne une chaîne vide si aucun exemple n'est disponible.
    """
    entries = load_episodic_memory()
    if not entries:
        return ""

    lines = [
        "\n\n--- PAST RESOLUTION EXAMPLES (episodic memory) ---",
        "Here are examples of successfully resolved questions. "
        "Use them as a guide for style and rigor. "
        "Always reply in the language of the current user's question, regardless of the language of these examples.\n",
    ]

    for i, e in enumerate(entries, 1):
        lines.append(f"Example {i} (tool: {e['tool_used']}):")
        lines.append(f"  Q: {e['question']}")
        lines.append(f"  A: {e['answer'][:300]}...")
        lines.append("")

    lines.append("--- END OF EXAMPLES ---\n")
    return "\n".join(lines)