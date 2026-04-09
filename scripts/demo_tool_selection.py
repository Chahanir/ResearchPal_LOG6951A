from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.graph import build_graph, run_agent
from src.observability.tracing import setup_phoenix


def demo():
    print("=" * 65)
    print("  T2 — Démonstration de la sélection dynamique d'outils")
    print("=" * 65)

    # démarrer Phoenix pour tracer
    # setup_phoenix()

    graph = build_graph(use_checkpointer=False)

    examples = [
        {
            "label": "1️⃣  Requête CORPUS — Info dans les documents indexés",
            "question": "Quel est le record de buts en carrière NHL d'Alexander Ovechkin ?",
            "expected_tool": "corpus",
        },
        {
            "label": "2️⃣  Requête WEB — Info hors corpus / actualité récente",
            "question": "Quels sont les résultats de la NHL d'hier soir ?",
            "expected_tool": "web",
        },
        {
            "label": "3️⃣  Requête MIXTE — Corpus + contexte actuel",
            "question": (
                "Ovechkin est-il encore actif en NHL en 2025 et combien de buts "
                "a-t-il marqués selon le corpus indexé ?"
            ),
            "expected_tool": "mixte",
        },
    ]

    for ex in examples:
        print(f"\n{ex['label']}")
        print(f"   Question : {ex['question']}")
        print(f"   Outil attendu : {ex['expected_tool']}")

        result = run_agent(
            question=ex["question"],
            thread_id=f"demo_{ex['expected_tool']}",
            graph=graph,
        )

        print(f"   Outil utilisé  : {result['tool_used']} "
              f"{'✅' if result['tool_used'] == ex['expected_tool'] else '⚠️'}")
        print(f"   Grade obtenu   : {result['grade']}")
        print(f"   Retries        : {result['retry_count']}")
        print(f"   Latence        : {result['latency_ms']:.0f} ms")
        print(f"   Réponse (extrait) : {result['generation'][:200]}...")
        print()

    print("=" * 65)
    print("  Démonstration terminée.")
    print("=" * 65)


if __name__ == "__main__":
    demo()
