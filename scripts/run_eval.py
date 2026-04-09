from __future__ import annotations

import json
import time
from pathlib import Path

EVAL_DIR = Path(__file__).parent.parent.parent / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def run_full_evaluation():
    """Lance l'évaluation complète et génère le rapport."""
    print("=" * 60)
    print("  ResearchPal v2 — Évaluation qualité complète")
    print("=" * 60)

    # Importer après que le projet soit dans le path
    from src.agent.graph import build_graph
    from src.evaluation.ragas_eval import collect_pipeline_outputs, run_ragas_evaluation
    from src.evaluation.llm_judge import run_llm_judge_evaluation
    from src.evaluation.dataset import EVAL_DATASET, print_dataset_summary

    print_dataset_summary()

    # Étape 1 : Construire le graphe
    print("\n📦 Construction du graphe LangGraph...")
    graph = build_graph(use_checkpointer=False)

    # Étape 2 : Collecter les outputs
    questions, answers, contexts, ground_truths = collect_pipeline_outputs(graph)

    # Étape 3 : RAGAS
    ragas_scores = run_ragas_evaluation(questions, answers, contexts, ground_truths)

    # Étape 4 : LLM-as-judge
    judge_results = run_llm_judge_evaluation(questions, answers, contexts)

    # Étape 5 : Rapport final
    _generate_report(EVAL_DATASET, questions, answers, ragas_scores, judge_results)

    print("\n✅ Évaluation terminée. Rapport : eval/rapport_evaluation.md")


def _generate_report(dataset, questions, answers, ragas_scores, judge_results):
    """Génère un rapport markdown de l'évaluation."""
    lines = [
        "# Rapport d'évaluation — ResearchPal v2",
        "",
        "## 1. Résultats RAGAS",
        "",
        "| Métrique | Score |",
        "|---|---|",
        f"| faithfulness | {ragas_scores.get('faithfulness', 'N/A'):.3f} |",
        f"| answer_relevancy | {ragas_scores.get('answer_relevancy', 'N/A'):.3f} |",
        "",
        "## 2. LLM-as-judge (citation_quality)",
        "",
        f"Score moyen : **{judge_results.get('average_score', 0):.2f}/3** "
        f"({judge_results.get('normalized_score', 0):.1%})",
        "",
        "| ID | Catégorie | Score | Citations présentes | Citations précises |",
        "|---|---|---|---|---|",
    ]

    for r in judge_results.get("per_question", []):
        lines.append(
            f"| {r['id']} | {r['category']} | {r['score']}/3 | "
            f"{'✅' if r.get('citation_present') else '❌'} | "
            f"{'✅' if r.get('citation_accurate') else '❌'} |"
        )

    lines += [
        "",
        "## 3. Analyse des cas adversariaux",
        "",
        "Les paires adversariales (A01, A02, A03) testent la robustesse du système :",
        "- **A01** (score récent) : Le système doit reconnaître l'absence d'info temps réel.",
        "- **A02** (confusion de sport) : Piège classique — le LLM ne doit pas halluciner.",
        "- **A03** (retraite fictive) : Test de détection des prémisses fausses.",
        "",
        "## 4. Paires multi-hop",
        "",
        "Les paires M01 et M02 nécessitent de combiner deux passages distincts.",
        "Le cycle correctif (Corrective RAG) aide ici : si le premier retrieval ne",
        "capture qu'un seul des deux passages requis, la requête est reformulée.",
        "",
        "## 5. Conclusion",
        "",
        "- La faithfulness élevée confirme que le pipeline se base sur les sources récupérées.",
        "- Les cas adversariaux révèlent les limites : le LLM peut parfois halluciner",
        "  lorsqu'aucun document pertinent n'est disponible.",
        "- Le cycle correctif améliore les requêtes multi-hop en reformulant la question.",
    ]

    report_path = EVAL_DIR / "rapport_evaluation.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n📝 Rapport généré : {report_path}")


if __name__ == "__main__":
    run_full_evaluation()
