"""
T5 — LLM-as-judge : évaluateur de qualité de citations de sources.

Critère évalué : QUALITÉ DES CITATIONS DE SOURCES
(critère complémentaire non couvert par RAGAS faithfulness/relevancy)

Le prompt d'évaluation est structuré avec :
  - Des critères explicites
  - Un barème de notation [0-3]
  - Un format de sortie JSON strict

Usage :
    python -m src.evaluation.llm_judge

Le prompt complet est reproduit en commentaire ci-dessous (requis par l'énoncé).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Any

EVAL_OUTPUT_DIR = Path(__file__).parent.parent.parent / "eval"
EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# PROMPT D'ÉVALUATION LLM-AS-JUDGE (inclus en annexe selon l'énoncé)
# ---------------------------------------------------------------------------
#
# SYSTEM_JUDGE_PROMPT :
# """
# Tu es un évaluateur expert en qualité des réponses de systèmes RAG.
# Ta mission est d'évaluer la QUALITÉ DES CITATIONS DE SOURCES dans une réponse.
#
# Critères d'évaluation — Qualité des citations :
# 1. PRÉSENCE : La réponse cite-t-elle explicitement ses sources ?
# 2. PRÉCISION : Les sources citées correspondent-elles aux passages réellement utilisés ?
# 3. FORMAT : Les citations suivent-elles un format cohérent et lisible ?
# 4. HONNÊTETÉ : La réponse admet-elle clairement quand aucune source n'est disponible ?
#
# Barème :
#   3 — Excellent : Sources citées précisément, format cohérent, honnêteté parfaite
#   2 — Bon       : Sources citées mais imprécises ou format inconsistant
#   1 — Faible    : Citations partielles ou trompeuses
#   0 — Absent    : Aucune citation, ou citations inventées
#
# Format de sortie OBLIGATOIRE (JSON strict, aucun texte avant ou après) :
# {
#   "score": <entier 0-3>,
#   "reasoning": "<explication en 2-3 phrases>",
#   "citation_present": <true|false>,
#   "citation_accurate": <true|false>,
#   "admits_ignorance_correctly": <true|false>
# }
# """
#
# USER_JUDGE_PROMPT :
# """
# Question posée : {question}
#
# Contexte fourni au système (passages récupérés) :
# {context}
#
# Réponse générée par le système RAG :
# {answer}
#
# Évalue la qualité des citations de sources dans cette réponse selon les critères.
# Réponds UNIQUEMENT avec le JSON demandé.
# """
#
# ---------------------------------------------------------------------------

SYSTEM_JUDGE_PROMPT = """Tu es un évaluateur expert en qualité des réponses de systèmes RAG.
Ta mission est d'évaluer la QUALITÉ DES CITATIONS DE SOURCES dans une réponse.

Critères d'évaluation — Qualité des citations :
1. PRÉSENCE : La réponse cite-t-elle explicitement ses sources ?
2. PRÉCISION : Les sources citées correspondent-elles aux passages réellement utilisés ?
3. FORMAT : Les citations suivent-elles un format cohérent et lisible ?
4. HONNÊTETÉ : La réponse admet-elle clairement quand aucune source n'est disponible ?

Barème :
  3 — Excellent : Sources citées précisément, format cohérent, honnêteté parfaite
  2 — Bon       : Sources citées mais imprécises ou format inconsistant
  1 — Faible    : Citations partielles ou trompeuses
  0 — Absent    : Aucune citation, ou citations inventées

Format de sortie OBLIGATOIRE (JSON strict, aucun texte avant ou après) :
{{
  "score": <entier 0-3>,
  "reasoning": "<explication en 2-3 phrases>",
  "citation_present": <true|false>,
  "citation_accurate": <true|false>,
  "admits_ignorance_correctly": <true|false>
}}"""

USER_JUDGE_PROMPT = """Question posée : {question}

Contexte fourni au système (passages récupérés) :
{context}

Réponse générée par le système RAG :
{answer}

Évalue la qualité des citations de sources dans cette réponse selon les critères.
Réponds UNIQUEMENT avec le JSON demandé."""


def judge_single_response(
    question: str,
    answer: str,
    context: List[str],
    llm=None,
) -> Dict[str, Any]:
    """
    Évalue une réponse unique avec le LLM-as-judge.

    Args:
        question: Question posée.
        answer: Réponse générée par le pipeline.
        context: Liste de passages récupérés.
        llm: Instance LLM (si None, utilise get_llm()).

    Returns:
        Dict avec score, reasoning et méta-indicateurs.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    if llm is None:
        from src.llm_factory import get_llm
        llm = get_llm()

    context_str = "\n\n".join([f"[Passage {i+1}]: {c[:400]}" for i, c in enumerate(context[:3])])

    user_prompt = USER_JUDGE_PROMPT.format(
        question=question,
        context=context_str,
        answer=answer[:800],
    )

    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_JUDGE_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        content = response.content.strip()

        # Extraire le JSON de la réponse
        if "{" in content:
            start = content.index("{")
            end = content.rindex("}") + 1
            result = json.loads(content[start:end])
        else:
            result = {"score": 0, "reasoning": "Format de sortie invalide", "error": content}

    except Exception as e:
        result = {"score": 0, "reasoning": f"Erreur LLM : {str(e)}"}

    return result


def run_llm_judge_evaluation(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    output_file: str = "llm_judge_results.json",
) -> Dict[str, Any]:
    """
    Exécute le LLM-as-judge sur toutes les paires du dataset.

    Returns:
        Dictionnaire avec score moyen et résultats par question.
    """
    from src.llm_factory import get_llm
    from src.evaluation.dataset import EVAL_DATASET

    llm = get_llm()
    results = []
    total_score = 0

    print(f"\n⚖️  LLM-as-judge sur {len(questions)} paires...")

    for i, (q, a, c) in enumerate(zip(questions, answers, contexts)):
        pair_id = EVAL_DATASET[i]["id"] if i < len(EVAL_DATASET) else f"Q{i+1}"
        category = EVAL_DATASET[i]["category"] if i < len(EVAL_DATASET) else "unknown"

        print(f"  [{i+1}/{len(questions)}] {pair_id} ({category})...")

        judgment = judge_single_response(q, a, c, llm)
        score = judgment.get("score", 0)
        total_score += score

        results.append({
            "id": pair_id,
            "category": category,
            "question": q[:80],
            "score": score,
            "reasoning": judgment.get("reasoning", ""),
            "citation_present": judgment.get("citation_present", False),
            "citation_accurate": judgment.get("citation_accurate", False),
            "admits_ignorance_correctly": judgment.get("admits_ignorance_correctly", False),
        })

    avg_score = total_score / len(questions) if questions else 0
    normalized = avg_score / 3.0  # Normaliser sur [0, 1] pour comparaison avec RAGAS

    summary = {
        "criterion": "citation_quality",
        "scale": "0-3",
        "average_score": round(avg_score, 3),
        "normalized_score": round(normalized, 3),
        "n_pairs": len(questions),
        "per_question": results,
    }

    output_path = EVAL_OUTPUT_DIR / output_file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Résultats LLM-as-judge sauvegardés dans {output_path}")
    print(f"   Score moyen citation_quality : {avg_score:.2f}/3 ({normalized:.2%})")

    return summary


if __name__ == "__main__":
    from src.evaluation.ragas_eval import collect_pipeline_outputs

    q, a, c, _ = collect_pipeline_outputs()
    run_llm_judge_evaluation(q, a, c)
