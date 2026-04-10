from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Dict, Any

EVAL_OUTPUT_DIR = Path(__file__).parent.parent.parent / "eval"
EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_ragas_evaluation(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truths: List[str],
    output_file: str = "ragas_results.json",
) -> Dict[str, Any]:
    """
    Évalue faithfulness et answer_relevancy via des appels LLM séquentiels directs.

    Implémentation manuelle des métriques RAGAS pour contourner l'infrastructure
    async de RAGAS qui cause des timeouts avec Ollama/llama3.2.

    - faithfulness     : la réponse est-elle fondée sur le contexte fourni ? (0 ou 1)
    - answer_relevancy : la réponse répond-elle à la question posée ? (0 ou 1)
    """
    from langchain_core.messages import HumanMessage
    from src.llm_factory import get_llm

    llm = get_llm()
    print(f"\n🔬 Évaluation RAGAS (séquentielle) sur {len(questions)} paires...")

    FAITH_PROMPT = """Tu es un évaluateur expert en qualité de réponses RAG.

Évalue si la réponse suivante est UNIQUEMENT fondée sur le contexte fourni (faithfulness).
Réponds UNIQUEMENT avec un JSON strict sans aucun texte avant ou après :
{{"faithfulness": <0 ou 1>, "reason": "<explication courte>"}}

0 = la réponse contient des affirmations non présentes dans le contexte (hallucination)
1 = toutes les affirmations de la réponse sont supportées par le contexte

Contexte :
{context}

Question : {question}
Réponse : {answer}"""

    RELEVANCY_PROMPT = """Tu es un évaluateur expert en qualité de réponses RAG.

Évalue si la réponse répond bien à la question posée (answer_relevancy).
Réponds UNIQUEMENT avec un JSON strict sans aucun texte avant ou après :
{{"answer_relevancy": <0 ou 1>, "reason": "<explication courte>"}}

0 = la réponse ne répond pas à la question ou est hors sujet
1 = la réponse répond directement et complètement à la question

Question : {question}
Réponse : {answer}"""

    per_question = []
    faith_scores = []
    relevancy_scores = []
    start = time.perf_counter()

    for i, (q, a, ctx, gt) in enumerate(zip(questions, answers, contexts, ground_truths)):
        print(f"  [{i+1}/{len(questions)}] Q{i+1}...", end=" ", flush=True)

        context_str = "\n\n".join([
            f"[Passage {j+1}]: {c[:400]}" for j, c in enumerate(ctx[:3])
        ])

        # ── Faithfulness ──────────────────────────────────────────────────
        faith, faith_reason = 0, "Erreur LLM"
        try:
            resp = llm.invoke([HumanMessage(content=FAITH_PROMPT.format(
                context=context_str, question=q, answer=a[:600]
            ))])
            content = resp.content.strip()
            if "{" in content:
                parsed = json.loads(content[content.index("{"):content.rindex("}")+1])
                faith = int(parsed.get("faithfulness", 0))
                faith_reason = parsed.get("reason", "")
        except Exception as e:
            faith_reason = str(e)[:80]

        # ── Answer relevancy ──────────────────────────────────────────────
        relevancy, relevancy_reason = 0, "Erreur LLM"
        try:
            resp = llm.invoke([HumanMessage(content=RELEVANCY_PROMPT.format(
                question=q, answer=a[:600]
            ))])
            content = resp.content.strip()
            if "{" in content:
                parsed = json.loads(content[content.index("{"):content.rindex("}")+1])
                relevancy = int(parsed.get("answer_relevancy", 0))
                relevancy_reason = parsed.get("reason", "")
        except Exception as e:
            relevancy_reason = str(e)[:80]

        faith_scores.append(faith)
        relevancy_scores.append(relevancy)
        per_question.append({
            "id": f"Q{i+1}",
            "question": q[:80],
            "faithfulness": faith,
            "faith_reason": faith_reason,
            "answer_relevancy": relevancy,
            "relevancy_reason": relevancy_reason,
        })
        print(f"faith={faith} | relevancy={relevancy}")

    elapsed = time.perf_counter() - start

    scores = {
        "faithfulness":      round(sum(faith_scores) / max(len(faith_scores), 1), 3),
        "answer_relevancy":  round(sum(relevancy_scores) / max(len(relevancy_scores), 1), 3),
        "evaluation_time_s": round(elapsed, 2),
        "n_pairs":           len(questions),
    }

    output_path = EVAL_OUTPUT_DIR / output_file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"scores": scores, "per_question": per_question}, f, ensure_ascii=False, indent=2)

    print(f"✅ Résultats sauvegardés dans {output_path}")
    _print_ragas_table(scores)
    return scores


def _print_ragas_table(scores: Dict[str, Any]) -> None:
    """Affiche un tableau markdown des scores RAGAS."""
    print("\n| Métrique             | Score  |")
    print("|----------------------|--------|")
    print(f"| faithfulness         | {scores.get('faithfulness', 'N/A'):.3f}  |")
    print(f"| answer_relevancy     | {scores.get('answer_relevancy', 'N/A'):.3f}  |")
    print(f"| Paires évaluées      | {scores.get('n_pairs', 'N/A')}     |")
    print(f"| Durée évaluation     | {scores.get('evaluation_time_s', 'N/A')} s   |")


def collect_pipeline_outputs(graph=None) -> tuple:
    """
    Exécute le pipeline agentique sur toutes les paires du dataset
    et collecte les outputs pour l'évaluation.

    Returns:
        Tuple (questions, answers, contexts, ground_truths)
    """
    from src.evaluation.dataset import EVAL_DATASET
    from src.agent.graph import run_agent, build_graph

    if graph is None:
        graph = build_graph(use_checkpointer=False)

    questions, answers, contexts, ground_truths = [], [], [], []

    print(f"\n🤖 Exécution du pipeline sur {len(EVAL_DATASET)} paires...")
    for i, pair in enumerate(EVAL_DATASET, 1):
        print(f"  [{i}/{len(EVAL_DATASET)}] {pair['id']} — {pair['question'][:60]}...")

        try:
            result = run_agent(
                question=pair["question"],
                thread_id=f"eval_{pair['id']}",
                graph=graph,
            )
            answer = result["generation"]
            docs = result["documents"]
            ctx = [doc.page_content for doc in docs] if docs else ["Aucun document récupéré."]
        except Exception as e:
            print(f"  ⚠️  Erreur : {e}")
            answer = "Erreur lors de la génération."
            ctx = ["Erreur lors du retrieval."]

        questions.append(pair["question"])
        answers.append(answer)
        contexts.append(ctx)
        ground_truths.append(pair["reference"])

    return questions, answers, contexts, ground_truths


def configure_ragas_for_ollama():
    """Conservé pour compatibilité — l'évaluation est maintenant séquentielle."""
    print("✅ Évaluation séquentielle activée (pas de dépendance OpenAI)")
    return True


if __name__ == "__main__":
    q, a, c, gt = collect_pipeline_outputs()
    run_ragas_evaluation(q, a, c, gt)