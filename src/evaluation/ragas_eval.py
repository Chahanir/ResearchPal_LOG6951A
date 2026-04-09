"""
T5 — Évaluation qualité avec RAGAS.

Calcule faithfulness et answer_relevance sur les 15 paires du dataset.
Génère un rapport JSON + tableau markdown dans eval/ragas_results.json.

Usage :
    python -m src.evaluation.ragas_eval

Prérequis :
    pip install ragas
"""
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
    Exécute l'évaluation RAGAS sur le dataset fourni.

    Args:
        questions: Liste des questions.
        answers: Réponses générées par le pipeline agentique.
        contexts: Liste de listes de passages récupérés (un par question).
        ground_truths: Réponses de référence.
        output_file: Nom du fichier de sortie JSON.

    Returns:
        Dictionnaire des scores moyens {metric: score}.
    """
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
        from datasets import Dataset
    except ImportError:
        print("⚠️  RAGAS non installé. Exécuter : pip install ragas datasets")
        return {}

    print(f"\n🔬 Évaluation RAGAS sur {len(questions)} paires...")

    # Construire le dataset HuggingFace
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }
    dataset = Dataset.from_dict(data)

    # Calcul des métriques
    start = time.perf_counter()
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
    )
    elapsed = time.perf_counter() - start

    scores = {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["answer_relevancy"]),
        "evaluation_time_s": round(elapsed, 2),
        "n_pairs": len(questions),
    }

    # Sauvegarde
    output_path = EVAL_OUTPUT_DIR / output_file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "scores": scores,
            "per_question": result.to_pandas().to_dict(orient="records"),
        }, f, ensure_ascii=False, indent=2)

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
    et collecte les outputs pour l'évaluation RAGAS.

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


if __name__ == "__main__":
    q, a, c, gt = collect_pipeline_outputs()
    run_ragas_evaluation(q, a, c, gt)


def configure_ragas_for_ollama():
    """
    Configure RAGAS pour utiliser Ollama (llama3.2) au lieu d'OpenAI.
    RAGAS nécessite une configuration explicite du LLM d'évaluation.

    IMPORTANT : appeler cette fonction AVANT run_ragas_evaluation().
    """
    try:
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas import evaluate
        from src.llm_factory import get_llm, get_embeddings

        llm_wrapper = LangchainLLMWrapper(get_llm())
        emb_wrapper = LangchainEmbeddingsWrapper(get_embeddings())

        # Injecter dans les métriques RAGAS
        from ragas.metrics import faithfulness, answer_relevancy
        faithfulness.llm = llm_wrapper
        faithfulness.embeddings = emb_wrapper
        answer_relevancy.llm = llm_wrapper
        answer_relevancy.embeddings = emb_wrapper

        print("✅ RAGAS configuré pour Ollama (llama3.2)")
        return True
    except Exception as e:
        print(f"⚠️  Configuration RAGAS/Ollama échouée : {e}")
        print("   → Assurez-vous qu'Ollama est démarré et que llama3.2 est disponible")
        return False
