import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.generation.rag_pipeline import RAGPipeline

SEPARATOR = "=" * 60


def print_result(turn: int, question: str, result: dict):
    print(f"\n{'─' * 60}")
    print(f"  Tour {turn} — {question}")
    print(f"{'─' * 60}")
    print(f"\n{result['answer']}")
    print(f"\n📎 Sources :\n{result['sources']}")


if __name__ == "__main__":
    print(SEPARATOR)
    print("  ResearchPal — Test Pipeline RAG T3")
    print(SEPARATOR)

    pipeline = RAGPipeline(retrieval_strategy="mmr", k=4)

    # Test 1 : Question connue en anglais
    print("\n>>> TEST 1 : Question générale connue")
    result0 = pipeline.ask("Can you name the Washington Capitals young prospects ?")
    print_result(0, "Washington Capitals young prospects ?", result0)

    # Test 2 : conversation sur 3 tours (exigence T3) 
    print("\n>>> TEST 2 : Conversation multi-tours (3 tours)")

    result1 = pipeline.ask("How many points did Dylan Strome score in the 2024-25 season?")
    print_result(1, "Dylan Strome point total in 2024-25?", result1)

    result2 = pipeline.ask("How does that compare to his 2025-26 season so far?")
    print_result(2, "How does that compare to his 2025-26 season so far?", result2)

    result3 = pipeline.ask("Who else had a strong offensive season for the Capitals in 2024-25?")
    print_result(3, "Who else had a strong offensive season for the Capitals in 2024-25?", result3)

    print(f"\n✅ {pipeline.turn_count} tours complétés dans l'historique")

    # Test 3 : requête hors corpus (cas limite)
    print(f"\n\n>>> TEST 3 : Cas limite — information absente du corpus")
    pipeline.clear_history()

    result4 = pipeline.ask("What is the salary cap situation for the Capitals?")
    print_result(1, "Salary cap situation?", result4)

    # Test 4 : requête en français
    print(f"\n\n>>> TEST 4 : Requête en français")
    pipeline.clear_history()

    result5 = pipeline.ask("Qui est le meilleur buteur de la Ligue Nationale de hockey ?")
    print_result(1, "Meilleur buteur de la Ligue Nationale de hockey ?", result5)

    # ─── Test 5 : pipeline avec optimisation multi-query activée (T4) ─────────────
    print(f"\n\n>>> TEST 5 : Pipeline avec multi-query activé")
    pipeline_optim = RAGPipeline(retrieval_strategy="cosine", k=4, use_query_optimization=True)

    result_optim = pipeline_optim.ask("Who are the young prospects for the Capitals?")
    print_result(1, "Young prospects (multi-query)", result_optim)

    if result_optim["variants"]:
        print(f"\n  Variantes générées :")
        for j, v in enumerate(result_optim["variants"], 1):
            print(f"    {j}. {v}")
    else:
        print(" Aucune variante générée")
    print(f"\n{SEPARATOR}")
    print("Tests T3 terminés")
    print(SEPARATOR)