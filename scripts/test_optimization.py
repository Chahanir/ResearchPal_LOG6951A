import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.query_optimization.optimizer import retrieve_multi_query

SEPARATOR = "=" * 60

# 3 requêtes pour montrer l'impact de l'optimisation
TEST_QUERIES = [
    "Ovechkin record",                          # Requête courte et ambiguë
    "Capitals young players future",            # Requête large
    "Washington performance this season",       # Requête vague
]


def display_docs(docs, label: str):
    print(f"\n  {label}")
    seen_types = set()
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("filename", doc.metadata.get("source", "?"))
        doc_type = doc.metadata.get("doc_type", "?")
        seen_types.add(doc_type)
        print(f"  [{i}] {doc_type:8} | {source[:40]:40} | {doc.page_content[:80].strip()}...")
    return seen_types


if __name__ == "__main__":
    print(SEPARATOR)
    print("  ResearchPal — Optimisation de requête T4")
    print("  Multi-Query + RRF : avant vs après")
    print(SEPARATOR)

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{'─' * 60}")
        print(f"  Exemple {i}/3 : '{query}'")
        print(f"{'─' * 60}")

        result = retrieve_multi_query(query, n=3, k=4)

        # Affichage des variantes générées
        print(f"\n  Variantes générées par le LLM :")
        for j, v in enumerate(result["variants"], 1):
            print(f"    {j}. {v}")

        # Avant (cosinus simple)
        before_types = display_docs(result["original_docs"], "AVANT (cosinus simple)")

        # Après (multi-query + RRF)
        after_types = display_docs(result["docs"], "APRÈS (multi-query + RRF)")

        # Analyse d'impact
        before_sources = set(
            d.metadata.get("doc_type") for d in result["original_docs"]
        )
        after_sources = set(
            d.metadata.get("doc_type") for d in result["docs"]
        )

        new_sources = after_sources - before_sources
        print(f"\n  Impact :")
        print(f"    Types avant : {before_sources}")
        print(f"    Types après : {after_sources}")
        if new_sources:
            print(f" Nouvelles sources introduites : {new_sources}")
        else:
            print(f"    → Même diversité de sources")

    print(f"\n{SEPARATOR}")
    print("Test T4 terminé")
    print(SEPARATOR)