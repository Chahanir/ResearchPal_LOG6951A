import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.strategies import retrieve_cosine_with_scores, retrieve_mmr

#  requêtes test 
TEST_QUERIES = [
    "How many career goals does Ovechkin have?",
    "How did the Capitals perform in the 2024-25 season compared to 2025-26?",
    "Who leads the Capitals in assists?",
    "Why is the Capitals power play struggling this season?",
    "Who are the young players on the Capitals roster?",
]

SEPARATOR = "-" * 60


def display_cosine(query: str, k: int = 4):
    results = retrieve_cosine_with_scores(query, k=k)
    print(f"\n  {'COSINUS':}")
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get("filename", doc.metadata.get("source", "?"))
        doc_type = doc.metadata.get("doc_type", "?")
        print(f"  [{i}] score={score:.4f} | {doc_type} | {source}")
        print(f"      {doc.page_content[:120].strip()}...")


def display_mmr(query: str, k: int = 4):
    results = retrieve_mmr(query, k=k)
    print(f"\n  {'MMR (λ=0.5)':}")
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("filename", doc.metadata.get("source", "?"))
        doc_type = doc.metadata.get("doc_type", "?")
        print(f"  [{i}] {doc_type} | {source}")
        print(f"      {doc.page_content[:120].strip()}...")


def analyze_diversity(cosine_results, mmr_results) -> dict:
    """Calcule des métriques simples de diversité pour le rapport."""
    def unique_sources(docs):
        return len(set(d.metadata.get("doc_type", "?") for d in docs))

    cosine_docs = [doc for doc, _ in cosine_results]
    cosine_diversity = unique_sources(cosine_docs)
    mmr_diversity = unique_sources(mmr_results)

    return {
        "cosine_unique_types": cosine_diversity,
        "mmr_unique_types": mmr_diversity,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("  ResearchPal — Comparaison Retrieval T2")
    print("  Cosinus vs MMR sur 5 requêtes test")
    print("=" * 60)

    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n{'=' * 60}")
        print(f"  Requête {i}/5 : {query}")
        print(SEPARATOR)

        cosine_results = retrieve_cosine_with_scores(query, k=4)
        mmr_results = retrieve_mmr(query, k=4)

        display_cosine(query)
        display_mmr(query)

        metrics = analyze_diversity(cosine_results, mmr_results)
        print(f"\n  Diversité (types de sources uniques) :")
        print(f"    Cosinus : {metrics['cosine_unique_types']}/3 types")
        print(f"    MMR     : {metrics['mmr_unique_types']}/3 types")

    print(f"\n{'=' * 60}")
    print("  Comparaison terminée !")
    print("=" * 60)