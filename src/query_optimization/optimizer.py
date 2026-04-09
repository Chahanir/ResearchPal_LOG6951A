from typing import List
from collections import defaultdict
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from src.config import MULTI_QUERY_N, RETRIEVAL_K
from src.llm_factory import get_llm
from src.retrieval.strategies import retrieve_cosine


# Prompt de génération de variantes
MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(
    """You are an assistant helping to improve document retrieval.
Generate {n} different reformulations of the following question.
Each variant should explore a different angle or use different terminology.
Respond with ONE variant per line, no numbering, no explanation.

Original question: {question}
Variants:"""
)


def generate_query_variants(question: str, n: int = MULTI_QUERY_N) -> List[str]:
    """Génère N variantes de la requête via le LLM."""
    llm = get_llm()
    chain = MULTI_QUERY_PROMPT | llm
    response = chain.invoke({"question": question, "n": n})
    variants = [
        line.strip()
        for line in response.content.strip().split("\n")
        if line.strip()
    ]
    return variants[:n]


def reciprocal_rank_fusion(
    results_list: List[List[Document]], k: int = 60
) -> List[Document]:
    """
    Reciprocal Rank Fusion — fusionne plusieurs listes de résultats.
    Formule : score(doc) = Σ 1/(k + rang_i)
    k=60 est la valeur standard recommandée dans la littérature.
    """
    scores: dict = defaultdict(float)
    doc_map: dict = {}

    for results in results_list:
        for rank, doc in enumerate(results):
            doc_id = doc.page_content[:150]
            scores[doc_id] += 1.0 / (k + rank + 1)
            doc_map[doc_id] = doc

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[doc_id] for doc_id in sorted_ids]


def retrieve_multi_query(
    question: str,
    n: int = MULTI_QUERY_N,
    k: int = RETRIEVAL_K
) -> dict:
    """
    Pipeline multi-query complet :
    1. Génère N variantes de la requête
    2. Récupère k documents pour chaque variante + requête originale
    3. Fusionne via RRF

    Retourne :
      - docs          : résultats fusionnés (optimisés)
      - variants      : variantes générées
      - original_docs : résultats sans optimisation (pour comparaison)
    """
    # Résultats de base pour comparaison avant/après
    original_docs = retrieve_cosine(question, k=k)

    # Génération des variantes
    variants = generate_query_variants(question, n=n)

    # Récupération pour chaque variante
    all_results = [original_docs]
    for variant in variants:
        docs = retrieve_cosine(variant, k=k)
        all_results.append(docs)

    # Fusion RRF
    fused_docs = reciprocal_rank_fusion(all_results)[:k]

    return {
        "docs": fused_docs,
        "variants": variants,
        "original_docs": original_docs,
    }