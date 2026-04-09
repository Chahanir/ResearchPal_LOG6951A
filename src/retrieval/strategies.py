from typing import List, Tuple
from langchain_core.documents import Document
from src.config import RETRIEVAL_K, MMR_LAMBDA
from src.ingestion.indexer import get_vectorstore


# Similarité cosinus

def retrieve_cosine(query: str, k: int = RETRIEVAL_K) -> List[Document]:
    """
    Recherche par similarité cosinus — stratégie de base obligatoire.
    Retourne les k chunks dont l'embedding est le plus proche de la requête.
    """
    vs = get_vectorstore()
    return vs.similarity_search(query, k=k)


def retrieve_cosine_with_scores(
    query: str, k: int = RETRIEVAL_K
) -> List[Tuple[Document, float]]:
    """Variante avec scores — utile pour le rapport comparatif."""
    vs = get_vectorstore()
    return vs.similarity_search_with_score(query, k=k)


# MMR

def retrieve_mmr(
    query: str,
    k: int = RETRIEVAL_K,
    lambda_mult: float = MMR_LAMBDA
) -> List[Document]:
    """
    MMR : Maximum Marginal Relevance.
    Sélectionne des chunks pertinents ET diversifiés pour éviter la redondance.
    
    lambda_mult=1.0 → pur cosinus (max pertinence)
    lambda_mult=0.0 → max diversité (ignore la pertinence)
    lambda_mult=0.5 → compromis équilibré (valeur retenue)
    
    fetch_k=k*3 : MMR tire d'abord 12 candidats, puis sélectionne les 4
    meilleurs en tenant compte de la diversité.
    """
    vs = get_vectorstore()
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "lambda_mult": lambda_mult,
            "fetch_k": k * 3,
        },
    )
    return retriever.invoke(query)


def retrieve(
    query: str,
    strategy: str = "cosine",
    k: int = RETRIEVAL_K
) -> List[Document]:
    """
    Point d'entrée unifié pour le pipeline RAG (T3) et l'optimisation (T4).
    strategy ∈ {"cosine", "mmr"}
    """
    if strategy == "cosine":
        return retrieve_cosine(query, k=k)
    elif strategy == "mmr":
        return retrieve_mmr(query, k=k)
    else:
        raise ValueError(f"Stratégie inconnue : '{strategy}'. Choisir 'cosine' ou 'mmr'.")