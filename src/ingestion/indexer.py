from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import CHROMA_DIR, CHROMA_COLLECTION_NAME
from src.llm_factory import get_embeddings


def get_vectorstore() -> Chroma:
    """Retourne (ou crée) la collection ChromaDB persistante."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def index_documents(chunks: List[Document]) -> Chroma:
    """
    Génère les embeddings des chunks et les indexe dans ChromaDB.
    Retourne le vectorstore mis à jour.
    """
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    print(f"  ✓ {len(chunks)} chunks indexés "
          f"(collection='{CHROMA_COLLECTION_NAME}', dossier='{CHROMA_DIR}')")
    return vectorstore


def ingest_source(source: str) -> Chroma:
    """
    Pipeline d'ingestion complet pour une source :
    chargement → segmentation → indexation.
    """
    from src.ingestion.loader import load_document, split_documents

    print(f"\n📥 Ingestion : {source}")
    docs = load_document(source)
    chunks = split_documents(docs)
    vectorstore = index_documents(chunks)
    return vectorstore


def get_collection_stats() -> dict:
    """Retourne des statistiques sur la collection ChromaDB."""
    vs = get_vectorstore()
    count = vs._collection.count()
    return {
        "collection": CHROMA_COLLECTION_NAME,
        "total_chunks": count,
        "chroma_dir": str(CHROMA_DIR),
    }


def reset_collection():
    """
    Supprime et recrée la collection ChromaDB.
    Utile lors des tests pour repartir d'une base vide.
    """
    vs = get_vectorstore()
    vs.delete_collection()
    print(f"  🗑️  Collection '{CHROMA_COLLECTION_NAME}' réinitialisée.")