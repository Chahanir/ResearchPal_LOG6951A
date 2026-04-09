import sys
import os

# Permet d'importer les modules src/ depuis la racine du projet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion.indexer import ingest_source, get_collection_stats, reset_collection

# Corpus 
SOURCES = [
    "data/sample_docs/capitals_2025_info.pdf",
    "data/sample_docs/WASHINGTON_CAPITALS_DATASET.md",
    "https://en.wikipedia.org/wiki/Alexander_Ovechkin",
]

if __name__ == "__main__":
    if "--reset" in sys.argv:
        print("🗑️  Réinitialisation de la base vectorielle...")
        reset_collection()

    print("=" * 55)
    print("  ResearchPal — Ingestion T1")
    print("=" * 55)

    errors = []
    for source in SOURCES:
        try:
            ingest_source(source)
        except Exception as e:
            print(f" Erreur sur '{source}' : {e}")
            errors.append(source)

    print("\n" + "=" * 55)
    stats = get_collection_stats()
    print(f" Ingestion terminée")
    print(f" Collection  : {stats['collection']}")
    print(f" Total chunks : {stats['total_chunks']}")
    print(f" ChromaDB    : {stats['chroma_dir']}")

    if errors:
        print(f"\n  ⚠️  {len(errors)} source(s) en erreur : {errors}")

    # Aperçu des chunks
    print("\n" + "=" * 55)
    print("  Aperçu de 3 chunks aléatoires")
    print("=" * 55)

    from src.ingestion.indexer import get_vectorstore
    vs = get_vectorstore()
    sample = vs.similarity_search("Ovechkin goals", k=3)

    for i, doc in enumerate(sample, 1):
        print(f"\n[Chunk {i}]")
        print(f"  Source    : {doc.metadata.get('source', '?')}")
        print(f"  Type      : {doc.metadata.get('doc_type', '?')}")
        print(f"  Contenu   : {doc.page_content[:150].strip()}...")