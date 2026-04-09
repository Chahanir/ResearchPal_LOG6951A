from datetime import datetime
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_SIZE, CHUNK_OVERLAP

def load_pdf(file_path: str) -> List[Document]:
    """Charge un fichier PDF page par page via PyPDF."""
    from langchain_community.document_loaders import PyPDFLoader

    print(f"  [PDF] Chargement de : {file_path}")
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    for doc in docs:
        doc.metadata.update({
            "source": file_path,
            "doc_type": "pdf",
            "filename": Path(file_path).name,
            "ingestion_date": datetime.now().isoformat(),
        })

    print(f"  [PDF] {len(docs)} page(s) chargée(s)")
    return docs


def load_markdown(file_path: str) -> List[Document]:
    from langchain_text_splitters import MarkdownHeaderTextSplitter

    print(f"  [MD] Chargement de : {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    headers_to_split_on = [
        ("#", "header_1"),
        ("##", "header_2"),
        ("###", "header_3"),
    ]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    raw_docs = splitter.split_text(content)

    # Fusion des petits chunks consécutifs de la même section parent pour enrichir les embeddings
    merged_docs = []
    buffer_content = ""
    buffer_header = ""

    for doc in raw_docs:
        header_2 = doc.metadata.get("header_2", "")

        if header_2 != buffer_header:
            # Nouvelle section, on sauvegarde le buffer précédent
            if buffer_content.strip():
                merged_docs.append(Document(
                    page_content=f"{buffer_header}\n{buffer_content}".strip(),
                    metadata={
                        "source": file_path,
                        "doc_type": "markdown",
                        "filename": Path(file_path).name,
                        "ingestion_date": datetime.now().isoformat(),
                        "section": buffer_header,
                    }
                ))
            buffer_header = header_2
            buffer_content = doc.page_content
        else:
            # Même section, donc on fusionne
            buffer_content += "\n" + doc.page_content

    # Dernier buffer
    if buffer_content.strip():
        merged_docs.append(Document(
            page_content=f"{buffer_header}\n{buffer_content}".strip(),
            metadata={
                "source": file_path,
                "doc_type": "markdown",
                "filename": Path(file_path).name,
                "ingestion_date": datetime.now().isoformat(),
                "section": buffer_header,
            }
        ))

    print(f"  [MD] {len(merged_docs)} section(s) fusionnées chargée(s)")
    return merged_docs


def load_url(url: str) -> List[Document]:
    """Charge une page web et extrait son contenu textuel via BeautifulSoup."""
    from langchain_community.document_loaders import WebBaseLoader
    import bs4

    print(f"  [URL] Chargement de : {url}")

    # cible les balises de contenu principal pour éviter le bruit
    loader = WebBaseLoader(
        web_paths=[url],
        bs_kwargs={
            "parse_only": bs4.SoupStrainer(
                ["p", "h1", "h2", "h3", "h4", "li", "td", "th"]
            )
        },
    )
    docs = loader.load()

    for doc in docs:
        doc.metadata.update({
            "source": url,
            "doc_type": "web",
            "filename": url.split("/")[-1] or "webpage",
            "ingestion_date": datetime.now().isoformat(),
        })

    print(f"  [URL] {len(docs)} document(s) chargé(s)")
    return docs


def load_document(source: str) -> List[Document]:
    """
    Détecte automatiquement le type de source et charge les documents.
    Supporte : URL (http/https), .pdf, .md, .markdown
    """
    if source.startswith("http://") or source.startswith("https://"):
        return load_url(source)

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {source}")

    ext = path.suffix.lower()
    if ext == ".pdf":
        return load_pdf(source)
    elif ext in (".md", ".markdown"):
        return load_markdown(source)
    else:
        raise ValueError(
            f"Type de fichier non supporté : '{ext}'. "
            f"Formats acceptés : .pdf, .md, .markdown, ou une URL."
        )


# Stratégie de segmentation 
def get_splitter() -> RecursiveCharacterTextSplitter:
    """
    Retourne un RecursiveCharacterTextSplitter.

    Paramètres retenus :
      - chunk_size=500  : assez grand pour capturer une idée complète,
                          assez petit pour des embeddings précis
      - chunk_overlap=50 : ~10% de chevauchement pour éviter de couper
                           une phrase à cheval entre deux chunks
      - separators : ordre de priorité pour les coupures
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
    )


def split_documents(docs: List[Document]) -> List[Document]:
    """
    Segmente une liste de documents en chunks.
    Note : les docs Markdown sont déjà pré-découpés par headers —
    on applique quand même le splitter pour uniformiser les tailles.
    """
    splitter = get_splitter()
    chunks = splitter.split_documents(docs)

    print(f"  → {len(docs)} doc(s) → {len(chunks)} chunks "
          f"(taille={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks