from typing import List, Tuple
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from src.config import MAX_HISTORY_TURNS, RETRIEVAL_K
from src.llm_factory import get_llm
from src.retrieval.strategies import retrieve


#  System Prompt
SYSTEM_PROMPT = """You are ResearchPal, a rigorous and reliable personal research assistant.

Your role:
- Answer questions based ONLY on the passages provided in the context below.
- Always cite your sources explicitly at the end of your response using this format:
  [Source 1]: <filename or URL>
  [Source 2]: <filename or URL>
- If the context does not contain enough information to answer, clearly state:
  "I could not find relevant information on this topic in the indexed documents."
- Never invent information or extrapolate beyond the provided context.
- Be factual, precise and concise.

Response format:
1. Direct answer to the question (2-5 sentences)
2. **Sources** section listing the passages used

Always respond in the same language as the user's question."""


def format_context(docs: List[Document]) -> str:
    """Formate les documents récupérés en bloc de contexte pour le prompt."""
    if not docs:
        return "No relevant documents found."
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("filename", doc.metadata.get("source", "unknown"))
        doc_type = doc.metadata.get("doc_type", "")
        parts.append(
            f"[Passage {i}] (source: {source}, type: {doc_type})\n"
            f"{doc.page_content}"
        )
    return "\n\n---\n\n".join(parts)


def format_sources(docs: List[Document]) -> str:
    """Formate la liste des sources pour l'affichage dans l'interface (T5)."""
    seen = set()
    sources = []
    for doc in docs:
        source = doc.metadata.get("filename", doc.metadata.get("source", "unknown"))
        doc_type = doc.metadata.get("doc_type", "")
        date = doc.metadata.get("ingestion_date", "")[:10]
        key = f"{source}_{doc_type}"
        if key not in seen:
            seen.add(key)
            sources.append(f"• {source} ({doc_type}) — indexé le {date}")
    return "\n".join(sources)


class RAGPipeline:
    """
    Pipeline RAG conversationnel avec gestion de l'historique.
    Gère au moins 3 tours de dialogue (T3).
    """

    def __init__(self, retrieval_strategy: str = "cosine", k: int = RETRIEVAL_K, use_query_optimization: bool = False):
        self.llm = get_llm()
        self.retrieval_strategy = retrieval_strategy
        self.k = k
        self.use_query_optimization = use_query_optimization
        # Historique : liste de tuples (question_utilisateur, réponse_assistant)
        self.history: List[Tuple[str, str]] = []

    def ask(self, question: str) -> dict:
        """
        Soumet une question au pipeline RAG.

        Retourne un dict :
          - answer  : réponse générée par le LLM
          - sources : sources formatées pour l'affichage
          - docs    : documents récupérés (pour debug)
        """
        # 1. Récupération des documents pertinents
        variants = []

        if self.use_query_optimization:
            from src.query_optimization.optimizer import retrieve_multi_query
            result = retrieve_multi_query(question, k=self.k)
            docs = result["docs"]
            variants = result["variants"]
        else:
            docs = retrieve(question, strategy=self.retrieval_strategy, k=self.k)

        context = format_context(docs)

        # 2. Construction des messages
        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        # Historique tronqué aux N derniers tours
        for human_msg, ai_msg in self.history[-MAX_HISTORY_TURNS:]:
            messages.append(HumanMessage(content=human_msg))
            messages.append(AIMessage(content=ai_msg))

        # Message utilisateur avec contexte RAG injecté
        user_message = (
            f"Context (retrieved passages):\n{context}\n\n"
            f"Question: {question}"
        )
        messages.append(HumanMessage(content=user_message))

        # 3. Génération
        response = self.llm.invoke(messages)
        answer = response.content

        # 4. Mise à jour de l'historique
        self.history.append((question, answer))

        return {
            "answer": answer,
            "sources": format_sources(docs),
            "docs": docs,
            "variants": variants,
        }

    def clear_history(self):
        """Réinitialise l'historique de conversation."""
        self.history = []
        print("Historique effacé.")

    @property
    def turn_count(self) -> int:
        """Nombre de tours de dialogue dans la session courante."""
        return len(self.history)