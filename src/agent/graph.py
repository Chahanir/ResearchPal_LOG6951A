from __future__ import annotations

import json
import time
from typing import Literal, List

from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage
)
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from src.agent.state import AgentState
from src.agent.memory import build_episodic_prompt, get_checkpointer, add_episodic_entry
from src.agent.tools import TOOLS, search_web
from src.observability.tracing import instrument_node

MAX_RETRIES = 3

# LLM instances
_llm_with_tools = None
_llm_plain = None


def get_llm_with_tools():
    """LLM avec outils liés — pattern du cours : model.bind_tools([...])"""
    global _llm_with_tools
    if _llm_with_tools is None:
        from src.llm_factory import get_llm
        _llm_with_tools = get_llm().bind_tools(TOOLS)
    return _llm_with_tools


def get_llm_plain():
    global _llm_plain
    if _llm_plain is None:
        from src.llm_factory import get_llm
        _llm_plain = get_llm()
    return _llm_plain


# System prompt
def get_system_prompt() -> str:
    episodic = build_episodic_prompt()
    return f"""You are ResearchPal v2, a rigorous research assistant.


    You have two tools:
    1. search_corpus  — search the indexed documents (Washington Capitals, Ovechkin, etc.)
    2. search_web     — DuckDuckGo search for recent information not in the corpus

    Rules:
    - Base your answers ONLY on the retrieved context.
    - Always cite your sources explicitly:
      - For corpus results: mention the filename (e.g. "According to capitals_2025_info.pdf, ...")
      - For web results: mention the title and URL (e.g. "According to NHL.com (https://...), ...")
    - Never invent information.
    - ALWAYS respond in the same language as the user's question.
    {episodic}"""


# Nœud 1 : LLM décide lui-même s'il émet un tool_call ou répond directement

@instrument_node("route_question")
def route_question_node(state: AgentState) -> AgentState:
    """
    Appelle le LLM lié aux outils (bind_tools).
    Le LLM décide dynamiquement :
      - d'émettre un tool_call (search_corpus ou search_web)
      - ou de répondre directement (on ira alors vers retrieve par défaut)
    """
    question = state["question"]
    messages = state.get("messages") or [
        SystemMessage(content=get_system_prompt()),
        HumanMessage(content=question),
    ]

    response = get_llm_with_tools().invoke(messages)
    return {**state, "messages": messages + [response]}


# ToolNode lit les tool_calls du dernier AIMessage, exécute les outils,
_tool_node = ToolNode(TOOLS)


@instrument_node("execute_tools")
def execute_tools_node(state: AgentState) -> AgentState:
    """
    Exécute les outils via ToolNode (pattern standard LangGraph).

    Deux cas :
    - Cas normal   : le LLM a émis un tool_call → ToolNode l'exécute.
    - Cas fallback : arrivée depuis decide_after_grading après MAX_RETRIES,
                     pas de tool_call en attente → on appelle search_web directement.
    """
    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    has_tool_calls = isinstance(last, AIMessage) and getattr(last, "tool_calls", None)

    # Cas fallback : pas de tool_call en attente
    if not has_tool_calls:
        question = state.get("rewritten_question") or state["question"]
        web_content = search_web.invoke({"query": question})
        web_results = None
        if (
            web_content
            and not web_content.startswith("AUCUN")
            and not web_content.startswith("ERREUR")
        ):
            web_results = web_content
        return {**state, "web_results": web_results, "tool_used": "web"}

    # Cas normal : tool_call émis par le LLM
    result = _tool_node.invoke({"messages": messages})
    new_messages = result.get("messages", messages)

    web_results = None
    tool_used = "corpus"

    for msg in new_messages:
        if isinstance(msg, ToolMessage):
            content = msg.content or ""
            if content.startswith("[") or content.startswith("{"):
                try:
                    json.loads(content)
                    web_results = content
                    tool_used = "web"
                except (json.JSONDecodeError, ValueError):
                    pass
            if content == "AUCUN_DOCUMENT_PERTINENT":
                tool_used = "corpus"

    return {**state, "messages": new_messages, "web_results": web_results, "tool_used": tool_used}


# Nœud 2 : récupération depuis ChromaDB (si pas de résultat web pertinent)
@instrument_node("retrieve")
def retrieve_node(state: AgentState) -> AgentState:
    """Récupère les passages pertinents depuis ChromaDB."""
    from src.ingestion.indexer import get_vectorstore

    question = state.get("rewritten_question") or state["question"]
    vs = get_vectorstore()
    docs = vs.similarity_search(question, k=4)

    return {
        **state,
        "documents": docs,
        "tool_used": state.get("tool_used") or "corpus",
    }


# Nœud 3 : grading de la pertinence des documents
@instrument_node("grade_documents")
def grade_documents_node(state: AgentState) -> AgentState:
    """
    Évalue la pertinence des documents récupérés.
    Si web_results disponible (depuis ToolNode web), grade = relevant directement.
    """
    web_results = state.get("web_results", "")
    documents = state.get("documents", [])
    question = state["question"]

    # Résultats web disponibles : grade directement sans LLM 
    if web_results and "AUCUN" not in web_results and "ERREUR" not in web_results:
        return {**state, "grade": "relevant"}

    if not documents:
        return {**state, "grade": "not_relevant"}

    llm = get_llm_plain()
    context = "\n".join([
        f"[Doc {i+1}]: {d.page_content[:300]}"
        for i, d in enumerate(documents[:3])
    ])

    response = llm.invoke([HumanMessage(content=f"""Question : {question}

Documents :
{context}

Ces documents contiennent-ils des informations pertinentes ?
Réponds UNIQUEMENT : "relevant" ou "not_relevant".""")])

    text = response.content.strip().lower()
    grade = "not_relevant" if "not_relevant" in text else "relevant"
    return {**state, "grade": grade}


# Nœud 4 : reformulation de la question après échec du grading
@instrument_node("rewrite_query")
def rewrite_query_node(state: AgentState) -> AgentState:
    """Reformule la question et incrémente retry_count (garde-fou MAX_RETRIES)."""
    retry_count = state.get("retry_count", 0) + 1
    llm = get_llm_plain()

    response = llm.invoke([HumanMessage(content=f"""Reformule cette question avec plus de mots-clés pour améliorer la recherche documentaire.
Question : {state['question']}
Réponds UNIQUEMENT avec la question reformulée.""")])

    rewritten = response.content.strip()
    return {**state, "rewritten_question": rewritten, "retry_count": retry_count, "grade": "not_relevant"}


# 
# Nœud 5 : génération de la réponse finale avec le contexte (corpus ou web)
@instrument_node("generate")
def generate_node(state: AgentState) -> AgentState:
    """Génère la réponse finale avec le contexte (corpus ou web)."""
    llm = get_llm_plain()
    question = state["question"]
    documents = state.get("documents", [])
    web_results = state.get("web_results", "")

    if documents:
        parts = [
            f"[Passage {i+1}] (source: {d.metadata.get('filename', d.metadata.get('source', '?'))})\n{d.page_content}"
            for i, d in enumerate(documents)
        ]
        context = "\n\n---\n\n".join(parts)
        src = "corpus local"
    elif web_results and "AUCUN" not in web_results and "ERREUR" not in web_results:
        try:
            results = json.loads(web_results)
            parts = [f"[Web {i+1}] {r.get('title','')}\n{r.get('body','')}" for i, r in enumerate(results)]
            context = "\n\n---\n\n".join(parts)
        except Exception:
            context = web_results[:1000]
        src = "recherche web"
    else:
        context = "Aucune source pertinente trouvée."
        src = "aucune"

    response = llm.invoke([
        SystemMessage(content=get_system_prompt()),
        HumanMessage(content=f"Contexte ({src}) :\n{context}\n\nQuestion : {question}"),
    ])
    return {**state, "generation": response.content}

 
# Décisions conditionnelles pour les edges du graphe
def decide_after_route(state: AgentState) -> Literal["execute_tools", "retrieve"]:
    """Si le LLM a émis un tool_call → ToolNode, sinon → retrieve."""
    messages = state.get("messages", [])
    last = messages[-1] if messages else None
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "execute_tools"
    return "retrieve"


def decide_after_tools(state: AgentState) -> Literal["retrieve", "generate"]:
    """Résultat web → générer directement. Corpus → passer par ChromaDB retrieve."""
    web = state.get("web_results", "")
    if web and "AUCUN" not in web and "ERREUR" not in web:
        return "generate"
    return "retrieve"


def decide_after_grading(state: AgentState) -> Literal["generate", "rewrite_query", "execute_tools"]:
    """Pertinent → generate. Max retries atteint → web search. Sinon → reformuler."""
    if state.get("grade") == "relevant":
        return "generate"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        return "execute_tools"  # fallback vers search_web
    return "rewrite_query"


# Construction du graphe
def build_graph(use_checkpointer: bool = True):
    """Construit et compile le graphe LangGraph ResearchPal v2."""
    builder = StateGraph(AgentState)

    builder.add_node("route_question", route_question_node)
    builder.add_node("execute_tools", execute_tools_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade_documents", grade_documents_node)
    builder.add_node("rewrite_query", rewrite_query_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "route_question")

    builder.add_conditional_edges(
        "route_question",
        decide_after_route,
        {"execute_tools": "execute_tools", "retrieve": "retrieve"},
    )
    builder.add_conditional_edges(
        "execute_tools",
        decide_after_tools,
        {"generate": "generate", "retrieve": "retrieve"},
    )
    builder.add_edge("retrieve", "grade_documents")
    builder.add_conditional_edges(
        "grade_documents",
        decide_after_grading,
        {"generate": "generate", "rewrite_query": "rewrite_query", "execute_tools": "execute_tools"},
    )
    builder.add_edge("rewrite_query", "retrieve")  # cycle correctif
    builder.add_edge("generate", END)

    if use_checkpointer:
        return builder.compile(checkpointer=get_checkpointer())
    return builder.compile()


# Fonction d'exécution du graphe
def run_agent(question: str, thread_id: str = "default", graph=None) -> dict:
    if graph is None:
        graph = build_graph()

    initial_state: AgentState = {
        "question": question,
        "documents": [],
        "generation": "",
        "retry_count": 0,
        "rewritten_question": None,
        "tool_used": None,
        "web_results": None,
        "grade": None,
        "session_id": thread_id,
        "episodic_context": None,
        "messages": [],
    }

    config = {"configurable": {"thread_id": thread_id}}
    start = time.perf_counter()
    final_state = graph.invoke(initial_state, config=config)
    elapsed = (time.perf_counter() - start) * 1000

    generation = final_state.get("generation", "")
    tool_used = final_state.get("tool_used", "corpus")
    grade = final_state.get("grade", "")

    # Mémoire long terme (Option B) : persister les résolutions réussies comme exemples few-shot.
    # On sauvegarde uniquement si le grade est "relevant" et que la réponse semble valide
    # (non vide, sans message d'erreur ou d'absence de source).
    if (
        generation
        and grade == "relevant"
        and "erreur" not in generation.lower()
        and "aucune source" not in generation.lower()
    ):
        add_episodic_entry(
            question=question,
            answer=generation,
            tool_used=tool_used or "corpus",
            quality_score=1.0,
        )

    return {
        "generation": generation,
        "documents": final_state.get("documents", []),
        "tool_used": tool_used,
        "retry_count": final_state.get("retry_count", 0),
        "grade": grade,
        "latency_ms": elapsed,
    }