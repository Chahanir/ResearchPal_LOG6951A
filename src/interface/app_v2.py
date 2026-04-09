import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import time
import uuid

import streamlit as st

st.set_page_config(
    page_title="ResearchPal v2",
    page_icon="🤖",
    layout="wide",
)


# Initialisation de la session Streamlit
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "graph" not in st.session_state:
    with st.spinner("⚙️ Chargement du graphe LangGraph..."):
        from src.agent.graph import build_graph
        st.session_state.graph = build_graph(use_checkpointer=True)

if "phoenix_active" not in st.session_state:
    st.session_state.phoenix_active = False

# Sidebar
with st.sidebar:
    st.title("⚙️ ResearchPal v2")
    st.caption("Pipeline RAG Agentique — LOG6951A TP2")
    st.divider()

    # Session
    st.subheader("📋 Session")
    st.code(f"Thread ID : {st.session_state.thread_id}")
    if st.button("🔄 Nouvelle session"):
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.messages = []
        st.rerun()

    st.divider()

    # Observabilité
    st.subheader("🔭 Observabilité")
    if st.button("🚀 Démarrer Phoenix"):
        from src.observability.tracing import setup_phoenix
        if setup_phoenix():
            st.session_state.phoenix_active = True
            st.success("Phoenix actif sur http://localhost:6006")
        else:
            st.error("Phoenix non disponible — pip install arize-phoenix")

    if st.session_state.phoenix_active:
        st.markdown("📊 [Ouvrir Phoenix](http://localhost:6006)")

    st.divider()

    # Ingestion de documents
    st.subheader("📥 Ajouter un document")
    uploaded_file = st.file_uploader("PDF ou Markdown", type=["pdf", "md"])
    if uploaded_file:
        import tempfile, os
        from src.ingestion.loader import load_document, split_documents
        from src.ingestion.indexer import index_documents

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
        ) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with st.spinner("Indexation..."):
            docs = load_document(tmp_path)
            chunks = split_documents(docs)
            index_documents(chunks)
        st.success(f"✅ {len(chunks)} chunks indexés")
        os.unlink(tmp_path)

    url_input = st.text_input("URL d'une page web")
    if st.button("Indexer l'URL") and url_input:
        from src.ingestion.loader import load_document, split_documents
        from src.ingestion.indexer import index_documents
        with st.spinner("Indexation..."):
            docs = load_document(url_input)
            chunks = split_documents(docs)
            index_documents(chunks)
        st.success(f"✅ {len(chunks)} chunks indexés")

    st.divider()

    # Mémoire épisodique
    st.subheader("🧠 Mémoire épisodique")
    from src.agent.memory import load_episodic_memory
    episodes = load_episodic_memory()
    st.caption(f"{len(episodes)} exemple(s) mémorisé(s)")
    if episodes:
        for e in episodes:
            st.caption(f"• {e['question'][:50]}... (score: {e['quality_score']:.2f})")

# Zone principale
st.title("🤖 ResearchPal v2")
st.caption(
    "Powered by LangGraph + Corrective RAG | "
    "Outils : search_corpus + search_web | "
    "Mémoire : SQLite (CT) + épisodique (LT)"
)

# Afficher l'historique des messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🔧 Outil", meta.get("tool_used", "—"))
            col2.metric("🔄 Retries", meta.get("retry_count", 0))
            col3.metric("✅ Grade", meta.get("grade", "—"))
            col4.metric("⏱ Latence", f"{meta.get('latency_ms', 0):.0f} ms")

# Input utilisateur
if prompt := st.chat_input("Posez votre question..."):
    # Afficher la question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Exécuter le graphe agentique
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            start = time.perf_counter()
            from src.agent.graph import run_agent
            result = run_agent(
                question=prompt,
                thread_id=st.session_state.thread_id,
                graph=st.session_state.graph,
            )
            elapsed = (time.perf_counter() - start) * 1000

        answer = result["generation"]
        st.markdown(answer)

        # Métriques agentiques
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🔧 Outil", result.get("tool_used", "corpus"))
        col2.metric("🔄 Retries", result.get("retry_count", 0))
        col3.metric("✅ Grade", result.get("grade", "relevant"))
        col4.metric("⏱ Latence", f"{elapsed:.0f} ms")

        # Sources
        docs = result.get("documents", [])
        if docs:
            with st.expander("📎 Sources utilisées"):
                for i, doc in enumerate(docs, 1):
                    source = doc.metadata.get("filename", doc.metadata.get("source", "inconnu"))
                    st.caption(f"[{i}] {source}")
                    st.text(doc.page_content[:200] + "...")

        # Mémoriser si qualité suffisante
        from src.agent.memory import add_episodic_entry
        if result.get("retry_count", 0) == 0 and result.get("grade") == "relevant":
            add_episodic_entry(
                question=prompt,
                answer=answer,
                tool_used=result.get("tool_used", "corpus"),
                quality_score=0.9,
            )

    # Sauvegarder dans l'historique Streamlit
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "metadata": {
            "tool_used": result.get("tool_used", "corpus"),
            "retry_count": result.get("retry_count", 0),
            "grade": result.get("grade", "relevant"),
            "latency_ms": elapsed,
        },
    })
