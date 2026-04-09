import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.generation.rag_pipeline import RAGPipeline
from src.ingestion.indexer import ingest_source, get_collection_stats


# Configuration de la page
st.set_page_config(
    page_title="ResearchPal",
    page_icon="🔬",
    layout="wide",
)

# Initialisation du session state 
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(retrieval_strategy="mmr", k=4)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources_history" not in st.session_state:
    st.session_state.sources_history = []

if "variants_history" not in st.session_state:
    st.session_state.variants_history = []


col_chat, col_sidebar = st.columns([3, 1])


# chat
with col_chat:
    st.title("🔬 ResearchPal v1")
    st.caption("Assistant de recherche RAG — LOG6951A")

    # Affichage de l'historique
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                turn_idx = i // 2
                # Sources
                if turn_idx < len(st.session_state.sources_history):
                    sources = st.session_state.sources_history[turn_idx]
                    if sources:
                        with st.expander("📎 Sources utilisées"):
                            st.markdown(sources)
                # Variantes multi-query (T4)
                if turn_idx < len(st.session_state.variants_history):
                    variants = st.session_state.variants_history[turn_idx]
                    if variants:
                        with st.expander("🔍 Variantes de requête générées (multi-query)"):
                            for j, v in enumerate(variants, 1):
                                st.markdown(f"{j}. {v}")

    # Champ de saisie
    if question := st.chat_input("Posez une question sur vos documents..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Génération de la réponse
        with st.chat_message("assistant"):
            with st.spinner("Recherche en cours..."):
                result = st.session_state.pipeline.ask(question)

            st.markdown(result["answer"])

            # Sources
            if result["sources"]:
                with st.expander("📎 Sources utilisées"):
                    st.markdown(result["sources"])

            # Variantes multi-query (T4) (affichées seulement si activé)
            if result.get("variants"):
                with st.expander("🔍 Variantes de requête générées (multi-query)"):
                    for j, v in enumerate(result["variants"], 1):
                        st.markdown(f"{j}. {v}")

        # Sauvegarde dans l'historique
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"]
        })
        st.session_state.sources_history.append(result["sources"])
        st.session_state.variants_history.append(result.get("variants", []))


# sidebar
with col_sidebar:
    st.header("📥 Ajouter un document")

    # Upload de fichier
    uploaded_file = st.file_uploader(
        "Fichier PDF ou Markdown",
        type=["pdf", "md"],
        help="Glissez un fichier PDF ou Markdown pour l'indexer"
    )

    if uploaded_file is not None:
        temp_path = f"data/sample_docs/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.button("📥 Indexer ce document", type="primary"):
            with st.spinner(f"Indexation de {uploaded_file.name}..."):
                try:
                    ingest_source(temp_path)
                    st.success(f"✅ {uploaded_file.name} indexé !")
                except Exception as e:
                    st.error(f"❌ Erreur : {e}")

    st.divider()

    # Ajout par URL
    st.subheader("🌐 Ajouter une URL")
    url_input = st.text_input("URL", placeholder="https://example.com/article")
    if st.button("📥 Indexer cette URL"):
        if url_input.strip():
            with st.spinner(f"Indexation de {url_input}..."):
                try:
                    ingest_source(url_input.strip())
                    st.success("✅ URL indexée !")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Erreur : {e}")
        else:
            st.warning("Entrez une URL valide.")

    st.divider()

    # Statistiques ChromaDB
    st.subheader("📊 Corpus")
    try:
        stats = get_collection_stats()
        st.metric("Chunks indexés", stats["total_chunks"])
    except Exception:
        st.metric("Chunks indexés", "—")

    st.divider()

    # Contrôles
    st.subheader("⚙️ Contrôles")

    strategy = st.selectbox(
        "Stratégie de retrieval",
        ["mmr", "cosine"],
        help="MMR : diversité | Cosinus : pertinence pure"
    )
    if strategy != st.session_state.pipeline.retrieval_strategy:
        st.session_state.pipeline.retrieval_strategy = strategy
        st.success(f"Stratégie changée : {strategy}")

    # ── Optimisation de requête (T4) ──────────────────────────────────────────
    use_optim = st.checkbox(
        "🔍 Multi-query (T4)",
        value=st.session_state.pipeline.use_query_optimization,
        help="Génère 3 variantes de la requête via le LLM, puis fusionne les résultats par RRF"
    )
    if use_optim != st.session_state.pipeline.use_query_optimization:
        st.session_state.pipeline.use_query_optimization = use_optim
        st.success(f"Multi-query {'activé' if use_optim else 'désactivé'}")

    if st.button("🗑️ Effacer la conversation"):
        st.session_state.messages = []
        st.session_state.sources_history = []
        st.session_state.variants_history = []
        st.session_state.pipeline.clear_history()
        st.rerun()