from src.config import (
    LLM_PROVIDER, LLM_MODEL, LLM_TEMPERATURE, LLM_BASE_URL,
    EMBEDDING_PROVIDER, EMBEDDING_MODEL,
)

def get_llm():
    if LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            base_url=LLM_BASE_URL,
        )

def get_embeddings():
    if EMBEDDING_PROVIDER == "sentence_transformers":
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )