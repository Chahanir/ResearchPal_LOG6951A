from pathlib import Path

# Chemins
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data" / "sample_docs"
CHROMA_DIR = ROOT_DIR / "chroma_db"
CHROMA_COLLECTION_NAME = "researchpal"

# Embeddings
EMBEDDING_PROVIDER = "sentence_transformers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# LLM
LLM_PROVIDER = "ollama"
LLM_MODEL = "llama3.2"
LLM_TEMPERATURE = 0.1
LLM_BASE_URL = "http://localhost:11434"

# Chunking
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retrieval
RETRIEVAL_K = 4
MMR_LAMBDA = 0.5

# Historique
MAX_HISTORY_TURNS = 10

# Optimisation de requête
MULTI_QUERY_N = 3