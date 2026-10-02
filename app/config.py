import os
from pathlib import Path
# Database
DATABASE_URL = os.environ.get(
"DATABASE_URL",
"postgresql://hr_app:local_development_only@localhost:5432/hr",
)
POLICY_INDEX = "hr_policy_chunks"
# Documents
DATA_DIR = Path(__file__).parent.parent / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
# Chunking
CHUNK_SIZE = 700
CHUNK_OVERLAP = 80
# Embeddings
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_SIZE = 384
# Retrieval
RESULTS_PER_SEARCH = 12
MIN_BEST_SIMILARITY = float(os.environ.get("MIN_BEST_SIMILARITY", "0.595"))
MAX_GAP_FROM_BEST = float(os.environ.get("MAX_GAP_FROM_BEST", "0.10"))
# Answering
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5")
MAX_ANSWER_TOKENS = 700
# Conversation history
MAX_HISTORY_TURNS = 6
# Database fields stored with each chunk
FIELDS = ["filename", "title", "document_id", "status", "section", "content"]
CARRIED_FIELDS = ["title", "document_id", "status", "category", "region", "version",
"effective_date", "owner"]