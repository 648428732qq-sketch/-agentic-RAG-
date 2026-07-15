import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent
BASE_DIR = PROJECT_DIR.parent
load_dotenv(PROJECT_DIR / ".env", override=False)


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_path(name: str, default: Path) -> str:
    raw = os.environ.get(name, "").strip()
    path = Path(raw).expanduser() if raw else default
    if not path.is_absolute():
        path = BASE_DIR / path
    return str(path.resolve())

# ============================================================
#  中医医院智能知识库 — 配置中心
# ============================================================

# --- 目录配置（可用环境变量覆盖，支持 Linux 挂载目录）---
MARKDOWN_DIR = _env_path("MARKDOWN_DIR", BASE_DIR / "markdown_docs")
PARENT_STORE_PATH = _env_path("PARENT_STORE_PATH", BASE_DIR / "parent_store")
QDRANT_DB_PATH = _env_path("QDRANT_DB_PATH", BASE_DIR / "qdrant_db")

# --- Qdrant 向量库配置 ---
CHILD_COLLECTION = "tcm_child_chunks"          # 中医子块集合
SYNDROME_COLLECTION = os.environ.get("SYNDROME_COLLECTION", "tcm_syndrome_entries")
SPARSE_VECTOR_NAME = "sparse"
QDRANT_URL = os.environ.get("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "").strip()
QDRANT_PREFER_GRPC = _env_bool("QDRANT_PREFER_GRPC", False)

# --- 模型配置 ---
# 中文优化 Embedding: BAAI/bge-large-zh-v1.5 (1024维, 512token上下文)
# 备选: BAAI/bge-small-zh-v1.5 (512维, 更快), shibing624/text2vec-base-chinese (768维)
DENSE_MODEL = os.environ.get("DENSE_MODEL", "BAAI/bge-small-zh-v1.5").strip()
EMBEDDING_LOCAL_FILES_ONLY = _env_bool("EMBEDDING_LOCAL_FILES_ONLY", True)
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "").strip().lower()
SPARSE_MODEL = "Qdrant/bm25"
LLM_MODEL = "qwen3:4b-instruct-2507-q4_K_M"   # 本地 Ollama 模型, 支持中文
LLM_TEMPERATURE = 0

# --- Structured syndrome retrieval / Query Translator ---
STRICT_LOCAL_EVIDENCE_MODE = _env_bool("STRICT_LOCAL_EVIDENCE_MODE", True)
ENABLE_HYBRID_SYNDROME_RETRIEVAL = _env_bool("ENABLE_HYBRID_SYNDROME_RETRIEVAL", True)
SYNDROME_MIN_QUERY_COVERAGE = _env_float("SYNDROME_MIN_QUERY_COVERAGE", 1.0)
SYNDROME_MIN_EVIDENCE_CONFIDENCE = _env_float("SYNDROME_MIN_EVIDENCE_CONFIDENCE", 0.6)
SYNDROME_PAYLOAD_CACHE_TTL_SECONDS = _env_int("SYNDROME_PAYLOAD_CACHE_TTL_SECONDS", 300)
SYNDROME_RAW_DENSE_CANDIDATES = _env_int("SYNDROME_RAW_DENSE_CANDIDATES", 40)
SYNDROME_LEXICAL_CANDIDATES = _env_int("SYNDROME_LEXICAL_CANDIDATES", 40)
SYNDROME_CANONICAL_DENSE_CANDIDATES = _env_int("SYNDROME_CANONICAL_DENSE_CANDIDATES", 40)
SYNDROME_RRF_K = _env_int("SYNDROME_RRF_K", 60)
SYNDROME_TRANSLATOR_CONTEXT_CANDIDATES = _env_int("SYNDROME_TRANSLATOR_CONTEXT_CANDIDATES", 16)
SYNDROME_TRANSLATOR_MAX_TERMS = _env_int("SYNDROME_TRANSLATOR_MAX_TERMS", 120)

# Offline-trained Query Translator bi-encoder. It only proposes terms from the
# frozen local catalog; payload filtering and the evidence gate remain authoritative.
ENABLE_QUERY_TRANSLATOR_BIENCODER = _env_bool("ENABLE_QUERY_TRANSLATOR_BIENCODER", False)
QUERY_TRANSLATOR_BIENCODER_MODEL = _env_path(
    "QUERY_TRANSLATOR_BIENCODER_MODEL",
    BASE_DIR / "artifacts" / "query_translator" / "20260630_biencoder_v3" / "model",
)
QUERY_TRANSLATOR_BIENCODER_CATALOG = _env_path(
    "QUERY_TRANSLATOR_BIENCODER_CATALOG",
    BASE_DIR / "artifacts" / "query_translator" / "20260630_biencoder_v3" / "term_catalog.jsonl",
)
QUERY_TRANSLATOR_BIENCODER_DEVICE = os.environ.get("QUERY_TRANSLATOR_BIENCODER_DEVICE", "cpu").strip().lower()
QUERY_TRANSLATOR_BIENCODER_TOP_K = _env_int("QUERY_TRANSLATOR_BIENCODER_TOP_K", 5)
QUERY_TRANSLATOR_BIENCODER_PRIMARY_K = _env_int("QUERY_TRANSLATOR_BIENCODER_PRIMARY_K", 3)
QUERY_TRANSLATOR_BIENCODER_MIN_SCORE = _env_float("QUERY_TRANSLATOR_BIENCODER_MIN_SCORE", 0.55)
QUERY_TRANSLATOR_BIENCODER_LOCAL_TERM_THRESHOLD = _env_int(
    "QUERY_TRANSLATOR_BIENCODER_LOCAL_TERM_THRESHOLD", 3
)

ENABLE_SYNDROME_RERANK = _env_bool("ENABLE_SYNDROME_RERANK", False)
SYNDROME_RERANK_MODEL = os.environ.get("SYNDROME_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
SYNDROME_RERANK_CANDIDATES = _env_int("SYNDROME_RERANK_CANDIDATES", 40)
SYNDROME_RERANK_INTENTS = os.environ.get("SYNDROME_RERANK_INTENTS", "clinical_symptom")
SYNDROME_RERANK_MODE = os.environ.get("SYNDROME_RERANK_MODE", "evidence_first").strip().lower()
SYNDROME_RERANK_LOCAL_FILES_ONLY = _env_bool("SYNDROME_RERANK_LOCAL_FILES_ONLY", EMBEDDING_LOCAL_FILES_ONLY)
SYNDROME_RERANK_MAX_LENGTH = _env_int("SYNDROME_RERANK_MAX_LENGTH", 512)
SYNDROME_RERANK_BATCH_SIZE = _env_int("SYNDROME_RERANK_BATCH_SIZE", 16)
SYNDROME_RERANK_DEVICE = os.environ.get("SYNDROME_RERANK_DEVICE", "").strip().lower()
SYNDROME_RERANK_CACHE_FOLDER = os.environ.get("SYNDROME_RERANK_CACHE_FOLDER", "").strip()
SYNDROME_RERANK_TRUST_REMOTE_CODE = _env_bool("SYNDROME_RERANK_TRUST_REMOTE_CODE", False)

ENABLE_LLM_SYMPTOM_TRANSLATOR = _env_bool("ENABLE_LLM_SYMPTOM_TRANSLATOR", True)
LLM_SYMPTOM_TRANSLATOR_ALWAYS = _env_bool("LLM_SYMPTOM_TRANSLATOR_ALWAYS", False)
LLM_SYMPTOM_TRANSLATOR_MIN_LOCAL_TERMS = _env_int("LLM_SYMPTOM_TRANSLATOR_MIN_LOCAL_TERMS", 3)
LLM_SYMPTOM_TRANSLATOR_TIMEOUT_SECONDS = _env_float("LLM_SYMPTOM_TRANSLATOR_TIMEOUT_SECONDS", 12)
LLM_SYMPTOM_TRANSLATOR_CONNECT_TIMEOUT_SECONDS = _env_float(
    "LLM_SYMPTOM_TRANSLATOR_CONNECT_TIMEOUT_SECONDS", 4
)
LLM_SYMPTOM_TRANSLATOR_MAX_RETRIES = _env_int("LLM_SYMPTOM_TRANSLATOR_MAX_RETRIES", 1)
LLM_SYMPTOM_TRANSLATOR_CACHE_SIZE = _env_int("LLM_SYMPTOM_TRANSLATOR_CACHE_SIZE", 128)
LLM_SYMPTOM_TRANSLATOR_FAILURE_THRESHOLD = _env_int("LLM_SYMPTOM_TRANSLATOR_FAILURE_THRESHOLD", 2)
LLM_SYMPTOM_TRANSLATOR_COOLDOWN_SECONDS = _env_int("LLM_SYMPTOM_TRANSLATOR_COOLDOWN_SECONDS", 120)
LLM_SYMPTOM_TRANSLATOR_MIN_MAPPING_CONFIDENCE = _env_float("LLM_SYMPTOM_TRANSLATOR_MIN_MAPPING_CONFIDENCE", 0.72)

# --- 多提供商配置 (可选) ---
LLM_CONFIGS = {
    "ollama": {
        "model": "qwen3:4b-instruct-2507-q4_K_M",
        "url": "http://localhost:11434",
        "temperature": 0
    },
    "openai": {
        "model": "gpt-4o-mini",
        "temperature": 0
    },
    "deepseek": {
        "model": "deepseek-chat",
        "temperature": 0
    },
    "anthropic": {
        "model": "claude-sonnet-4-5-20250929",
        "temperature": 0
    },
    "google": {
        "model": "gemini-2.5-flash",
        "temperature": 0
    }
}
ACTIVE_LLM_CONFIG = os.environ.get("ACTIVE_LLM_CONFIG", "deepseek").strip().lower()

# --- Agent 配置 ---
MAX_TOOL_CALLS = 8            # 单 Agent 最大工具调用次数
MAX_ITERATIONS = 10           # Agent 最大循环迭代次数
GRAPH_RECURSION_LIMIT = 50    # 图递归上限
BASE_TOKEN_THRESHOLD = 2000   # 初始上下文压缩阈值
TOKEN_GROWTH_FACTOR = 0.9     # 每次压缩后阈值乘数

# --- 文本分块配置 (适配中文) ---
# 中文1个token约1.5个汉字, 以下用字符数估算
CHILD_CHUNK_SIZE = 400        # 子块大小 (检索用, 中文场景约250-400字)
CHILD_CHUNK_OVERLAP = 80      # 子块重叠 (避免句子截断)
MIN_PARENT_SIZE = 1500        # 最小父块大小
MAX_PARENT_SIZE = 3500        # 最大父块大小
HEADERS_TO_SPLIT_ON = [
    ("#", "H1"),
    ("##", "H2"),
    ("###", "H3")
]

# --- Langfuse 可观测性 ---
LANGFUSE_ENABLED = os.environ.get("LANGFUSE_ENABLED", "false").lower() == "true"
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_BASE_URL = os.environ.get("LANGFUSE_BASE_URL", "http://localhost:3000")

# --- WebUI 服务配置 ---
GRADIO_SERVER_NAME = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1").strip()
GRADIO_SERVER_PORT = _env_int("GRADIO_SERVER_PORT", 7860)
GRADIO_SHARE = _env_bool("GRADIO_SHARE", False)
