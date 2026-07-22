"""Единая точка чтения конфигурации из переменных окружения.

Все модули берут настройки отсюда — env нигде больше не читается,
поэтому дефолты не расходятся между offline- и online-контурами.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Qdrant ---
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "normative_docs")

# --- Эмбеддинги ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "deepvk/USER-bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")

# известные размерности dense-векторов по имени модели; размерность нужна
# при создании коллекции Qdrant и должна соответствовать модели
_KNOWN_EMBEDDING_DIMS = {
    "deepvk/USER-bge-m3": 1024,
    "BAAI/bge-m3": 1024,
}


def _resolve_embedding_dim() -> int:
    """Размерность dense-эмбеддинга: из EMBEDDING_DIM или по имени модели.

    Для неизвестной модели без явного EMBEDDING_DIM падаем с понятной ошибкой —
    иначе коллекция создастся с неверной размерностью и upsert сломается позже.
    """
    env_dim = os.getenv("EMBEDDING_DIM")
    if env_dim:
        return int(env_dim)
    if EMBEDDING_MODEL_NAME in _KNOWN_EMBEDDING_DIMS:
        return _KNOWN_EMBEDDING_DIMS[EMBEDDING_MODEL_NAME]
    raise ValueError(
        f"Неизвестна размерность эмбеддинга для модели '{EMBEDDING_MODEL_NAME}'. "
        "Задайте переменную окружения EMBEDDING_DIM или добавьте модель в "
        "_KNOWN_EMBEDDING_DIMS (src/config.py)."
    )


EMBEDDING_DIM = _resolve_embedding_dim()

# --- Ollama (LLM) ---
# Нативный API Ollama (сервис на хосте, порт 11434, без суффикса /v1);
# в docker-compose для контейнеров — http://host.docker.internal:11434
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://localhost:11434"
).removesuffix("/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b")
# Контекстное окно: дефолтные 4096 Ollama не вмещают RAG-промпт (~2900 токенов)
# + thinking + ответ — сервер делал context shift посреди генерации (медленно,
# обрезался system prompt). Через /v1 num_ctx передать нельзя, поэтому ChatOllama.
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", 16384))

# --- Поиск ---
# число чанков, передаваемых в контекст LLM (общий дефолт retrieval/generation/api)
TOP_K = int(os.getenv("TOP_K", 10))
