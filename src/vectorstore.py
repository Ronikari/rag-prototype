"""Общие фабрики эмбеддингов и vector store для offline- (indexing) и
online- (retrieval) контуров.

Вынесены в отдельный модуль, чтобы API-сервер не тянул offline-зависимости
ingestion (fitz, pytesseract, python-docx) через импорт src.indexing.
"""

from typing import Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient

from src.config import (
    COLLECTION_NAME,
    EMBEDDING_DEVICE,
    EMBEDDING_MODEL_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
)

# гибридный поиск: именованные dense + sparse (BM25) векторы в одной коллекции
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
SPARSE_MODEL_NAME = "Qdrant/bm25"
BM25_LANGUAGE = "russian"


def get_embeddings() -> HuggingFaceEmbeddings:
    """Инициализация embedding-модели."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={
            "device": EMBEDDING_DEVICE,
            "trust_remote_code": True,
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
        },
    )


def get_sparse_embeddings() -> FastEmbedSparse:
    """Инициализация sparse-модели BM25 (fastembed) для гибридного поиска."""
    return FastEmbedSparse(model_name=SPARSE_MODEL_NAME, language=BM25_LANGUAGE)


def get_vector_store(client: Optional[QdrantClient] = None) -> QdrantVectorStore:
    """Подключение к коллекции Qdrant в режиме гибридного поиска.

    Dense-эмбеддинг (семантика) и sparse BM25 (точное совпадение термов)
    выполняются параллельно, результаты сливаются на стороне Qdrant через
    Reciprocal Rank Fusion (RRF).
    """
    return QdrantVectorStore(
        client=client or QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT),
        collection_name=COLLECTION_NAME,
        embedding=get_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )
