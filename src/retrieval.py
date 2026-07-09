import os
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from typing import List, Tuple
from langchain.schema import Document

from src.indexing import (
    DENSE_VECTOR_NAME,
    SPARSE_VECTOR_NAME,
    get_embeddings,
    get_sparse_embeddings,
)

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "normative_docs")


def get_vector_store() -> QdrantVectorStore:
    """Подключение к существующей коллекции Qdrant в режиме гибридного поиска.

    Dense-эмбеддинг (семантика) и sparse BM25 (точное совпадение термов)
    выполняются параллельно, результаты сливаются на стороне Qdrant через
    Reciprocal Rank Fusion (RRF).
    """
    return QdrantVectorStore(
        client=QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT),
        collection_name=COLLECTION_NAME,
        embedding=get_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
    )


def get_retriever(top_k: int = 10):
    """Возвращает LangChain ретривер с настройками поиска"""

    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    return retriever


def search_documents(
    query: str,
    top_k: int = 10,
) -> List[Tuple[Document, float]]:
    """Выполнить гибридный поиск с оценками релевантности (RRF-score)."""

    vector_store = get_vector_store()

    results = vector_store.similarity_search_with_score(
        query=query,
        k=top_k,
    )

    return results
