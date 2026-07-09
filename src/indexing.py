import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    SparseVectorParams,
    VectorParams,
)
from typing import List
from langchain.schema import Document

from src.ingestion import run_chunking_pipeline

load_dotenv()

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "deepvk/USER-bge-m3")
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "normative_docs")
EMBEDDING_DIM = 1024  # размерность DeepVK/USER-bge-m3

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


def create_qdrant_collection(client: QdrantClient) -> None:
    """Создание коллекции с именованными dense + sparse векторами, если не существует."""
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF),
            },
        )
        print(f"Коллекция '{COLLECTION_NAME}' создана (dense + sparse BM25).")
    else:
        print(f"Коллекция '{COLLECTION_NAME}' уже существует.")


def index_documents(chunks: List[Document]) -> QdrantVectorStore:
    """Создание dense + sparse эмбеддингов и загрузка в Qdrant."""

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    create_qdrant_collection(client)

    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        sparse_embedding=get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name=DENSE_VECTOR_NAME,
        sparse_vector_name=SPARSE_VECTOR_NAME,
        collection_name=COLLECTION_NAME,
        url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
        batch_size=64,
    )

    print(f"Проиндексировано {len(chunks)} чанков.")
    return vector_store


def run_indexing_pipeline(processed_dir: str = "data/processed") -> QdrantVectorStore:
    """Фаза 2: чанкинг обработанных документов -> эмбеддинги -> загрузка в Qdrant."""
    chunks = run_chunking_pipeline(processed_dir)
    return index_documents(chunks)


if __name__ == "__main__":
    run_indexing_pipeline()
