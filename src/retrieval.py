from typing import List, Tuple
from langchain.schema import Document

from src.config import TOP_K
from src.vectorstore import get_vector_store


def get_retriever(top_k: int = TOP_K):
    """Возвращает LangChain ретривер с настройками поиска"""

    vector_store = get_vector_store()

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    return retriever


def search_documents(
    query: str,
    top_k: int = TOP_K,
) -> List[Tuple[Document, float]]:
    """Выполнить гибридный поиск с оценками релевантности (RRF-score)."""

    vector_store = get_vector_store()

    results = vector_store.similarity_search_with_score(
        query=query,
        k=top_k,
    )

    return results
