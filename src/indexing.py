import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    Modifier,
    SparseVectorParams,
    VectorParams,
)
from typing import List, Tuple
from langchain.schema import Document

from src.chunking import split_documents
from src.config import COLLECTION_NAME, EMBEDDING_DIM, QDRANT_HOST, QDRANT_PORT
from src.manifest import MANIFEST_PATH, load_manifest, now_iso, save_manifest
from src.vectorstore import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME, get_vector_store

logger = logging.getLogger(__name__)

# фиксированное пространство имён для детерминированных ID точек Qdrant:
# uuid5(namespace, f"{source}::{chunk_index}") — повторный расчёт эмбеддингов
# для того же чанка того же источника даёт тот же ID, поэтому upsert идемпотентен
# (повторный запуск индексации не плодит дублей).
POINT_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "rag-prototype.normative-docs")


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
        logger.info("Коллекция '%s' создана (dense + sparse BM25).", COLLECTION_NAME)
    else:
        logger.info("Коллекция '%s' уже существует.", COLLECTION_NAME)


def _chunk_point_ids(source: str, n: int) -> List[str]:
    """Детерминированные ID чанков источника — обеспечивают идемпотентность upsert."""
    return [str(uuid.uuid5(POINT_ID_NAMESPACE, f"{source}::{i}")) for i in range(n)]


def _delete_points_by_source(client: QdrantClient, source: str) -> None:
    """Удаляет все точки коллекции, чей metadata.source == source (перед переиндексацией/удалением)."""
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))]
        ),
    )


@dataclass
class IndexResult:
    """Итог стадии index: какие источники переиндексированы/пропущены/удалены/провалились."""

    indexed: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    failed: List[Tuple[str, str]] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)


def run_indexing_pipeline(
    processed_dir: str = "data/processed",
    manifest_path: Path = MANIFEST_PATH,
    force: bool = False,
) -> IndexResult:
    """Фаза 2, инкрементально: переиндексирует в Qdrant только источники с изменившимся
    содержимым (по манифесту, заполняемому стадией ingest), удаляет точки источников,
    помеченных как удалённые, и пропускает неизменившиеся — без пересчёта эмбеддингов.

    `processed_dir` не используется напрямую (пути уже записаны в манифесте стадией
    ingest) — параметр сохранён для совместимости сигнатуры с предыдущей версией.
    """

    manifest = load_manifest(manifest_path)
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    create_qdrant_collection(client)
    vector_store = get_vector_store(client)

    result = IndexResult()

    # 1. удалённые исходники (raw_deleted проставлен стадией ingest)
    for source in [s for s, e in manifest.items() if e.get("raw_deleted")]:
        entry = manifest[source]
        chunk_source = entry.get("processed_path")
        try:
            if chunk_source:
                _delete_points_by_source(client, chunk_source)
                logger.info("Удалены точки Qdrant источника: %s", chunk_source)
            del manifest[source]
            result.removed.append(source)
        except Exception as exc:
            logger.error("Ошибка удаления точек для %s: %s", source, exc)
            result.failed.append((source, str(exc)))

    # 2. новые/изменившиеся исходники
    for source, entry in manifest.items():
        processed_path = entry.get("processed_path")
        processed_sha256 = entry.get("processed_sha256")
        if not processed_path or not Path(processed_path).exists():
            continue

        unchanged = (
            not force
            and entry.get("indexed_sha256") == processed_sha256
            and entry.get("point_ids")
        )
        if unchanged:
            result.skipped.append(source)
            continue

        try:
            text = Path(processed_path).read_text(encoding="utf-8")
            doc = Document(page_content=text, metadata={"source": processed_path})
            chunks = split_documents([doc])

            if entry.get("point_ids"):
                _delete_points_by_source(client, processed_path)

            ids = _chunk_point_ids(processed_path, len(chunks))
            if chunks:
                vector_store.add_documents(chunks, ids=ids)

            entry["indexed_sha256"] = processed_sha256
            entry["indexed_at"] = now_iso()
            entry["point_ids"] = ids
            result.indexed.append(source)
            logger.info("Проиндексирован источник: %s (%d чанков)", processed_path, len(chunks))
        except Exception as exc:
            logger.error("Ошибка индексации %s: %s", processed_path, exc)
            result.failed.append((source, str(exc)))

    save_manifest(manifest, manifest_path)
    logger.info(
        "Index: обновлено %d, без изменений %d, ошибок %d, удалено источников %d",
        len(result.indexed), len(result.skipped), len(result.failed), len(result.removed),
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_indexing_pipeline()
