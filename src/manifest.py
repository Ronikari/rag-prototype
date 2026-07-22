"""Манифест изменений исходных документов: хэши файлов и состояние стадий ingest/index.

Ключ манифеста — путь исходного файла в data/raw (то же значение, что попадает
в metadata["source"] на стадии ingestion, до слияния постраничных документов).
Manifest — единственный источник истины о том, какие файлы обработаны/проиндексированы
и с каким содержимым, что даёт идемпотентность и инкрементальность обеим стадиям
без внешней инфраструктуры (очереди, брокера сообщений и т.п.).
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

MANIFEST_PATH = Path("data/manifest.json")


def sha256_file(path: Path) -> str:
    """Хэш содержимого файла — основа для обнаружения created/updated."""
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_manifest(path: Path = MANIFEST_PATH) -> Dict[str, dict]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(manifest: Dict[str, dict], path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
