"""Нарезка очищенных документов на чанки по структуре нормативных текстов.

Вынесено из ingestion.py в отдельный модуль: логически чанкинг — часть
предобработки текста, но исполняется он на стадии индексации (indexing.py),
непосредственно перед векторизацией, а не на стадии ingest. Такое разделение
позволяет менять параметры нарезки и переиндексировать (`index --force`) без
повторного парсинга и OCR исходников; при этом реализация остаётся единственной
(один источник истины), а indexing.py её переиспользует.
"""

import logging
import re
from typing import List, Optional, Tuple

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# параметры чанкинга
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
CHUNK_SEPARATORS = [
    "\n\n",          # параграф
    "\n",            # перенос строки
    ". ",            # конец предложения
    " ",             # слово
    "",              # символ
]

# максимальная длина строки-заголовка, дублируемой в чанки-продолжения
CONTEXT_HEADER_MAX_LEN = 200

# начало статьи закона
ARTICLE_RE = re.compile(r"^Статья\s+(\d+(?:\.\d+)*)\.?\s")
# начало пункта правил/приказа: "10. ...", "9(1). ...", "18(2). ...", "10.1. ..."
POINT_RE = re.compile(r"^(\d+(?:\.\d+)*(?:\(\d+\))?)\.\s")

# строка markdown-таблицы (в этот формат сериализуются таблицы PDF/DOCX)
TABLE_LINE_RE = re.compile(r"^\s*\|")


def split_into_structural_units(text: str) -> List[Tuple[str, Optional[str], str]]:
    """Разбивает текст нормативного документа на структурные единицы (статьи и пункты).

    Возвращает список кортежей (section, article_heading, unit_text):
    - section — метка для цитирования: "статья 26", "пункт 10", "статья 26, пункт 2"
      или "" для текста вне структуры (преамбула, приложения без нумерации);
    - article_heading — строка-заголовок текущей статьи (для дублирования в чанки);
    - unit_text — текст единицы, начинающийся со строки статьи/пункта.
    """
    units: List[Tuple[str, Optional[str], str]] = []
    current_lines: List[str] = []
    current_article: Optional[str] = None       # номер статьи
    article_heading: Optional[str] = None       # полная строка "Статья N. Название"
    current_point: Optional[str] = None         # номер пункта

    def make_label() -> str:
        parts = []
        if current_article:
            parts.append(f"статья {current_article}")
        if current_point:
            parts.append(f"пункт {current_point}")
        return ", ".join(parts)

    def flush() -> None:
        unit_text = "\n".join(current_lines).strip()
        if unit_text:
            units.append((make_label(), article_heading, unit_text))
        current_lines.clear()

    for line in text.split("\n"):
        stripped = line.lstrip()
        article_match = ARTICLE_RE.match(stripped)
        point_match = None if article_match else POINT_RE.match(stripped)
        if article_match or point_match:
            flush()
            if article_match:
                current_article = article_match.group(1)
                article_heading = stripped[:CONTEXT_HEADER_MAX_LEN]
                current_point = None
            else:
                current_point = point_match.group(1)
        current_lines.append(line)
    flush()

    return units


def _unit_header(unit_text: str) -> str:
    """Первая строка структурной единицы"""
    first_line = unit_text.split("\n", 1)[0].strip()
    return first_line[:CONTEXT_HEADER_MAX_LEN]


def _split_text_and_tables(text: str) -> List[Tuple[str, bool]]:
    """Разбивает текст на чередующиеся блоки (block_text, is_table).

    Таблицей считается непрерывная последовательность markdown-строк "| ... |".
    """

    blocks: List[Tuple[str, bool]] = []
    current: List[str] = []
    current_is_table = False
    for line in text.split("\n"):
        is_table = bool(TABLE_LINE_RE.match(line))
        if current and is_table != current_is_table:
            blocks.append(("\n".join(current), current_is_table))
            current = []
        current_is_table = is_table
        current.append(line)
    if current:
        blocks.append(("\n".join(current), current_is_table))
    return blocks


def _split_table(table_text: str) -> List[str]:
    """Режет длинную markdown-таблицу на части по границам строк.

    Шапка (первая строка + разделитель) дублируется в каждую часть — иначе
    продолжение таблицы теряет названия колонок и значения ячеек становятся
    бессмысленными и для эмбеддинга, и для LLM.
    """

    if len(table_text) <= CHUNK_SIZE:
        return [table_text]

    lines = table_text.split("\n")
    header, body = lines[:2], lines[2:]
    parts: List[str] = []
    current = list(header)
    size = sum(len(line) + 1 for line in current)
    for line in body:
        if size + len(line) + 1 > CHUNK_SIZE and len(current) > len(header):
            parts.append("\n".join(current))
            current = list(header)
            size = sum(len(l) + 1 for l in current)
        current.append(line)
        size += len(line) + 1
    if len(current) > len(header):
        parts.append("\n".join(current))
    return parts


def split_documents(documents: List[Document]) -> List[Document]:
    """Разбивка документов на чанки по структуре нормативных текстов.

    Текст сначала режется по границам статей и пунктов, затем длинные единицы
    дробятся сплиттером. Номер статьи/пункта записывается в метаданные чанка
    (ключ "section"), а в чанки-продолжения дублируется строка-заголовок единицы —
    иначе хвост длинного пункта теряет свой номер и вводную фразу, из-за чего
    не находится поиском и не может быть корректно процитирован моделью.

    Markdown-таблицы дробятся отдельно от текста (_split_table): по границам
    строк таблицы и с дублированием шапки; чанк с таблицей получает метку
    content_type="table" в метаданных.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )

    chunks: List[Document] = []
    for doc in documents:
        for section, article_heading, unit_text in split_into_structural_units(
            doc.page_content
        ):
            parts: List[Tuple[str, bool]] = []
            for block, is_table in _split_text_and_tables(unit_text):
                if is_table:
                    parts.extend((p, True) for p in _split_table(block))
                elif block.strip():
                    parts.extend((p, False) for p in splitter.split_text(block))
            unit_header = _unit_header(unit_text)
            for j, (part, is_table) in enumerate(parts):
                header_lines = []
                # статья указана в метке, но её заголовок в тексте пункта не виден
                if article_heading and "пункт" in section:
                    header_lines.append(article_heading)
                # чанк-продолжение потерял первую строку своей единицы
                if j > 0 and section:
                    header_lines.append(unit_header)
                content = "\n".join(header_lines + [part]) if header_lines else part
                metadata = {**doc.metadata, "section": section}
                if is_table:
                    metadata["content_type"] = "table"
                chunks.append(Document(page_content=content, metadata=metadata))

    # добавляем порядковый номер чанка в метаданные для отладки
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)

    if chunks:
        avg_size = sum(len(c.page_content) for c in chunks) / len(chunks)
        logger.info("Создано чанков: %d, средний размер: %.0f символов", len(chunks), avg_size)
    return chunks
