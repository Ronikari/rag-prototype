import os
import re
from pathlib import Path
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import Dict, Iterable, List, Optional, Tuple, Union

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from docx import Document as DocxFile
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph

# среднее число символов на страницу, ниже которого PDF считается сканом без текстового слоя
OCR_MIN_CHARS_PER_PAGE = 20

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

# минимальная доля площади текстового блока внутри таблицы,
# при которой блок считается частью таблицы и исключается из обычного текста
TABLE_BLOCK_OVERLAP = 0.5


def _table_rows_to_markdown(rows: List[List[Optional[str]]]) -> str:
    """Сериализует таблицу (список строк с ячейками) в markdown.

    Объединённые ячейки DOCX/PDF дублируют содержимое — повторы соседних
    ячеек в строке заменяются на пустые. Возвращает "" для вырожденных
    таблиц (меньше двух строк или одного столбца) — такие артефакты
    детектора лучше оставить обычным текстом.
    """

    def fmt(cell: Optional[str]) -> str:
        if not cell:
            return ""
        return re.sub(r"\s+", " ", str(cell)).replace("|", "\\|").strip()

    clean_rows = []
    for row in rows:
        cells = [fmt(c) for c in row]
        if not any(cells):
            continue
        # повтор соседней ячейки слева — след горизонтального объединения
        deduped = [cells[0]]
        for i in range(1, len(cells)):
            deduped.append("" if cells[i] and cells[i] == cells[i - 1] else cells[i])
        clean_rows.append(deduped)

    if len(clean_rows) < 2:
        return ""
    n_cols = max(len(r) for r in clean_rows)
    if n_cols < 2:
        return ""

    lines = []
    for row in clean_rows:
        row = row + [""] * (n_cols - len(row))
        lines.append("| " + " | ".join(row) + " |")
    lines.insert(1, "| " + " | ".join(["---"] * n_cols) + " |")
    return "\n".join(lines)


def ocr_pdf(path: str, dpi: int = 300, lang: str = "rus") -> List[Document]:
    """Извлекает текст из отсканированного PDF через рендер страниц в изображение + Tesseract OCR."""

    pdf = fitz.open(path)
    documents = []
    for page_number in range(pdf.page_count):
        pix = pdf[page_number].get_pixmap(dpi=dpi)
        mode = "RGB" if pix.n < 4 else "RGBA"
        image = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
        text = pytesseract.image_to_string(image, lang=lang)
        documents.append(
            Document(page_content=text, metadata={"source": path, "page": page_number})
        )
    return documents


def _ocr_scanned_pdfs(
    documents: List[Document], ocr_lang: str = "rus", ocr_dpi: int = 300
) -> List[Document]:
    """Заменяет документы PDF без текстового слоя (сканы) на результат OCR."""

    by_source: Dict[str, List[Document]] = {}
    for doc in documents:
        by_source.setdefault(doc.metadata.get("source", ""), []).append(doc)

    result = []
    for source, docs in by_source.items():
        avg_chars = sum(len(d.page_content.strip()) for d in docs) / len(docs)
        if source.endswith(".pdf") and avg_chars < OCR_MIN_CHARS_PER_PAGE:
            print(f"OCR: {source} (текстовый слой отсутствует, {len(docs)} стр.)")
            result.extend(ocr_pdf(source, dpi=ocr_dpi, lang=ocr_lang))
        else:
            result.extend(docs)
    return result


def _pdf_page_content(page: "fitz.Page") -> str:
    """Извлекает текст страницы PDF, сериализуя таблицы в markdown.

    Таблицы находятся детектором PyMuPDF; текстовые блоки, лежащие внутри
    рамки таблицы, исключаются из обычного текста (иначе содержимое ячеек
    дублировалось бы построчной выгрузкой). Блоки и таблицы объединяются
    в порядке следования на странице (по вертикальной координате).
    """

    table_items: List[Tuple[float, str]] = []
    table_rects: List[fitz.Rect] = []
    for table in page.find_tables().tables:
        markdown = _table_rows_to_markdown(table.extract())
        if markdown:
            rect = fitz.Rect(table.bbox)
            table_items.append((rect.y0, markdown))
            table_rects.append(rect)

    if not table_items:
        return page.get_text()

    text_items: List[Tuple[float, str]] = []
    for x0, y0, x1, y1, text, _, block_type in page.get_text("blocks"):
        if block_type != 0 or not text.strip():
            continue
        rect = fitz.Rect(x0, y0, x1, y1)
        in_table = any(
            (rect & table_rect).get_area() / max(rect.get_area(), 1.0)
            > TABLE_BLOCK_OVERLAP
            for table_rect in table_rects
        )
        if not in_table:
            text_items.append((y0, text.strip()))

    items = sorted(text_items + table_items, key=lambda item: item[0])
    return "\n\n".join(content for _, content in items)


def load_pdf(path: str) -> List[Document]:
    """Загружает PDF постранично (PyMuPDF) с таблицами в виде markdown."""

    pdf = fitz.open(path)
    return [
        Document(
            page_content=_pdf_page_content(pdf[page_number]),
            metadata={"source": path, "page": page_number},
        )
        for page_number in range(pdf.page_count)
    ]


def _iter_docx_blocks(docx: "DocxFile") -> Iterable[Union[DocxParagraph, DocxTable]]:
    """Итерирует параграфы и таблицы DOCX в порядке следования в документе."""

    for child in docx.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield DocxParagraph(child, docx)
        elif isinstance(child, CT_Tbl):
            yield DocxTable(child, docx)


def load_docx(path: str) -> List[Document]:
    """Загружает DOCX (python-docx): текст параграфов + таблицы в markdown.

    В отличие от docx2txt сохраняет структуру таблиц — ячейки не
    рассыпаются в плоский текст.
    """

    docx = DocxFile(path)
    blocks = []
    for block in _iter_docx_blocks(docx):
        if isinstance(block, DocxParagraph):
            if block.text.strip():
                blocks.append(block.text)
        else:
            markdown = _table_rows_to_markdown(
                [[cell.text for cell in row.cells] for row in block.rows]
            )
            if markdown:
                blocks.append(markdown)
    return [Document(page_content="\n\n".join(blocks), metadata={"source": path})]


def load_documents(data_dir: str = "data/raw") -> List[Document]:
    """Загружает PDF, TXT и DOCX документы из директории.

    PDF и DOCX загружаются с сохранением таблиц (markdown);
    PDF без текстового слоя (сканы) автоматически распознаются через OCR (Tesseract).
    """

    txt_loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=lambda path: TextLoader(path, encoding="utf-8"),
        show_progress=True,
    )

    pdf_documents = []
    for path in sorted(Path(data_dir).rglob("*.pdf")):
        pdf_documents.extend(load_pdf(str(path)))
    pdf_documents = _ocr_scanned_pdfs(pdf_documents)

    documents = []
    documents.extend(pdf_documents)
    documents.extend(txt_loader.load())
    for path in sorted(Path(data_dir).rglob("*.docx")):
        documents.extend(load_docx(str(path)))

    print(f"Загружено документов: {len(documents)}")
    return documents


def clean_text(text: str) -> str:
    """Очистка текста для русскоязычных нормативных документов.

    Помимо общих артефактов PDF-экстракции убирает колонтитулы КонсультантПлюс
    (шапка "Документ предоставлен...", подвал "надежная правовая поддержка",
    номера страниц, ссылки на www.consultant.ru) — типичные для документов,
    выгруженных с сайта КонсультантПлюс в PDF/DOCX.
    """

    # колонтитулы КонсультантПлюс
    text = re.sub(r'Документ предоставлен КонсультантПлюс\s*\n?', '', text)
    text = re.sub(r'Дата сохранения:\s*\d{2}\.\d{2}\.\d{4}\s*\n?', '', text)
    text = re.sub(r'www\.consultant\.ru\s*\n?', '', text)
    text = re.sub(r'КонсультантПлюс\s*\n?\s*надежная правовая поддержка\s*\n?', '', text)
    text = re.sub(r'Напечатано с сайта[^\n]*\n?', '', text)
    text = re.sub(r'Страница\s+\d+\s+из\s+\d+\s*\n?', '', text)

    # убираем дублирующиеся пробелы и переносы
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)

    # КонсультантПлюс дублирует титульный блок документа в начале файла — убираем повторы абзацев
    paragraphs = text.split('\n\n')
    deduped = []
    for p in paragraphs:
        if deduped and p.strip() and deduped[-1].strip() == p.strip():
            continue
        deduped.append(p)
    text = '\n\n'.join(deduped)

    # убираем артефакты нумерации страниц (одиночные числа на отдельной строке)
    text = re.sub(r'\n\s*-?\s*\d+\s*-?\s*\n', '\n', text)
    text = re.sub(r'^\s*\d+\s*\n', '', text)
    text = re.sub(r'\n\s*\d+\s*$', '', text)

    # нормализуем кавычки
    text = text.replace('«', '"').replace('»', '"')

    # убираем мягкий перенос (символ \xad в PDF)
    text = text.replace('\xad', '')

    return text.strip()


def preprocess_documents(documents: List[Document]) -> List[Document]:
    """Предобработка всех документов."""
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
    # фильтруем пустые документы (в т.ч. отсканированные PDF без текстового слоя)
    return [doc for doc in documents if len(doc.page_content) > 100]


def merge_documents_by_source(documents: List[Document]) -> List[Document]:
    """Объединяет постраничные документы (PDF грузится по страницам) в один документ на файл-источник."""

    grouped: Dict[str, List[Document]] = {}
    for doc in documents:
        source = doc.metadata.get("source", "")
        grouped.setdefault(source, []).append(doc)

    merged = []
    for source, docs in grouped.items():
        docs.sort(key=lambda d: d.metadata.get("page", 0))
        content = "\n\n".join(d.page_content for d in docs)
        metadata = {**docs[0].metadata}
        metadata.pop("page", None)
        metadata["page_count"] = len(docs)
        merged.append(Document(page_content=content, metadata=metadata))

    return merged


def save_processed_documents(
    documents: List[Document], output_dir: str = "data/processed"
) -> List[Path]:
    """Сохраняет предобработанные документы в виде .txt файлов (один файл на источник)."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for doc in documents:
        source = doc.metadata.get("source", "unknown")
        stem = Path(source).stem
        out_path = out_dir / f"{stem}.txt"
        out_path.write_text(doc.page_content, encoding="utf-8")
        saved_paths.append(out_path)

    print(f"Сохранено обработанных документов: {len(saved_paths)} -> {out_dir}/")
    return saved_paths


def run_preprocessing_pipeline(
    data_dir: str = "data/raw", output_dir: str = "data/processed"
) -> List[Document]:
    """Полный цикл фазы 1: загрузка -> очистка -> объединение по источнику -> сохранение."""
    documents = load_documents(data_dir)
    documents = preprocess_documents(documents)
    documents = merge_documents_by_source(documents)
    save_processed_documents(documents, output_dir)
    return documents


def load_processed_documents(processed_dir: str = "data/processed") -> List[Document]:
    """Загружает предобработанные .txt документы из data/processed (по одному на источник)."""

    loader = DirectoryLoader(
        processed_dir,
        glob="**/*.txt",
        loader_cls=lambda path: TextLoader(path, encoding="utf-8"),
        show_progress=True,
    )
    documents = loader.load()
    print(f"Загружено обработанных документов: {len(documents)}")
    return documents


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

    print(f"Создано чанков: {len(chunks)}")
    print(f"Средний размер чанка: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} символов")
    return chunks


def run_chunking_pipeline(processed_dir: str = "data/processed") -> List[Document]:
    """Загрузка предобработанных документов из data/processed"""
    documents = load_processed_documents(processed_dir)
    return split_documents(documents)


if __name__ == "__main__":
    run_preprocessing_pipeline()
    run_chunking_pipeline()
