"""Оценка качества поиска (retrieval) по golden set: Recall@k и MRR.

Golden set (data/golden_set.json) — эталонные пары вопрос-ответ, каждая из
которых привязана к конкретному документу корпуса и месту в нём. В отличие от
RAGAS-оценки (src/evaluation.py), которой нужен LLM-судья, эти метрики
детерминированы и дёшевы: они проверяют только работу поиска, без генерации,
поэтому подходят для быстрого сравнения вариантов ретривера (top_k, гибридный
против dense-only, реранкер и т.д.).

Метрики (у каждого вопроса ровно один эталонный фрагмент):
- Recall@k — доля вопросов, для которых релевантный чанк попал в топ-k выдачи
  (при одном релевантном фрагменте на вопрос совпадает с Hit Rate@k);
- MRR — средний обратный ранг первого релевантного чанка (1/rank; 0, если
  релевантный чанк не найден в топ-k_max).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence

from langchain.schema import Document

from src.retrieval import search_documents

logger = logging.getLogger(__name__)

GOLDEN_SET_PATH = Path("data/golden_set.json")

# значения k, для которых считается Recall@k; MRR считается по max(k)
DEFAULT_K_VALUES = (1, 3, 5, 10)


def _normalize(text: str) -> str:
    """Нормализация текста для проверки вхождения эталонной фразы в чанк.

    Регистр приводится к нижнему, любые последовательности пробельных символов
    схлопываются в один пробел: в обработанных текстах (особенно после OCR)
    переносы строк разрывают предложения в произвольных местах, поэтому
    дословное сравнение без нормализации давало бы ложные промахи.
    """
    return " ".join(text.lower().split())


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> List[Dict]:
    """Читает golden set и возвращает список эталонных вопросов (ключ "items")."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["items"]


def is_relevant(doc: Document, item: Dict) -> bool:
    """Релевантен ли найденный чанк эталонному вопросу.

    Чанк релевантен, если он взят из нужного документа (metadata.source) и
    содержит одну из эталонных фраз (match_phrases) — короткую дословную цитату
    из фрагмента, отвечающего на вопрос. Фразы, а не метки пунктов, выбраны
    основным критерием, потому что в одном файле постановления может быть
    несколько приложений с независимой нумерацией пунктов: метка "пункт 16"
    в метаданных чанка неоднозначна, а цитата из нужного пункта — однозначна.

    Резервные критерии (когда match_phrases пуст): совпадение метки section
    из списка sections, а если и он пуст — только совпадение источника.
    """
    if doc.metadata.get("source") != item["source"]:
        return False

    phrases = item.get("match_phrases") or []
    if phrases:
        content = _normalize(doc.page_content)
        return any(_normalize(p) in content for p in phrases)

    sections = item.get("sections") or []
    if sections:
        return doc.metadata.get("section") in sections

    return True


def evaluate_item(item: Dict, k_max: int) -> Dict:
    """Прогоняет один вопрос через поиск и находит ранг первого релевантного чанка.

    Возвращает словарь с рангом (None — промах) и краткой сводкой топа выдачи —
    она нужна для ручного разбора промахов (какие источники вытеснили эталонный).
    """
    results = search_documents(item["question"], top_k=k_max)

    first_rank = None
    for rank, (doc, _score) in enumerate(results, start=1):
        if is_relevant(doc, item):
            first_rank = rank
            break

    return {
        "id": item["id"],
        "question": item["question"],
        "reference": item.get("reference", ""),
        "first_relevant_rank": first_rank,
        "top_results": [
            {
                "rank": rank,
                "source": Path(doc.metadata.get("source", "")).name,
                "section": doc.metadata.get("section", ""),
            }
            for rank, (doc, _score) in enumerate(results[:3], start=1)
        ],
    }


def run_retrieval_evaluation(
    golden_path: Path = GOLDEN_SET_PATH,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> Dict:
    """Считает Recall@k и MRR по всем вопросам golden set.

    Поиск выполняется один раз на вопрос с top_k = max(k_values), метрики для
    меньших k вычисляются по рангу первого релевантного чанка — без повторных
    запросов к Qdrant.
    """
    items = load_golden_set(golden_path)
    k_values = sorted(set(k_values))
    k_max = max(k_values)

    per_item: List[Dict] = []
    for i, item in enumerate(items, start=1):
        result = evaluate_item(item, k_max)
        per_item.append(result)
        logger.info(
            "[%d/%d] %s -> ранг %s",
            i, len(items), item["id"], result["first_relevant_rank"] or "промах",
        )

    ranks = [r["first_relevant_rank"] for r in per_item]
    n = len(ranks)

    recall_at_k = {
        k: sum(1 for r in ranks if r is not None and r <= k) / n
        for k in k_values
    }
    mrr = sum(1.0 / r for r in ranks if r is not None) / n

    return {
        "golden_set": str(golden_path),
        "num_questions": n,
        "k_max": k_max,
        "recall_at_k": recall_at_k,
        "mrr": round(mrr, 4),
        "misses": [r for r in per_item if r["first_relevant_rank"] is None],
        "per_question": per_item,
    }


def print_report(report: Dict) -> None:
    """Печатает сводку метрик и список промахов в терминал."""
    print(f"\nВопросов: {report['num_questions']} (top_k поиска: {report['k_max']})")
    for k, value in report["recall_at_k"].items():
        print(f"  Recall@{k:<3} {value:.3f}")
    print(f"  MRR      {report['mrr']:.3f}")

    misses = report["misses"]
    if misses:
        print(f"\nПромахи ({len(misses)}) — релевантный чанк не найден в топ-{report['k_max']}:")
        for miss in misses:
            print(f"  [{miss['id']}] {miss['question']}")
            for top in miss["top_results"]:
                print(f"      {top['rank']}. {top['source']} | {top['section']}")
    else:
        print("\nПромахов нет: каждый вопрос находит свой фрагмент в выдаче.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print_report(run_retrieval_evaluation())
