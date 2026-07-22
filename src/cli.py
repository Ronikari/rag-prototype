"""Единая точка входа RAG-контура.

Offline-часть (ingest -> index) и online-часть (serve) имеют разные жизненные
циклы запуска (первая — вручную при обновлении корпуса, вторая — как долгоживущий
процесс), но обе стадии удобно держать под одной CLI-командой вместо разрозненных
`python -m src.<module>`.
"""

import logging
from typing import List, Tuple

import typer

app = typer.Typer(add_completion=False, help="RAG-контур: индексация нормативной документации и запуск сервисов.")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Подробный лог (DEBUG).")) -> None:
    _setup_logging(verbose)


def _report_failures(stage: str, failed: List[Tuple[str, str]]) -> None:
    if not failed:
        return
    typer.secho(f"[{stage}] ошибок: {len(failed)}", fg=typer.colors.RED, err=True)
    for source, error in failed:
        typer.echo(f"  {source}: {error}", err=True)


def _do_ingest(data_dir: str, output_dir: str, force: bool):
    from src.ingestion import run_preprocessing_pipeline

    result = run_preprocessing_pipeline(data_dir=data_dir, output_dir=output_dir, force=force)
    _report_failures("ingest", result.failed)
    return result


def _do_index(processed_dir: str, force: bool):
    from src.indexing import run_indexing_pipeline

    result = run_indexing_pipeline(processed_dir=processed_dir, force=force)
    _report_failures("index", result.failed)
    return result


@app.command()
def ingest(
    data_dir: str = typer.Option("data/raw", help="Каталог с исходными документами."),
    output_dir: str = typer.Option("data/processed", help="Каталог для очищенных текстов."),
    force: bool = typer.Option(False, "--force", help="Игнорировать манифест, переобработать все файлы."),
) -> None:
    """Фаза 1: загрузка, очистка, OCR сканов, извлечение таблиц -> data/processed/.

    Обрабатывает только новые/изменившиеся файлы (по манифесту data/manifest.json);
    ошибка на одном файле не прерывает обработку остальных.
    """
    result = _do_ingest(data_dir, output_dir, force)
    if result.failed:
        raise typer.Exit(code=1)


@app.command(name="index")
def index_cmd(
    processed_dir: str = typer.Option("data/processed", help="Каталог с очищенными текстами."),
    force: bool = typer.Option(False, "--force", help="Переиндексировать все источники, даже неизменившиеся."),
) -> None:
    """Фаза 2: инкрементальная индексация изменившихся источников в Qdrant.

    Пропускает источники, чьё содержимое не изменилось с прошлой индексации;
    точки удалённых исходников удаляются из коллекции.
    """
    result = _do_index(processed_dir, force)
    if result.failed:
        raise typer.Exit(code=1)


@app.command()
def reindex(
    data_dir: str = typer.Option("data/raw", help="Каталог с исходными документами."),
    output_dir: str = typer.Option("data/processed", help="Каталог для очищенных текстов."),
    processed_dir: str = typer.Option("data/processed", help="Каталог с очищенными текстами (для index)."),
    force: bool = typer.Option(False, "--force", help="Игнорировать манифест на обеих стадиях."),
) -> None:
    """ingest + index одной командой."""
    ingest_result = _do_ingest(data_dir, output_dir, force)
    index_result = _do_index(processed_dir, force)
    if ingest_result.failed or index_result.failed:
        raise typer.Exit(code=1)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Хост uvicorn (0.0.0.0 обязателен для доступа из Docker)."),
    port: int = typer.Option(8000, help="Порт uvicorn."),
) -> None:
    """Запуск FastAPI RAG-сервера (src/api.py) поверх Qdrant + Ollama."""
    import uvicorn

    uvicorn.run("src.api:app", host=host, port=port)


@app.command(name="eval")
def evaluate_cmd(
    limit: int = typer.Option(None, help="Ограничить число вопросов golden set (RAGAS с локальной LLM медленный)."),
) -> None:
    """RAGAS-оценка контура (faithfulness, answer_relevancy, context_precision, context_recall)."""
    from src.evaluation import run_evaluation

    run_evaluation(limit=limit)


@app.command(name="eval-retrieval")
def evaluate_retrieval_cmd(
    golden_path: str = typer.Option("data/golden_set.json", help="Путь к golden set (JSON с ключом items)."),
    json_out: str = typer.Option(None, help="Сохранить полный отчёт (метрики + ранги по каждому вопросу) в JSON-файл."),
) -> None:
    """Метрики поиска по golden set: Recall@k (k=1,3,5,10) и MRR, без LLM.

    Быстрая детерминированная проверка качества ретривера — в отличие от `eval`,
    не вызывает генерацию и LLM-судью, поэтому пригодна для сравнения вариантов
    поиска (top_k, реранкер и т.п.) после каждого изменения.
    """
    import json
    from pathlib import Path

    from src.retrieval_eval import print_report, run_retrieval_evaluation

    report = run_retrieval_evaluation(golden_path=Path(golden_path))
    print_report(report)

    if json_out:
        Path(json_out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        typer.echo(f"Отчёт сохранён: {json_out}")


if __name__ == "__main__":
    app()
