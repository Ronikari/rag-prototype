from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from datasets import Dataset
from typing import List, Dict, Optional

from src.generation import build_rag_chain, get_llm
from src.retrieval import search_documents
from src.retrieval_eval import load_golden_set
from src.vectorstore import get_embeddings


def run_evaluation(eval_data: List[Dict] = None, limit: Optional[int] = None) -> Dict:
    """Запуск оценки RAG системы через RAGAS.

    По умолчанию вопросы и эталонные ответы берутся из golden set
    (data/golden_set.json) — того же, по которому считаются метрики поиска
    в src/retrieval_eval.py. `limit` ограничивает число вопросов: RAGAS
    прогоняет каждый вопрос через генерацию и LLM-судью, на локальной модели
    полный набор занимает десятки минут.
    """

    if eval_data is None:
        eval_data = load_golden_set()
    if limit:
        eval_data = eval_data[:limit]

    # собираем данные для оценки
    questions = []
    ground_truths = []
    answers = []
    contexts = []

    chain = build_rag_chain()

    for item in eval_data:
        q = item["question"]
        questions.append(q)
        ground_truths.append(item["ground_truth"])

        # получаем ответ и контексты
        retrieved = search_documents(q, top_k=5)
        ctx_texts = [doc.page_content for doc, _ in retrieved]
        contexts.append(ctx_texts)

        answer = chain.invoke(q)
        answers.append(answer)

    # датасет для RAGAS
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    # используем Ollama как judge LLM
    judge_llm = LangchainLLMWrapper(get_llm(temperature=0.0, max_tokens=4096))
    judge_embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    run_config = RunConfig(timeout=300, max_workers=2)

    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=run_config,
    )

    print(results)
    return results


if __name__ == "__main__":
    run_evaluation()
