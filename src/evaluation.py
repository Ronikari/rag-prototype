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
from typing import List, Dict

from src.generation import build_rag_chain, get_llm
from src.retrieval import search_documents
from src.indexing import get_embeddings


EVAL_DATASET = [
    {
        "question": "Носит ли технологическое присоединение к электрическим сетям однократный характер?",
        "ground_truth": "Да, технологическое присоединение энергопринимающих устройств потребителей электрической энергии к объектам электросетевого хозяйства осуществляется в порядке, установленном Правительством Российской Федерации, и носит однократный характер.",
    },
    {
        "question": "На какой срок допускается продление плановой выездной проверки в рамках федерального государственного энергетического надзора?",
        "ground_truth": "Продление срока проведения выездной плановой проверки допускается в исключительных случаях (сложные и/или длительные исследования, испытания, экспертизы, расследования), но не более чем на пятнадцать рабочих дней.",
    },
]


def run_evaluation(eval_data: List[Dict] = None) -> Dict:
    """Запуск оценки RAG системы через RAGAS."""

    if eval_data is None:
        eval_data = EVAL_DATASET

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
