import re
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import Document
from typing import List

from src.config import LLM_MODEL, OLLAMA_BASE_URL, OLLAMA_NUM_CTX, TOP_K
from src.retrieval import get_retriever

NO_CONTEXT_ANSWER = (
    "В доступной нормативной документации не найдено информации по данному вопросу."
)


def get_llm(temperature: float = 0.2, max_tokens: int = 6144) -> ChatOllama:
    """Инициализация LLM через нативный API Ollama.

    У qwen3:4b режим рассуждений не отключается (`think: false` лишь убирает теги,
    и рассуждения утекают в текст ответа; `/no_think` игнорируется), поэтому thinking
    оставлен включённым — нативный API отдаёт его отдельным полем, и в content
    попадает только чистый ответ. num_predict при этом покрывает рассуждения
    (~2-3.5K токенов) И сам ответ, отсюда бюджет 6144 вместо прежних 2048.
    """
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=LLM_MODEL,
        temperature=temperature,    # низкая температура для фактических ответов
        num_predict=max_tokens,     # thinking + развёрнутый ответ, см. docstring
        num_ctx=OLLAMA_NUM_CTX,
        keep_alive=-1,              # не выгружать модель после простоя (как LM Studio)
    )


SYSTEM_PROMPT = """Ты — эксперт-консультант по нормативной и правовой документации \
в области электроэнергетики. Дай развёрнутый, структурированный ответ на вопрос \
пользователя, опираясь ИСКЛЮЧИТЕЛЬНО на приведённый ниже контекст из нормативных документов.

Требования к ответу:
1. Начни с краткого прямого ответа или определения (1-2 предложения).
2. Затем раскрой тему подробно: суть, условия, порядок, участники, сроки, \
исключения — всё, что содержится в контексте по данному вопросу.
3. Ссылаясь на положения, указывай название документа и номер статьи или пункта \
(например: "согласно статье 26 Федерального закона N 35-ФЗ..."). Номер статьи или \
пункта можно указывать ТОЛЬКО если он явно виден в контексте — в подписи \
[Документ: ...] или в самом тексте фрагмента. Если номер в контексте не виден, \
сошлись только на название документа. Никогда не восстанавливай и не угадывай \
номера пунктов по памяти или по соседним фрагментам.
4. Никогда не используй внутренние обозначения вида "Фрагмент 1" или "Источник 2" — \
ссылайся только на названия самих документов.
5. Если контекст не содержит ответа на вопрос, прямо скажи об этом — не придумывай информацию.
6. Пиши только чистый текст ответа на русском языке: без служебных тегов, \
без рассуждений вслух, без повторения вопроса.
7. Не добавляй раздел "Источники" в конце — он будет сформирован автоматически.

Контекст из нормативной документации:
{context}"""

HUMAN_TEMPLATE = "Вопрос: {question}"

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_TEMPLATE),
])


# закрытые блоки рассуждений (<think>...</think> у Qwen и подобных)
THINK_BLOCK_RE = re.compile(
    r"<(think|thinking|reasoning)>.*?</\1>", re.DOTALL | re.IGNORECASE
)
# одиночные служебные токены шаблонов чата (Gemma, Qwen, Llama и т.д.)
SERVICE_TOKEN_RE = re.compile(
    r"</?(?:think|thinking|reasoning)>"
    r"|<\|[^|>]{0,40}\|>"
    r"|<(?:start|end)_of_turn>"
    r"|<(?:bos|eos|pad|unused\d*)>"
    r"|\[/?INST\]",
    re.IGNORECASE,
)


def doc_title(doc: Document) -> str:
    """Человекочитаемое название документа: имя файла-источника без пути и расширения."""
    source = doc.metadata.get("source", "Неизвестный источник")
    return Path(source).stem


def clean_llm_output(text: str) -> str:
    """Очистка ответа LLM от служебных тегов и артефактов chat-шаблона."""
    text = THINK_BLOCK_RE.sub("", text)
    text = SERVICE_TOKEN_RE.sub("", text)
    # метка роли, остающаяся после удаления токенов chat-шаблона (<start_of_turn>model)
    text = re.sub(r"^\s*(?:model|assistant)\s*\n", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_docs(docs: List[Document]) -> str:
    """Форматирование документов для передачи в промпт.

    Каждый чанк подписывается названием документа-источника (а не номером фрагмента)
    и, если известен, номером статьи/пункта из метаданных — чтобы модель ссылалась
    на сами документы и цитировала только реально видимые номера пунктов.
    """
    formatted = []
    for doc in docs:
        source_info = doc_title(doc)
        section = doc.metadata.get("section", "")
        if section:
            source_info += f", {section}"
        formatted.append(f'[Документ: "{source_info}"]\n{doc.page_content}')
    return "\n\n---\n\n".join(formatted)


def format_sources(docs: List[Document]) -> str:
    """Список использованных документов (без дублей, в порядке релевантности)."""
    titles: List[str] = []
    for doc in docs:
        title = doc_title(doc)
        if title not in titles:
            titles.append(title)
    lines = "\n".join(f"{i}. {title}" for i, title in enumerate(titles, 1))
    return f"Использованные нормативные документы:\n{lines}"


def build_generation_chain(llm: ChatOllama = None):
    """Промпт -> LLM -> очистка вывода. Вход: {"context": str, "question": str}."""
    llm = llm or get_llm()
    return RAG_PROMPT | llm | StrOutputParser() | RunnableLambda(clean_llm_output)


def build_rag_chain(top_k: int = TOP_K):
    """Сборка RAG цепочки: retrieval -> генерация -> очистка -> добавление источников.

    Список источников формируется программно из метаданных найденных чанков Qdrant,
    а не генерацией LLM — это гарантирует его наличие и точность.
    """
    retriever = get_retriever(top_k=top_k)
    generation = build_generation_chain()

    def _answer(inputs: dict) -> str:
        docs = inputs["docs"]
        if not docs:
            return NO_CONTEXT_ANSWER
        answer = generation.invoke(
            {"context": format_docs(docs), "question": inputs["question"]}
        )
        return f"{answer}\n\n{format_sources(docs)}"

    chain = (
        {"docs": retriever, "question": RunnablePassthrough()}
        | RunnableLambda(_answer)
    )

    return chain


def stream_answer(question: str, top_k: int = TOP_K) -> None:
    """Стриминг ответа из Ollama; источники печатаются после ответа.

    Служебные теги фильтруются на лету: наружу выводится только очищенная часть
    накопленного текста, а незакрытые теги придерживаются до их закрытия.
    """
    retriever = get_retriever(top_k=top_k)
    docs = retriever.invoke(question)
    if not docs:
        print(NO_CONTEXT_ANSWER)
        return

    generation = build_generation_chain()
    accumulated = ""
    printed = 0
    for chunk in generation.stream(
        {"context": format_docs(docs), "question": question}
    ):
        accumulated += chunk
        visible = clean_llm_output(_hold_back_open_tags(accumulated))
        if len(visible) > printed:
            print(visible[printed:], end="", flush=True)
            printed = len(visible)
    print(f"\n\n{format_sources(docs)}")


def _hold_back_open_tags(text: str) -> str:
    """Отрезает хвост стрима с незакрытым блоком рассуждений или недописанным тегом."""
    open_think = re.search(r"<(think|thinking|reasoning)>(?!.*</\1>)", text,
                           re.DOTALL | re.IGNORECASE)
    if open_think:
        return text[: open_think.start()]
    last_angle = text.rfind("<")
    if last_angle != -1 and ">" not in text[last_angle:]:
        return text[:last_angle]
    return text
