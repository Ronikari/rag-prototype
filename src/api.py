"""Оборачивает полный пайплайн Qdrant -> LLM в endpoint /v1/chat/completions,
чтобы чат-интерфейсы (Open WebUI и другие OpenAI-совместимые клиенты) работали
через гарантированный RAG-контур, не полагаясь на tool calling модели.
"""

import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.config import TOP_K
from src.generation import (
    NO_CONTEXT_ANSWER,
    _hold_back_open_tags,
    build_generation_chain,
    clean_llm_output,
    format_docs,
    format_sources,
)
from src.retrieval import get_retriever

MODEL_ID = "rag-normative-docs"

# тяжёлые компоненты (embedding-модель, подключение к БД) инициализируются
# один раз на старте сервера, а не на каждый запрос
components: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    components["retriever"] = get_retriever(top_k=TOP_K)
    components["generation"] = build_generation_chain()
    yield
    components.clear()


app = FastAPI(title="RAG normative docs API", lifespan=lifespan)


class ChatMessage(BaseModel, extra="ignore"):
    role: str
    content: Union[str, List[dict], None] = None


class ChatRequest(BaseModel, extra="ignore"):
    model: str = MODEL_ID
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None


def _extract_question(messages: List[ChatMessage]) -> str:
    """Последнее пользовательское сообщение; multimodal-части сводятся к тексту."""
    for message in reversed(messages):
        if message.role != "user":
            continue
        if isinstance(message.content, str):
            return message.content
        if isinstance(message.content, list):
            texts = [p.get("text", "") for p in message.content if p.get("type") == "text"]
            return "\n".join(t for t in texts if t)
    return ""


def _completion_response(answer: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": answer},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _chunk(delta: dict, finish_reason: Optional[str] = None) -> str:
    payload = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream_completion(question: str, docs: list):
    yield _chunk({"role": "assistant"})

    if not docs:
        yield _chunk({"content": NO_CONTEXT_ANSWER})
    else:
        accumulated = ""
        printed = 0
        for part in components["generation"].stream(
            {"context": format_docs(docs), "question": question}
        ):
            accumulated += part
            visible = clean_llm_output(_hold_back_open_tags(accumulated))
            if len(visible) > printed:
                yield _chunk({"content": visible[printed:]})
                printed = len(visible)
        yield _chunk({"content": f"\n\n{format_sources(docs)}"})

    yield _chunk({}, finish_reason="stop")
    yield "data: [DONE]\n\n"


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [{
            "id": MODEL_ID,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "rag-app",
        }],
    }


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    question = _extract_question(req.messages)
    if not question.strip():
        raise HTTPException(status_code=400, detail="Пустой вопрос: нет user-сообщения с текстом.")

    docs = components["retriever"].invoke(question)

    if req.stream:
        return StreamingResponse(
            _stream_completion(question, docs),
            media_type="text/event-stream",
        )

    if not docs:
        return _completion_response(NO_CONTEXT_ANSWER)

    answer = components["generation"].invoke(
        {"context": format_docs(docs), "question": question}
    )
    return _completion_response(f"{answer}\n\n{format_sources(docs)}")


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": MODEL_ID}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
