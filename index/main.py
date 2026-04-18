import logging
import os
import asyncio
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("index-service")


class Chat(BaseModel):
    id: str
    name: str
    sn: str
    type: str
    is_public: bool | None = None
    members_count: int | None = None
    members: list[dict[str, Any]] | None = None


class Message(BaseModel):
    id: str
    thread_sn: str | None = None
    time: int
    text: str
    sender_id: str
    file_snippets: str
    parts: list[dict[str, Any]] | None = None
    mentions: list[str] | None = None
    member_event: dict[str, Any] | None = None
    is_system: bool
    is_hidden: bool
    is_forward: bool
    is_quote: bool


class ChatData(BaseModel):
    chat: Chat
    overlap_messages: list[Message]
    new_messages: list[Message]


class IndexAPIRequest(BaseModel):
    data: ChatData


class IndexAPIItem(BaseModel):
    page_content: str
    dense_content: str
    sparse_content: str
    message_ids: list[str]


class IndexAPIResponse(BaseModel):
    results: list[IndexAPIItem]


class SparseEmbeddingRequest(BaseModel):
    texts: list[str]


class SparseVector(BaseModel):
    indices: list[int]
    values: list[float]


class SparseEmbeddingResponse(BaseModel):
    vectors: list[SparseVector]


app = FastAPI(title="Index Service", version="0.2.0")

CHUNK_CHAR_BUDGET = 1200
OVERLAP_CHAR_BUDGET = 300
MIN_CHUNK_LEN = 20
SPARSE_MODEL_NAME = "Qdrant/bm25"
FASTEMBED_CACHE_PATH = "/models/fastembed"
UVICORN_WORKERS = 4


def _sender_short(sender_id: str) -> str:
    if not sender_id:
        return ""
    local = sender_id.split("@", 1)[0]
    return local.replace(".", " ").replace("_", " ").strip()


def render_message(message: Message, chat_name: str = "") -> str:
    """Преобразует одно сообщение в текст для чанкинга.

    Включает имя отправителя и типы parts (forward/quote) — это даёт сигнал
    и для dense (семантика кто-что сказал), и для sparse (BM25 по токенам).
    """
    segments: list[str] = []
    sender_short = _sender_short(message.sender_id)
    if sender_short:
        segments.append(f"{sender_short}:")

    if message.text:
        segments.append(message.text)

    if message.parts:
        for part in message.parts:
            part_text = part.get("text")
            if not isinstance(part_text, str) or not part_text:
                continue
            media_type = part.get("mediaType") or part.get("media_type") or ""
            if media_type == "forward":
                segments.append(f"[переслано] {part_text}")
            elif media_type == "quote":
                segments.append(f"[цитата] {part_text}")
            else:
                segments.append(part_text)

    if message.file_snippets:
        segments.append(f"[файл] {message.file_snippets}")

    if message.mentions:
        mentions_short = " ".join(_sender_short(m) for m in message.mentions if m)
        if mentions_short:
            segments.append(f"(упоминания: {mentions_short})")

    return " ".join(segments).strip()


def _should_skip(message: Message) -> bool:
    if message.is_system or message.is_hidden:
        return True
    has_text = bool(message.text)
    has_parts = bool(message.parts) and any(
        isinstance(p.get("text"), str) and p.get("text") for p in message.parts
    )
    has_files = bool(message.file_snippets)
    return not (has_text or has_parts or has_files)


def build_chunks(
    chat: Chat,
    overlap_messages: list[Message],
    new_messages: list[Message],
) -> list[IndexAPIItem]:
    """Группирует сообщения в чанки по бюджету символов с текстовым overlap'ом.

    Чанки режутся на границах сообщений, чтобы не ломать контекст. Оверлап
    реализован как текстовый хвост предыдущего чанка, не как повтор сообщений —
    так мы не плодим дубликаты message_ids и не теряем границы.
    """
    results: list[IndexAPIItem] = []

    rendered_overlap = [
        render_message(m, chat.name) for m in overlap_messages if not _should_skip(m)
    ]
    overlap_text = "\n".join(t for t in rendered_overlap if t)
    overlap_tail = overlap_text[-OVERLAP_CHAR_BUDGET:] if overlap_text else ""

    useful_new = [m for m in new_messages if not _should_skip(m)]
    if not useful_new:
        return results

    current_texts: list[str] = []
    current_ids: list[str] = []
    current_len = 0

    def flush(tail: str) -> str:
        nonlocal current_texts, current_ids, current_len
        if not current_texts:
            return tail
        body = "\n".join(current_texts)
        chunk_text = f"{tail}\n{body}" if tail else body
        if len(chunk_text.strip()) >= MIN_CHUNK_LEN:
            results.append(
                IndexAPIItem(
                    page_content=chunk_text,
                    dense_content=chunk_text,
                    sparse_content=chunk_text,
                    message_ids=list(current_ids),
                )
            )
        next_tail = body[-OVERLAP_CHAR_BUDGET:] if body else tail
        current_texts = []
        current_ids = []
        current_len = 0
        return next_tail

    for message in useful_new:
        rendered = render_message(message, chat.name)
        if not rendered:
            continue
        rendered_len = len(rendered) + 1

        if current_len + rendered_len > CHUNK_CHAR_BUDGET and current_texts:
            overlap_tail = flush(overlap_tail)

        current_texts.append(rendered)
        current_ids.append(message.id)
        current_len += rendered_len

    flush(overlap_tail)

    return results


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/index", response_model=IndexAPIResponse)
async def index(payload: IndexAPIRequest) -> IndexAPIResponse:
    return IndexAPIResponse(
        results=build_chunks(
            payload.data.chat,
            payload.data.overlap_messages,
            payload.data.new_messages,
        )
    )


@lru_cache(maxsize=1)
def get_sparse_model():
    from fastembed import SparseTextEmbedding

    logger.info(
        "Loading sparse model %s from cache %s",
        SPARSE_MODEL_NAME,
        FASTEMBED_CACHE_PATH,
    )
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


def embed_sparse_texts(texts: list[str]) -> list[dict[str, list[int] | list[float]]]:
    model = get_sparse_model()
    vectors: list[dict[str, list[int] | list[float]]] = []
    for item in model.embed(texts):
        vectors.append(
            {
                "indices": item.indices.tolist(),
                "values": item.values.tolist(),
            }
        )
    return vectors


@app.post("/sparse_embedding")
async def sparse_embedding(payload: SparseEmbeddingRequest) -> dict[str, Any]:
    vectors = await asyncio.to_thread(embed_sparse_texts, payload.texts)
    return {"vectors": vectors}


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(exc)

    if isinstance(exc, RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    return JSONResponse(status_code=500, content={"detail": str(exc)})


def main() -> None:
    import uvicorn

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
        workers=UVICORN_WORKERS,
    )


if __name__ == "__main__":
    main()
