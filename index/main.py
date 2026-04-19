import asyncio
import logging
import os
import re
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import pymorphy3

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


app = FastAPI(title="Index Service", version="0.3.0")



TIME_GAP_SECONDS = 20 * 60       # 20 минут — если разрыв больше, считаем новую сессию
MAX_MESSAGES_PER_CHUNK = 12      # жёсткий потолок сообщений в чанке (анти-размытие dense)
MIN_MESSAGES_PER_CHUNK = 1
OVERLAP_MESSAGES = 2             # сколько сообщений из предыдущей сессии переносим в следующую
MAX_QUOTE_CONTEXTS = 2           # не более 2-х подмешиваемых цитируемых сессий на чанк

MAX_CHARS_PER_CHUNK_DENSE = 4000
MIN_CHUNK_LEN = 20

SPARSE_MODEL_NAME = "Qdrant/bm25"
FASTEMBED_CACHE_PATH = "/models/fastembed"
MAX_SPARSE_LEN = 4000

UVICORN_WORKERS = 2

_WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z0-9]+", re.UNICODE)
_LATIN_ONLY = re.compile(r"^[A-Za-z]+$")


@lru_cache(maxsize=1)
def get_morph() -> pymorphy3.MorphAnalyzer:
    logger.info("Loading pymorphy3 MorphAnalyzer (ru)")
    return pymorphy3.MorphAnalyzer(lang="ru")


@lru_cache(maxsize=200_000)
def _lemma_ru(word: str) -> str:
    try:
        return get_morph().parse(word)[0].normal_form
    except Exception:
        return word


def lemmatize_text(text: str) -> str:
    """Токенизация + лемматизация русских слов; латиница/цифры - в lower без изменений.

    Задача: сколлапсить падежи, числа, глагольные формы в нормальную форму,
    чтобы BM25 видел 'совещание' / 'совещания' / 'совещанию' как один токен.
    """
    if not text:
        return ""
    tokens: list[str] = []
    for match in _WORD_RE.finditer(text):
        token = match.group(0).lower()
        if token.isdigit():
            tokens.append(token)
        elif _LATIN_ONLY.match(token):
            # Английские слова pymorphy3 (ru) парсит плохо — оставляем как есть.
            tokens.append(token)
        else:
            tokens.append(_lemma_ru(token))
    return " ".join(tokens)


def _sender_short(sender_id: str) -> str:
    """ivan.petrov@vk.company -> 'Ivan Petrov'."""
    if not sender_id:
        return ""
    local = sender_id.split("@", 1)[0]
    pretty = local.replace(".", " ").replace("_", " ").strip()
    return pretty.title() if pretty else ""


def _is_useful(message: Message) -> bool:
    """Считаем сообщение полезным, если в нём есть реальный контент."""
    if message.is_system or message.is_hidden:
        return False
    has_text = bool(message.text and message.text.strip())
    has_parts = bool(message.parts) and any(
        isinstance(p.get("text"), str) and p.get("text", "").strip()
        for p in (message.parts or [])
    )
    has_files = bool(message.file_snippets and message.file_snippets.strip())
    return has_text or has_parts or has_files


def _extract_parts(message: Message) -> dict[str, list[str]]:
    """Разбирает parts по media_type -> quote / forward / other тексты."""
    buckets: dict[str, list[str]] = {"quote": [], "forward": [], "other": []}
    for p in (message.parts or []):
        text = p.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        mtype = (p.get("mediaType") or p.get("media_type") or "").strip()
        if mtype == "quote":
            buckets["quote"].append(text.strip())
        elif mtype == "forward":
            buckets["forward"].append(text.strip())
        else:
            buckets["other"].append(text.strip())
    return buckets


def render_for_dense(message: Message) -> str:
    """Натуральный текст для Qwen3-Embedding.

    Модель обучена на связной речи — подаём форматом
    'Пользователь X пишет: ...' с явными ремарками про пересылки,
    цитаты и вложения. Это даёт семантический сигнал "кто что сказал".
    """
    sender = _sender_short(message.sender_id) or "Пользователь"
    lines: list[str] = []

    if message.text and message.text.strip():
        lines.append(f"Пользователь {sender} пишет: {message.text.strip()}")

    parts = _extract_parts(message)
    for quote_text in parts["quote"]:
        lines.append(f"Пользователь {sender} цитирует сообщение: «{quote_text}»")
    for fwd_text in parts["forward"]:
        lines.append(f"Пользователь {sender} пересылает сообщение: «{fwd_text}»")
    for other_text in parts["other"]:
        # избегаем дублирования с message.text
        if message.text and other_text in message.text:
            continue
        lines.append(f"Пользователь {sender} добавляет: {other_text}")

    if message.file_snippets and message.file_snippets.strip():
        lines.append(
            f"Пользователь {sender} приложил файл с содержимым: {message.file_snippets.strip()}"
        )

    if message.mentions:
        mentions = [_sender_short(m) for m in message.mentions if m]
        mentions = [m for m in mentions if m]
        if mentions:
            lines.append(f"В сообщении упоминаются: {', '.join(mentions)}.")

    if not lines and message.text and message.text.strip():
        lines.append(f"Пользователь {sender} пишет: {message.text.strip()}")

    return "\n".join(lines).strip()


def render_for_sparse(message: Message) -> str:
    """Keyword-rich текст для BM25, лемматизированный.

    В BM25 важна частота токенов и их нормальная форма. Конкатенируем
    всё, что имеет лексическое значение (sender, text, parts, файлы,
    mentions), и прогоняем через pymorphy3.
    """
    sender = _sender_short(message.sender_id)
    raw: list[str] = []

    if sender:
        raw.append(sender)

    if message.text and message.text.strip():
        raw.append(message.text.strip())

    parts = _extract_parts(message)
    raw.extend(parts["quote"])
    raw.extend(parts["forward"])
    raw.extend(parts["other"])

    if message.file_snippets and message.file_snippets.strip():
        raw.append(message.file_snippets.strip())

    if message.mentions:
        mentions_clean = [_sender_short(m) for m in message.mentions if m]
        raw.extend([m for m in mentions_clean if m])

    if message.is_forward:
        raw.append("пересылка")
    if message.is_quote:
        raw.append("цитата")

    joined = " ".join(raw)
    lemmatized = lemmatize_text(joined)
    return lemmatized if lemmatized else joined.lower()


def build_sessions(messages: list[Message]) -> list[list[Message]]:
    """Режет отсортированные по времени сообщения на сессии.

    Правила разреза:
      * разрыв во времени > TIME_GAP_SECONDS ⇒ новая сессия;
      * текущая сессия достигла MAX_MESSAGES_PER_CHUNK ⇒ принудительный разрез.
    Системные/скрытые/пустые отфильтрованы.
    """
    useful = [m for m in messages if _is_useful(m)]
    useful.sort(key=lambda m: m.time)

    sessions: list[list[Message]] = []
    current: list[Message] = []

    for msg in useful:
        if not current:
            current.append(msg)
            continue

        gap = msg.time - current[-1].time
        if gap > TIME_GAP_SECONDS or len(current) >= MAX_MESSAGES_PER_CHUNK:
            sessions.append(current)
            current = [msg]
        else:
            current.append(msg)

    if current:
        sessions.append(current)

    return sessions


def _find_quoted_session(
    quoting_msg: Message,
    all_sessions: list[list[Message]],
    own_session_head_id: str | None,
) -> list[Message] | None:
    """Ищет сессию, в которой лежит оригинал цитируемого сообщения.

    Эвристика:
      1. Пробуем найти по id/sn в полях quote-part'а.
      2. Fallback: подстрока текста цитаты (первые 80 символов) в m.text.
    Возвращает None, если не нашли или нашли саму себя.
    """
    parts = _extract_parts(quoting_msg)
    if not parts["quote"]:
        return None

    # 1) Поиск по id/sn внутри quote-part
    target_ids: set[str] = set()
    for p in (quoting_msg.parts or []):
        mtype = (p.get("mediaType") or p.get("media_type") or "").strip()
        if mtype != "quote":
            continue
        for key in (
            "quoted_message_id",
            "quotedMessageId",
            "messageId",
            "message_id",
            "sn",
            "messageSn",
            "quoteSn",
        ):
            val = p.get(key)
            if isinstance(val, str) and val:
                target_ids.add(val)

    if target_ids:
        for session in all_sessions:
            head_id = session[0].id if session else None
            if head_id == own_session_head_id:
                continue
            for m in session:
                if m.id in target_ids:
                    return session

    # 2) Fallback — подстрока текста цитаты
    probe_raw = parts["quote"][0].strip().lower()
    probe = probe_raw[:80]
    if len(probe) < 15:
        return None
    for session in all_sessions:
        head_id = session[0].id if session else None
        if head_id == own_session_head_id:
            continue
        for m in session:
            if not m.text:
                continue
            if probe in m.text.lower():
                return session

    return None


def build_chunks(
    chat: Chat,
    overlap_messages: list[Message],
    new_messages: list[Message],
) -> list[IndexAPIItem]:
    """Основная функция: строим сессии, рендерим dense/sparse, подмешиваем quote-контекст.

    Логика:
      1. Строим сессии по всему корпусу (overlap + new) — чтобы границы были честными.
      2. Индексируем только сессии, где есть хотя бы одно new-сообщение.
      3. В начало каждой сессии добавляем OVERLAP_MESSAGES из конца предыдущей.
      4. Если в сессии есть is_quote-сообщения — ищем сессии их оригиналов
         и подмешиваем в конец чанка как [Контекст цитируемого обсуждения].
    """
    results: list[IndexAPIItem] = []

    new_ids = {m.id for m in new_messages if _is_useful(m)}
    if not new_ids:
        return results

    all_messages = list(overlap_messages) + list(new_messages)
    all_sessions = build_sessions(all_messages)

    prev_tail: list[Message] = []

    for session in all_sessions:
        # Индексируем только сессии, которые содержат хотя бы одно новое сообщение
        if not any(m.id in new_ids for m in session):
            prev_tail = session[-OVERLAP_MESSAGES:] if OVERLAP_MESSAGES else []
            continue

        # Базовый набор: overlap (хвост прошлой сессии) + сама сессия
        chunk_messages: list[Message] = []
        seen_ids: set[str] = set()

        for m in prev_tail:
            if m.id not in seen_ids:
                chunk_messages.append(m)
                seen_ids.add(m.id)
        for m in session:
            if m.id not in seen_ids:
                chunk_messages.append(m)
                seen_ids.add(m.id)

        # Сортировка по времени — чтобы overlap встал в начало естественно
        chunk_messages.sort(key=lambda x: x.time)

        # Quote-контекст: ищем сессии цитируемых оригиналов
        session_head_id = session[0].id if session else None
        quote_context_blocks: list[list[Message]] = []
        quote_block_head_ids: set[str] = set()

        for m in session:
            if not m.is_quote:
                continue
            if len(quote_context_blocks) >= MAX_QUOTE_CONTEXTS:
                break
            quoted_session = _find_quoted_session(m, all_sessions, session_head_id)
            if not quoted_session:
                continue
            head = quoted_session[0].id if quoted_session else None
            if not head or head in quote_block_head_ids:
                continue
            quote_block_head_ids.add(head)
            quote_context_blocks.append(quoted_session)

        chat_header_dense = f"Чат «{chat.name}» (тип: {chat.type}):"
        chat_header_sparse = lemmatize_text(f"{chat.name} {chat.type}")

        dense_lines: list[str] = [chat_header_dense]
        sparse_tokens: list[str] = [chat_header_sparse] if chat_header_sparse else []
        page_lines: list[str] = [chat_header_dense]

        for m in chunk_messages:
            d = render_for_dense(m)
            s = render_for_sparse(m)
            if d:
                dense_lines.append(d)
                page_lines.append(d)
            if s:
                sparse_tokens.append(s)

        # Подмешиваем цитируемые сессии в хвост
        for qblock in quote_context_blocks:
            dense_lines.append("\n[Контекст цитируемого обсуждения]:")
            page_lines.append("\n[Контекст цитируемого обсуждения]:")
            for qm in qblock:
                qd = render_for_dense(qm)
                qs = render_for_sparse(qm)
                if qd:
                    dense_lines.append(qd)
                    page_lines.append(qd)
                if qs:
                    sparse_tokens.append(qs)

        dense_content = "\n".join(l for l in dense_lines if l).strip()
        page_content = "\n".join(l for l in page_lines if l).strip()
        sparse_content = " ".join(t for t in sparse_tokens if t).strip()

        # Safety cap — не размываем dense-вектор гипер-длиной
        if len(dense_content) > MAX_CHARS_PER_CHUNK_DENSE:
            dense_content = dense_content[:MAX_CHARS_PER_CHUNK_DENSE]
        if len(page_content) > MAX_CHARS_PER_CHUNK_DENSE:
            page_content = page_content[:MAX_CHARS_PER_CHUNK_DENSE]

        if len(page_content) < MIN_CHUNK_LEN:
            prev_tail = session[-OVERLAP_MESSAGES:] if OVERLAP_MESSAGES else []
            continue

        # Собираем message_ids: сначала основной чанк, потом quote-контекст
        message_ids: list[str] = []
        seen_out: set[str] = set()
        for m in chunk_messages:
            if m.id not in seen_out:
                message_ids.append(m.id)
                seen_out.add(m.id)
        for qblock in quote_context_blocks:
            for m in qblock:
                if m.id not in seen_out:
                    message_ids.append(m.id)
                    seen_out.add(m.id)

        results.append(
            IndexAPIItem(
                page_content=page_content,
                dense_content=dense_content,
                sparse_content=sparse_content,
                message_ids=message_ids,
            )
        )

        prev_tail = session[-OVERLAP_MESSAGES:] if OVERLAP_MESSAGES else []

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
    return SparseTextEmbedding(
        model_name=SPARSE_MODEL_NAME,
        cache_dir=FASTEMBED_CACHE_PATH,
    )


def embed_sparse_texts(texts: list[str]) -> list[dict[str, list[int] | list[float]]]:
    """Важно: лемматизируем И на индексации, И на запросах — чтобы BM25 видел одни токены.

    /sparse_embedding вызывается тестирующей системой в обоих режимах.
    Лемматизация идемпотентна для уже лемматизированных текстов.
    """
    model = get_sparse_model()

    safe_texts: list[str] = []
    for t in texts:
        lem = lemmatize_text(t)[:MAX_SPARSE_LEN]
        if not lem.strip():
            # fallback: не даём пустой текст в BM25 (получим пустой вектор)
            fallback = (t or "").strip()[:MAX_SPARSE_LEN].lower() or "пусто"
            lem = fallback
        safe_texts.append(lem)

    vectors: list[dict[str, list[int] | list[float]]] = []
    for item in model.embed(safe_texts, batch_size=128):
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