import logging
import os
import re
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any

import httpx
import pymorphy3
from fastembed import SparseTextEmbedding
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, models

EMBEDDINGS_DENSE_MODEL = "Qwen/Qwen3-Embedding-0.6B"

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

API_KEY = os.getenv("API_KEY")
EMBEDDINGS_DENSE_URL = os.getenv("EMBEDDINGS_DENSE_URL")
QDRANT_DENSE_VECTOR_NAME = os.getenv("QDRANT_DENSE_VECTOR_NAME", "dense")
QDRANT_SPARSE_VECTOR_NAME = os.getenv("QDRANT_SPARSE_VECTOR_NAME", "sparse")
SPARSE_MODEL_NAME = "Qdrant/bm25"
RERANKER_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"
RERANKER_URL = os.getenv("RERANKER_URL")
OPEN_API_LOGIN = os.getenv("OPEN_API_LOGIN")
OPEN_API_PASSWORD = os.getenv("OPEN_API_PASSWORD")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "evaluation")
REQUIRED_ENV_VARS = [
    "EMBEDDINGS_DENSE_URL",
    "RERANKER_URL",
    "QDRANT_URL",
]

# Qwen3-Embedding официально рекомендует оборачивать QUERY (не документы) в
# формат "Instruct: {task}\nQuery: {q}". Это асимметрия by design — модель
# обучена её ожидать, и на retrieval-бенчмарках даёт +1–3% к recall.
# Если в какой-то момент захочешь отключить — выстави DENSE_QUERY_INSTRUCT=0.
DENSE_QUERY_INSTRUCT_ENABLED = os.getenv("DENSE_QUERY_INSTRUCT", "1") not in ("0", "false", "False", "")
DENSE_INSTRUCT_TASK = os.getenv(
    "DENSE_INSTRUCT_TASK",
    "Given a user question, retrieve relevant chat messages that answer the question.",
)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("search-service")


def validate_required_env() -> None:
    if bool(OPEN_API_LOGIN) != bool(OPEN_API_PASSWORD):
        raise RuntimeError("OPEN_API_LOGIN and OPEN_API_PASSWORD must be set together")

    if not API_KEY and not (OPEN_API_LOGIN and OPEN_API_PASSWORD):
        raise RuntimeError("Either API_KEY or OPEN_API_LOGIN and OPEN_API_PASSWORD must be set")

    missing_env_vars = [
        name for name in REQUIRED_ENV_VARS if os.getenv(name) is None or os.getenv(name) == ""
    ]
    if not missing_env_vars:
        return

    logger.error("Empty required env vars: %s", ", ".join(missing_env_vars))
    raise RuntimeError(f"Empty required env vars: {', '.join(missing_env_vars)}")


validate_required_env()


def get_upstream_request_kwargs() -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    kwargs: dict[str, Any] = {"headers": headers}

    if OPEN_API_LOGIN and OPEN_API_PASSWORD:
        kwargs["auth"] = (OPEN_API_LOGIN, OPEN_API_PASSWORD)
        return kwargs

    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    return kwargs


# =============================================================================
# Schemas
# =============================================================================

class DateRange(BaseModel):
    from_: str = Field(alias="from")
    to: str


class Entities(BaseModel):
    people: list[str] | None = None
    emails: list[str] | None = None
    documents: list[str] | None = None
    names: list[str] | None = None
    links: list[str] | None = None


class Question(BaseModel):
    text: str
    asker: str = ""
    asked_on: str = ""
    variants: list[str] | None = None
    hyde: list[str] | None = None
    keywords: list[str] | None = None
    entities: Entities | None = None
    date_mentions: list[str] | None = None
    date_range: DateRange | None = None
    search_text: str = ""


class SearchAPIRequest(BaseModel):
    question: Question


class SearchAPIItem(BaseModel):
    message_ids: list[str]


class SearchAPIResponse(BaseModel):
    results: list[SearchAPIItem]


class DenseEmbeddingItem(BaseModel):
    index: int
    embedding: list[float]


class DenseEmbeddingResponse(BaseModel):
    data: list[DenseEmbeddingItem]


class SparseVector(BaseModel):
    indices: list[int] = Field(default_factory=list)
    values: list[float] = Field(default_factory=list)


class ChunkMetadata(BaseModel):
    chat_name: str
    chat_type: str
    chat_id: str
    chat_sn: str
    thread_sn: str | None = None
    message_ids: list[str]
    start: str
    end: str
    participants: list[str] = Field(default_factory=list)
    mentions: list[str] = Field(default_factory=list)
    contains_forward: bool = False
    contains_quote: bool = False


# =============================================================================
# Лемматизация — должна 1-в-1 совпадать с index/main.py
# =============================================================================

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
    """Лемматизация: кириллица → normal_form, латиница/цифры → lower без изменений.

    КРИТИЧНО: совпадает с index-service. Иначе BM25 не найдёт ничего
    (индекс хранит леммы, а запрос был бы в словоформе).
    """
    if not text:
        return ""
    tokens: list[str] = []
    for match in _WORD_RE.finditer(text):
        token = match.group(0).lower()
        if token.isdigit():
            tokens.append(token)
        elif _LATIN_ONLY.match(token):
            tokens.append(token)
        else:
            tokens.append(_lemma_ru(token))
    return " ".join(tokens)


def _email_to_pretty(email: str) -> str:
    """ivan.petrov@vk.com -> 'ivan petrov' — чтобы матчить sender'ов из индекса."""
    if not email:
        return ""
    local = email.split("@", 1)[0]
    return local.replace(".", " ").replace("_", " ").strip().lower()


# =============================================================================
# Sparse model (локально, для запросов)
# =============================================================================

@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    logger.info("Loading local sparse model %s", SPARSE_MODEL_NAME)
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    app.state.qdrant = AsyncQdrantClient(
        url=QDRANT_URL,
        api_key=API_KEY,
    )
    # warm-up: грузим BM25 и pymorphy3 ДО первого запроса,
    # чтобы не словить cold-start timeout.
    get_sparse_model()
    get_morph()
    try:
        yield
    finally:
        await app.state.http.aclose()
        await app.state.qdrant.close()


app = FastAPI(title="Search Service", version="0.3.0", lifespan=lifespan)


# =============================================================================
# Конфиг retrieval'а
# =============================================================================

MAX_VARIANTS = 2
MAX_HYDE = 1
MAX_QUERIES = 4
DENSE_PREFETCH_K = 40
SPARSE_PREFETCH_K = 60
RETRIEVE_K = 60
RERANK_N = 20
MAX_RESULT_IDS = 50


# =============================================================================
# Построение запросов
# =============================================================================

def wrap_dense_query(text: str) -> str:
    """Qwen3-Embedding query instruction format.

    Документы (которые индексирует тестирующая система) НЕ оборачиваются —
    это асимметричный retrieval, model trained это ожидать.
    """
    if not DENSE_QUERY_INSTRUCT_ENABLED or not text:
        return text
    return f"Instruct: {DENSE_INSTRUCT_TASK}\nQuery: {text}"


def build_dense_queries(question: Question) -> tuple[str, list[str]]:
    """Возвращает (primary_raw, dense_queries_ready_for_embed).

    primary_raw — сырой текст основного запроса, нужен для reranker'а.
    dense_queries — список, где primary+variants обёрнуты в Instruct,
    а hyde оставлены как есть (они эмулируют документный стиль).
    """
    primary_raw = (question.search_text.strip() or question.text.strip())

    wrapped: list[str] = []
    seen: set[str] = set()

    def _add(q: str, wrap: bool) -> None:
        q = q.strip() if q else ""
        if not q:
            return
        key = q.lower()
        if key in seen:
            return
        seen.add(key)
        wrapped.append(wrap_dense_query(q) if wrap else q)

    _add(primary_raw, wrap=True)

    for v in (question.variants or [])[:MAX_VARIANTS]:
        if isinstance(v, str):
            _add(v, wrap=True)

    for h in (question.hyde or [])[:MAX_HYDE]:
        if isinstance(h, str):
            _add(h, wrap=False)  # hyde = гипотетический документ, без Instruct

    # Отсечём лишнее, чтобы не бить в rate-limit dense API
    return primary_raw, wrapped[:MAX_QUERIES]


def build_sparse_query_text(question: Question) -> str:
    """Собирает богатый keyword-запрос для BM25 и лемматизирует.

    Источники: primary text + variants + keywords + entities.
    Entities.emails дополнительно раскрываем в pretty-form ('ivan petrov'),
    т.к. индекс рендерит sender'ов именно так.
    """
    parts: list[str] = []

    primary = (question.search_text.strip() or question.text.strip())
    if primary:
        parts.append(primary)

    for v in (question.variants or [])[:MAX_VARIANTS]:
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())

    for kw in (question.keywords or []):
        if isinstance(kw, str) and kw.strip():
            parts.append(kw.strip())

    if question.entities:
        for field in ("people", "documents", "names", "links"):
            vals = getattr(question.entities, field, None) or []
            for v in vals:
                if isinstance(v, str) and v.strip():
                    parts.append(v.strip())
        # emails: и сам email, и pretty-форма
        for em in (question.entities.emails or []):
            if not isinstance(em, str) or not em.strip():
                continue
            parts.append(em.strip())
            pretty = _email_to_pretty(em)
            if pretty:
                parts.append(pretty)

    # asker иногда важен: если в индексе есть фразы типа "Ivan Petrov пишет..."
    if question.asker:
        asker_pretty = _email_to_pretty(question.asker)
        if asker_pretty:
            parts.append(asker_pretty)

    combined = " ".join(parts)
    lemmatized = lemmatize_text(combined)
    return lemmatized if lemmatized else combined.lower()


# =============================================================================
# Embeddings
# =============================================================================

async def embed_dense_many(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    response = await client.post(
        EMBEDDINGS_DENSE_URL,
        **get_upstream_request_kwargs(),
        json={
            "model": os.getenv("EMBEDDINGS_DENSE_MODEL", EMBEDDINGS_DENSE_MODEL),
            "input": texts,
        },
    )
    response.raise_for_status()
    payload = DenseEmbeddingResponse.model_validate(response.json())
    if not payload.data:
        raise ValueError("Dense embedding response is empty")
    sorted_data = sorted(payload.data, key=lambda x: x.index)
    return [item.embedding for item in sorted_data]


async def embed_sparse(text: str) -> SparseVector:
    """Лемматизирует текст и прогоняет через локальный BM25."""
    prepared = lemmatize_text(text)
    if not prepared.strip():
        prepared = (text or "").strip().lower() or "пусто"

    vectors = list(get_sparse_model().embed([prepared]))
    if not vectors:
        raise ValueError("Sparse embedding response is empty")
    item = vectors[0]
    return SparseVector(
        indices=[int(index) for index in item.indices.tolist()],
        values=[float(value) for value in item.values.tolist()],
    )


# =============================================================================
# Qdrant retrieval
# =============================================================================

async def qdrant_search(
    client: AsyncQdrantClient,
    dense_vectors: list[list[float]],
    sparse_vector: SparseVector,
) -> list[Any]:
    prefetch: list[models.Prefetch] = []
    for dense_vector in dense_vectors:
        prefetch.append(
            models.Prefetch(
                query=dense_vector,
                using=QDRANT_DENSE_VECTOR_NAME,
                limit=DENSE_PREFETCH_K,
            )
        )
    prefetch.append(
        models.Prefetch(
            query=models.SparseVector(
                indices=sparse_vector.indices,
                values=sparse_vector.values,
            ),
            using=QDRANT_SPARSE_VECTOR_NAME,
            limit=SPARSE_PREFETCH_K,
        )
    )

    response = await client.query_points(
        collection_name=QDRANT_COLLECTION_NAME,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=RETRIEVE_K,
        with_payload=True,
    )

    return list(response.points or [])


def extract_message_ids(point: Any) -> list[str]:
    payload = point.payload or {}
    metadata = payload.get("metadata") or {}
    message_ids = metadata.get("message_ids") or payload.get("message_ids") or []
    return [str(message_id) for message_id in message_ids]


# =============================================================================
# Reranker
# =============================================================================

async def get_rerank_scores(
    client: httpx.AsyncClient,
    label: str,
    targets: list[str],
) -> list[float]:
    if not targets:
        return []

    response = await client.post(
        RERANKER_URL,
        **get_upstream_request_kwargs(),
        json={
            "model": RERANKER_MODEL,
            "encoding_format": "float",
            "text_1": label,
            "text_2": targets,
        },
    )
    response.raise_for_status()

    payload = response.json()
    data = payload.get("data") or []
    return [float(sample["score"]) for sample in data]


async def rerank_points(
    client: httpx.AsyncClient,
    query: str,
    points: list[Any],
) -> list[Any]:
    if len(points) <= 1:
        return points

    head = points[:RERANK_N]
    tail = points[RERANK_N:]
    targets = [(p.payload or {}).get("page_content") or "" for p in head]

    try:
        scores = await get_rerank_scores(client, query, targets)
    except Exception as exc:
        logger.warning("Rerank failed, falling back to RRF order: %s", exc)
        return points

    if len(scores) != len(head):
        logger.warning(
            "Rerank returned %d scores for %d targets, falling back",
            len(scores),
            len(head),
        )
        return points

    reranked = [
        point
        for _, point in sorted(
            zip(scores, head),
            key=lambda item: item[0],
            reverse=True,
        )
    ]
    return reranked + tail


def flatten_message_ids(points: list[Any], limit: int) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for point in points:
        for mid in extract_message_ids(point):
            if mid in seen:
                continue
            seen.add(mid)
            result.append(mid)
            if len(result) >= limit:
                return result
    return result


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/search", response_model=SearchAPIResponse)
async def search(payload: SearchAPIRequest) -> SearchAPIResponse:
    question = payload.question

    primary_raw, dense_queries = build_dense_queries(question)
    if not primary_raw:
        raise HTTPException(status_code=400, detail="question.text is required")

    sparse_query_text = build_sparse_query_text(question)

    client: httpx.AsyncClient = app.state.http
    qdrant: AsyncQdrantClient = app.state.qdrant

    # Dense: пробуем multi-query, при сбое (rate limit, timeout) — только primary
    try:
        dense_vectors = await embed_dense_many(client, dense_queries)
    except Exception as exc:
        logger.warning(
            "Multi-query dense failed (%s), retrying with primary only", exc
        )
        dense_vectors = await embed_dense_many(client, [dense_queries[0]])

    # Sparse: обогащённый лемматизированный запрос (keywords + entities + text)
    sparse_vector = await embed_sparse(sparse_query_text)

    points = await qdrant_search(qdrant, dense_vectors, sparse_vector)

    if not points:
        return SearchAPIResponse(results=[SearchAPIItem(message_ids=[])])

    # Rerank: передаём reranker'у СЫРОЙ вопрос (без Instruct-обёртки) —
    # nvidia/llama-nemotron-rerank-1b-v2 ожидает естественный текст.
    points = await rerank_points(client, primary_raw, points)
    message_ids = flatten_message_ids(points, MAX_RESULT_IDS)

    return SearchAPIResponse(results=[SearchAPIItem(message_ids=message_ids)])


@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(exc)
    detail = str(exc) or repr(exc)

    if isinstance(exc, RequestValidationError):
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    if isinstance(exc, HTTPException):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    return JSONResponse(status_code=500, content={"detail": detail})


def main() -> None:
    import uvicorn

    uvicorn.run(
        "main:app",
        host=HOST,
        port=PORT,
        reload=False,
    )


if __name__ == "__main__":
    main()