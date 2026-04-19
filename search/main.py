import asyncio
import logging
import os
import random
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

# --- Qwen3 Instruct (асимметричная обёртка query, см. предыдущую итерацию) ---
DENSE_QUERY_INSTRUCT_ENABLED = os.getenv("DENSE_QUERY_INSTRUCT", "1") not in ("0", "false", "False", "")
DENSE_INSTRUCT_TASK = os.getenv(
    "DENSE_INSTRUCT_TASK",
    "Given a user question, retrieve relevant chat messages that answer the question.",
)

# --- Retrieval (всё через env — удобно крутить без пересборки) ---
MAX_VARIANTS = int(os.getenv("MAX_VARIANTS", "2"))
MAX_HYDE = int(os.getenv("MAX_HYDE", "1"))
MAX_QUERIES = int(os.getenv("MAX_QUERIES", "4"))
DENSE_PREFETCH_K = int(os.getenv("DENSE_PREFETCH_K", "40"))
SPARSE_PREFETCH_K = int(os.getenv("SPARSE_PREFETCH_K", "60"))
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "60"))
RERANK_N = int(os.getenv("RERANK_N", "20"))
MAX_RESULT_IDS = int(os.getenv("MAX_RESULT_IDS", "50"))

# --- Троттлинг upstream (dense + rerank): главный рычаг против 429 ---
UPSTREAM_CONCURRENCY = int(os.getenv("UPSTREAM_CONCURRENCY", "2"))
UPSTREAM_MAX_RETRIES = int(os.getenv("UPSTREAM_MAX_RETRIES", "4"))
UPSTREAM_INITIAL_BACKOFF = float(os.getenv("UPSTREAM_INITIAL_BACKOFF", "0.6"))
UPSTREAM_MAX_BACKOFF = float(os.getenv("UPSTREAM_MAX_BACKOFF", "10.0"))
UPSTREAM_JITTER = float(os.getenv("UPSTREAM_JITTER", "0.3"))
UPSTREAM_HTTP_TIMEOUT = float(os.getenv("UPSTREAM_HTTP_TIMEOUT", "45.0"))

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
    if not email:
        return ""
    local = email.split("@", 1)[0]
    return local.replace(".", " ").replace("_", " ").strip().lower()


@lru_cache(maxsize=1)
def get_sparse_model() -> SparseTextEmbedding:
    logger.info("Loading local sparse model %s", SPARSE_MODEL_NAME)
    return SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient(timeout=httpx.Timeout(UPSTREAM_HTTP_TIMEOUT))
    app.state.qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=API_KEY)
    # Semaphore нужно создавать в running loop → здесь, в lifespan.
    app.state.upstream_sem = asyncio.Semaphore(UPSTREAM_CONCURRENCY)
    # warm-up, чтобы cold-start не съел SLA первого запроса
    get_sparse_model()
    get_morph()
    try:
        yield
    finally:
        await app.state.http.aclose()
        await app.state.qdrant.close()


app = FastAPI(title="Search Service", version="0.4.0", lifespan=lifespan)


_RETRYABLE_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


def _parse_retry_after(value: str | None) -> float | None:
    """Retry-After может быть и числом, и HTTP-датой — берём только число."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


async def upstream_post(
    client: httpx.AsyncClient,
    url: str,
    sem: asyncio.Semaphore,
    json_payload: dict[str, Any],
    op_label: str,
) -> httpx.Response:
    """POST к upstream с адаптивным retry.

    * Семафор — конкурентность внутри воркера: сглаживает burst при параллельной оценке.
    * На 429 уважаем Retry-After, если пришёл.
    * Exponential backoff с jitter: разносим retry-волны разных запросов.
    * 5xx тоже ретраим — upstream мог просто перегреться.
    """
    backoff = UPSTREAM_INITIAL_BACKOFF
    last_exc: Exception | None = None

    for attempt in range(UPSTREAM_MAX_RETRIES + 1):
        response: httpx.Response | None = None
        status: int | None = None
        retry_after_sec: float | None = None

        try:
            async with sem:
                response = await client.post(
                    url,
                    **get_upstream_request_kwargs(),
                    json=json_payload,
                )
            status = response.status_code
        except httpx.TimeoutException as exc:
            last_exc = exc
            logger.warning(
                "%s: timeout on attempt %d/%d",
                op_label, attempt + 1, UPSTREAM_MAX_RETRIES + 1,
            )
        except httpx.HTTPError as exc:
            last_exc = exc
            logger.warning(
                "%s: httpx error on attempt %d: %s", op_label, attempt + 1, exc
            )

        # Успех
        if response is not None and status == 200:
            return response

        # Не-retryable статус — сразу выбрасываем
        if response is not None and status not in _RETRYABLE_STATUSES:
            response.raise_for_status()
            return response  # unreachable

        if response is not None:
            retry_after_sec = _parse_retry_after(response.headers.get("Retry-After"))
            last_exc = httpx.HTTPStatusError(
                f"{op_label}: {status}",
                request=response.request,
                response=response,
            )

        # Попытки кончились → вверх
        if attempt >= UPSTREAM_MAX_RETRIES:
            logger.warning("%s: retries exhausted (status=%s)", op_label, status)
            if last_exc:
                raise last_exc
            raise RuntimeError(f"{op_label}: retries exhausted")

        # Считаем паузу
        wait = backoff
        if retry_after_sec is not None:
            wait = max(wait, retry_after_sec)
        wait = min(wait, UPSTREAM_MAX_BACKOFF)
        jittered = wait + random.uniform(0.0, wait * UPSTREAM_JITTER)

        logger.warning(
            "%s: attempt %d failed (status=%s), backing off %.2fs",
            op_label, attempt + 1, status, jittered,
        )
        await asyncio.sleep(jittered)
        backoff = min(backoff * 2, UPSTREAM_MAX_BACKOFF)

    raise RuntimeError(f"{op_label}: unexpected retry exit")


def wrap_dense_query(text: str) -> str:
    if not DENSE_QUERY_INSTRUCT_ENABLED or not text:
        return text
    return f"Instruct: {DENSE_INSTRUCT_TASK}\nQuery: {text}"


def build_dense_queries(question: Question) -> tuple[str, list[str]]:
    """→ (primary_raw для rerank, список dense-текстов для embed)."""
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
            _add(h, wrap=False)  # hyde — документный стиль, без Instruct

    return primary_raw, wrapped[:MAX_QUERIES]


def build_sparse_query_text(question: Question) -> str:
    """Лемматизированный keyword-rich текст: text + variants + keywords + entities."""
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
        for em in (question.entities.emails or []):
            if not isinstance(em, str) or not em.strip():
                continue
            parts.append(em.strip())
            pretty = _email_to_pretty(em)
            if pretty:
                parts.append(pretty)

    if question.asker:
        asker_pretty = _email_to_pretty(question.asker)
        if asker_pretty:
            parts.append(asker_pretty)

    combined = " ".join(parts)
    lemmatized = lemmatize_text(combined)
    return lemmatized if lemmatized else combined.lower()


async def embed_dense_many(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    texts: list[str],
) -> list[list[float]]:
    if not texts:
        return []
    response = await upstream_post(
        client,
        EMBEDDINGS_DENSE_URL,
        sem=sem,
        json_payload={
            "model": os.getenv("EMBEDDINGS_DENSE_MODEL", EMBEDDINGS_DENSE_MODEL),
            "input": texts,
        },
        op_label="dense",
    )
    payload = DenseEmbeddingResponse.model_validate(response.json())
    if not payload.data:
        raise ValueError("Dense embedding response is empty")
    sorted_data = sorted(payload.data, key=lambda x: x.index)
    return [item.embedding for item in sorted_data]


async def embed_sparse(text: str) -> SparseVector:
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


async def get_rerank_scores(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    label: str,
    targets: list[str],
) -> list[float]:
    if not targets:
        return []

    response = await upstream_post(
        client,
        RERANKER_URL,
        sem=sem,
        json_payload={
            "model": RERANKER_MODEL,
            "encoding_format": "float",
            "text_1": label,
            "text_2": targets,
        },
        op_label="rerank",
    )
    payload = response.json()
    data = payload.get("data") or []
    return [float(sample["score"]) for sample in data]


async def rerank_points(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    query: str,
    points: list[Any],
) -> list[Any]:
    if len(points) <= 1:
        return points

    head = points[:RERANK_N]
    tail = points[RERANK_N:]
    targets = [(p.payload or {}).get("page_content") or "" for p in head]

    try:
        scores = await get_rerank_scores(client, sem, query, targets)
    except Exception as exc:
        logger.warning("Rerank failed after retries, falling back to RRF: %s", exc)
        return points

    if len(scores) != len(head):
        logger.warning(
            "Rerank returned %d scores for %d targets, falling back",
            len(scores), len(head),
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
    sem: asyncio.Semaphore = app.state.upstream_sem

    # Dense: 3-уровневый fallback
    #   1) multi-query (primary + variants + hyde) с retry
    #   2) primary only с retry
    #   3) sparse-only retrieval (полностью без dense)
    # Даже если dense упёрся в лимит — запрос не падает 500-кой, а отдаёт что смог.
    dense_vectors: list[list[float]] = []
    try:
        dense_vectors = await embed_dense_many(client, sem, dense_queries)
    except Exception as exc:
        logger.warning("Multi-query dense failed (%s), retrying with primary only", exc)
        try:
            dense_vectors = await embed_dense_many(
                client, sem, [dense_queries[0]] if dense_queries else []
            )
        except Exception as exc2:
            logger.warning(
                "Primary dense also failed (%s), continuing with sparse-only", exc2
            )
            dense_vectors = []

    sparse_vector = await embed_sparse(sparse_query_text)

    points = await qdrant_search(qdrant, dense_vectors, sparse_vector)

    if not points:
        return SearchAPIResponse(results=[SearchAPIItem(message_ids=[])])

    points = await rerank_points(client, sem, primary_raw, points)
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