"""End-to-end тест локально поднятого решения на data/Go Nova.json.

Повторяет конвейер проверяющей системы:
  1. читает чат и сообщения из data/Go Nova.json
  2. дёргает POST /index нашего index-service → получает чанки
  3. зовёт внешний dense API (Qwen/Qwen3-Embedding-0.6B) через Basic Auth
  4. зовёт наш /sparse_embedding для sparse-векторов
  5. апсертит точки в Qdrant с метадатой, совместимой с ChunkMetadata
  6. дёргает POST /search на пару тестовых вопросов и печатает результаты

Запуск:
    export OPEN_API_LOGIN=... OPEN_API_PASSWORD=...
    python3 scripts/e2e_test.py
"""
from __future__ import annotations

import base64
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

INDEX_URL = os.getenv("INDEX_URL", "http://localhost:8001")
SEARCH_URL = os.getenv("SEARCH_URL", "http://localhost:8002")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_NAME", "evaluation")
DENSE_URL = os.getenv("EMBEDDINGS_DENSE_URL", "http://83.166.249.64:18001/embeddings")
DENSE_MODEL = os.getenv("EMBEDDINGS_DENSE_MODEL", "Qwen/Qwen3-Embedding-0.6B")

OPEN_API_LOGIN = os.getenv("OPEN_API_LOGIN")
OPEN_API_PASSWORD = os.getenv("OPEN_API_PASSWORD")

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Go Nova.json"

DENSE_BATCH = 16
QDRANT_BATCH = 64


def http_json(method: str, url: str, body: dict | None = None, headers: dict | None = None, timeout: int = 120) -> dict:
    data = None
    hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
    if headers:
        hdrs.update(headers)
    if body is not None:
        data = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code} from {method} {url}: {e.read().decode('utf-8', errors='ignore')}")
    except urllib.error.URLError as e:
        sys.exit(f"URL error on {method} {url}: {e}")
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def basic_auth_header(login: str, password: str) -> dict:
    token = base64.b64encode(f"{login}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def load_chat() -> tuple[dict, list[dict]]:
    with open(DATA_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    return payload["chat"], payload["messages"]


def build_chunks(chat: dict, messages: list[dict]) -> list[dict]:
    print(f"[1/5] POST {INDEX_URL}/index with {len(messages)} messages")
    resp = http_json(
        "POST",
        f"{INDEX_URL}/index",
        body={
            "data": {
                "chat": chat,
                "overlap_messages": [],
                "new_messages": messages,
            }
        },
    )
    chunks = resp.get("results", [])
    print(f"      → {len(chunks)} chunks")
    return chunks


def compute_dense(texts: list[str]) -> list[list[float]]:
    if not OPEN_API_LOGIN or not OPEN_API_PASSWORD:
        sys.exit("OPEN_API_LOGIN / OPEN_API_PASSWORD must be set for dense API")
    headers = basic_auth_header(OPEN_API_LOGIN, OPEN_API_PASSWORD)
    vectors: list[list[float]] = []
    for i in range(0, len(texts), DENSE_BATCH):
        batch = texts[i : i + DENSE_BATCH]
        print(f"[2/5] dense batch {i // DENSE_BATCH + 1}: {len(batch)} texts")
        resp = http_json(
            "POST",
            DENSE_URL,
            body={"model": DENSE_MODEL, "input": batch},
            headers=headers,
            timeout=180,
        )
        data = sorted(resp.get("data", []), key=lambda x: x["index"])
        if len(data) != len(batch):
            sys.exit(f"dense API returned {len(data)} items for {len(batch)} inputs")
        vectors.extend(item["embedding"] for item in data)
    print(f"      → {len(vectors)} dense vectors (dim {len(vectors[0]) if vectors else 0})")
    return vectors


def compute_sparse(texts: list[str]) -> list[dict]:
    print(f"[3/5] POST {INDEX_URL}/sparse_embedding with {len(texts)} texts")
    resp = http_json("POST", f"{INDEX_URL}/sparse_embedding", body={"texts": texts}, timeout=180)
    vectors = resp.get("vectors", [])
    print(f"      → {len(vectors)} sparse vectors")
    return vectors


def build_points(chunks: list[dict], messages: list[dict], chat: dict, dense: list[list[float]], sparse: list[dict]) -> list[dict]:
    by_id = {m["id"]: m for m in messages}
    points: list[dict] = []
    for i, (chunk, dense_vec, sparse_vec) in enumerate(zip(chunks, dense, sparse)):
        chunk_msgs = [by_id[mid] for mid in chunk["message_ids"] if mid in by_id]
        times = [m.get("time", 0) for m in chunk_msgs]
        participants = sorted({m.get("sender_id", "") for m in chunk_msgs if m.get("sender_id")})
        mentions = sorted({mention for m in chunk_msgs for mention in (m.get("mentions") or [])})
        points.append(
            {
                "id": i,
                "vector": {
                    "dense": dense_vec,
                    "sparse": {
                        "indices": sparse_vec["indices"],
                        "values": sparse_vec["values"],
                    },
                },
                "payload": {
                    "page_content": chunk["page_content"],
                    "metadata": {
                        "chat_name": chat["name"],
                        "chat_type": chat["type"],
                        "chat_id": chat["id"],
                        "chat_sn": chat["sn"],
                        "message_ids": chunk["message_ids"],
                        "start": str(min(times) if times else 0),
                        "end": str(max(times) if times else 0),
                        "participants": participants,
                        "mentions": mentions,
                        "contains_forward": any(m.get("is_forward") for m in chunk_msgs),
                        "contains_quote": any(m.get("is_quote") for m in chunk_msgs),
                    },
                },
            }
        )
    return points


def upsert_qdrant(points: list[dict]) -> None:
    print(f"[4/5] upserting {len(points)} points to {QDRANT_URL}/collections/{QDRANT_COLLECTION}")
    for i in range(0, len(points), QDRANT_BATCH):
        batch = points[i : i + QDRANT_BATCH]
        http_json(
            "PUT",
            f"{QDRANT_URL}/collections/{QDRANT_COLLECTION}/points?wait=true",
            body={"points": batch},
            timeout=120,
        )
        print(f"      → upserted {min(i + QDRANT_BATCH, len(points))}/{len(points)}")


def run_searches(questions: list[str], messages: list[dict]) -> None:
    by_id = {m["id"]: m for m in messages}
    print(f"[5/5] running /search on {len(questions)} test questions\n")
    for q in questions:
        resp = http_json("POST", f"{SEARCH_URL}/search", body={"question": {"text": q}}, timeout=120)
        results = resp.get("results", [])
        msg_ids = results[0].get("message_ids", []) if results else []
        print(f"━━━ Q: {q}")
        print(f"    found {len(msg_ids)} message_ids")
        for mid in msg_ids[:5]:
            msg = by_id.get(mid)
            if not msg:
                print(f"    [{mid}] <not found in source>")
                continue
            text = (msg.get("text") or "").strip()
            if not text and msg.get("parts"):
                for part in msg["parts"]:
                    if isinstance(part.get("text"), str) and part.get("text"):
                        text = part["text"].strip()
                        break
            preview = text.replace("\n", " ")[:180]
            print(f"    [{mid}] {preview}")
        print()


def main() -> None:
    chat, messages = load_chat()
    print(f"chat: {chat['name']} ({chat['type']}), {len(messages)} messages\n")

    chunks = build_chunks(chat, messages)
    if not chunks:
        sys.exit("index returned no chunks")

    dense = compute_dense([c["dense_content"] for c in chunks])
    sparse = compute_sparse([c["sparse_content"] for c in chunks])

    points = build_points(chunks, messages, chat, dense, sparse)
    upsert_qdrant(points)

    questions = [
        "Что обсуждали про релиз Go 1.18?",
        "Когда митап DC Backend Tech Talk про Go?",
        "Есть проблемы с syscall или SIGABRT?",
        "Кто отвечает за DevRel в сообществе Go?",
    ]
    run_searches(questions, messages)


if __name__ == "__main__":
    main()
