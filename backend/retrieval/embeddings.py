import json
import os
from urllib import error, request
from pathlib import Path
import numpy as np
from dotenv import load_dotenv

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
load_dotenv(ENV_PATH, override=True)

MODEL_NAME = os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v3")
JINA_API_URL = os.getenv("JINA_EMBEDDING_URL", "https://api.jina.ai/v1/embeddings")
JINA_DIMENSIONS = int(os.getenv("JINA_EMBEDDING_DIMENSIONS", "384"))
JINA_BATCH_SIZE = int(os.getenv("JINA_EMBEDDING_BATCH_SIZE", "64"))
ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "shl_catalog.json"
EMBEDDINGS_PATH = ROOT / "data" / "embeddings.npy"


def _jina_api_key():
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        raise RuntimeError("JINA_API_KEY is required for Jina embeddings.")
    return api_key


def build_document(item):

    return f"""

    Assessment Name: {item.get("name", "")}

    Description: {item.get("description", "")}

    Suitable Job Levels: {", ".join(item.get("job_levels", []))}

    Assessment Categories: {", ".join(item.get("keys", []))}

    Assesment Languages: {", ".join(item.get("languages", []))}

    Assessment Duration: {item.get("duration", "")}

    remote: {item.get("remote", "")}

    Adaptive: {item.get("adaptive", "")}

    """


def generate_documents(catalog):

    return [build_document(item) for item in catalog]


def load_catalog():
    return json.load(open(CATALOG_PATH, "r", encoding="utf-8"))


def save_embeddings(embeddings):
    np.save(EMBEDDINGS_PATH, embeddings)


def _jina_payload(texts, task):
    payload = {
        "model": MODEL_NAME,
        "input": texts,
        "task": task,
    }

    if JINA_DIMENSIONS:
        payload["dimensions"] = JINA_DIMENSIONS

    return payload


def _request_embeddings(texts, task):
    if not texts:
        return np.empty((0, JINA_DIMENSIONS), dtype="float32")

    payload = json.dumps(_jina_payload(texts, task)).encode("utf-8")
    req = request.Request(
        JINA_API_URL,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {_jina_api_key()}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "signalfit/0.1",
        },
    )

    try:
        with request.urlopen(req, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Jina embedding request failed: {exc.code} {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Jina embedding request failed: {exc.reason}") from exc

    rows = sorted(body.get("data", []), key=lambda item: item.get("index", 0))
    embeddings = [row.get("embedding") for row in rows]

    if len(embeddings) != len(texts) or any(embedding is None for embedding in embeddings):
        raise RuntimeError("Jina embedding response did not include all embeddings.")

    return np.array(embeddings).astype("float32")


def embed_query(text):
    embeddings = _request_embeddings([text], task="retrieval.query")
    return embeddings


def embed_documents(documents):
    batches = []

    for start in range(0, len(documents), JINA_BATCH_SIZE):
        batch = documents[start : start + JINA_BATCH_SIZE]
        batches.append(_request_embeddings(batch, task="retrieval.passage"))

    if not batches:
        return np.empty((0, JINA_DIMENSIONS), dtype="float32")

    return np.vstack(batches).astype("float32")


def generate_embeddings(documents):
    return embed_documents(documents)


if __name__ == "__main__":
    try:
        catalog = load_catalog()
        documents = generate_documents(catalog)
        embeddings = generate_embeddings(documents)
        save_embeddings(embeddings)
    except RuntimeError as exc:
        raise SystemExit(f"Could not generate embeddings: {exc}") from exc

    print("Embeddings saved.")
    print("Shape:", embeddings.shape)
