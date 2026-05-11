# retrieval/embeddings.py

import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data" / "shl_catalog.json"
EMBEDDINGS_PATH = ROOT / "data" / "embeddings.npy"

model = None


def get_model():
    global model

    if model is not None:
        return model

    try:
        model = SentenceTransformer(MODEL_NAME, local_files_only=True)
    except Exception:
        model = None

    return model


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


def generate_embeddings(documents):

    embedding_model = get_model()

    if embedding_model is None:
        raise RuntimeError(
            "Embedding model is not available locally. "
            "Install/cache all-MiniLM-L6-v2 before generating embeddings."
        )

    embeddings = embedding_model.encode(documents,show_progress_bar=True)

    return np.array(embeddings).astype("float32")



if __name__ == "__main__":

    catalog = load_catalog()

    documents = generate_documents(catalog)

    embeddings = generate_embeddings(documents)

    save_embeddings(embeddings)

    print("Embeddings saved.")
    print("Shape:", embeddings.shape)
