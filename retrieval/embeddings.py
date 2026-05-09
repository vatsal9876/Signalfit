# retrieval/embeddings.py

import json
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


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


def generate_embeddings(documents):

    embeddings = model.encode(documents,show_progress_bar=True)

    return np.array(embeddings).astype("float32")



if __name__ == "__main__":

    catalog = load_catalog()

    documents = generate_documents(catalog)

    embeddings = generate_embeddings(documents)

    save_embeddings(embeddings)

    print("Embeddings saved.")
    print("Shape:", embeddings.shape)