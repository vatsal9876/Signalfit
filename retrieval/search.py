import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from embeddings import model

catalog = json.load(open("data/shl_catalog.json", "r", encoding="utf-8"))

index = faiss.read_index("data/data_index")


def build_query(state):

    return f"""
    Role:
    {state.get("role", "")}

    Seniority:
    {state.get("seniority", "")}

    Requirements:
    {", ".join(state.get("requirements", []))}
    """


def retrieve(state,k=10):

    query = build_query(state)

    query_embedding = model.encode([query])

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    distances, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for idx in indices[0]:

        item = catalog[idx]

        results.append({
            "name": item["name"],
            "url": item["link"],
            "keys": item["keys"]
        })

    return results


if __name__ == "__main__":

    state = {
        "role": "executive leadership",
        "seniority": "director",
        "requirements": [
            "benchmarking",
            "personality evaluation"
        ]
    }

    results = retrieve(state)

    for i, item in enumerate(results, start=1):

        print(f"\n{i}. {item['name']}")
        print(item["keys"])
        print(item["url"])