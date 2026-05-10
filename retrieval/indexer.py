from pathlib import Path
import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_PATH = ROOT / "data" / "embeddings.npy"
INDEX_PATH = ROOT / "data" / "data_index"

embeddings = np.load(EMBEDDINGS_PATH)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

faiss.write_index(index, str(INDEX_PATH))

print("Index created.")
print("Total vectors:", index.ntotal)
