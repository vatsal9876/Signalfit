import faiss
import numpy as np

embeddings = np.load("data/embeddings.npy")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

faiss.write_index(index,"data/shl.index")

print("Index created.")
print("Total vectors:", index.ntotal)