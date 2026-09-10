import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

print("Loading training data...")

# 1. Load train dataset
df = pd.read_csv("train_data.csv")

print("Dataset shape:", df.shape)

# 2. Use ONLY Description
df["Description"] = df["Description"].fillna("").astype(str)

documents = df["Description"].tolist()

print("Total descriptions:", len(documents))


# ------------------------------------------------
# 3. Create BM25 index
# ------------------------------------------------

print("\nCreating BM25 index...")

tokenized_documents = [
    document.lower().split()
    for document in documents
]

bm25 = BM25Okapi(tokenized_documents)

with open("bm25_index.pkl", "wb") as f:
    pickle.dump(bm25, f)

print("BM25 index saved successfully.")


# ------------------------------------------------
# 4. Load BGE-M3
# ------------------------------------------------

print("\nLoading BGE-M3 model...")

model = SentenceTransformer("BAAI/bge-m3")

print("BGE-M3 loaded successfully.")


# ------------------------------------------------
# 5. Create vector embeddings
# ------------------------------------------------

print("\nCreating embeddings...")

embeddings = model.encode(
    documents,
    batch_size=16,
    show_progress_bar=True,
    convert_to_numpy=True
)

embeddings = embeddings.astype("float32")

print("Embeddings shape:", embeddings.shape)


# ------------------------------------------------
# 6. Normalize embeddings
# ------------------------------------------------

faiss.normalize_L2(embeddings)


# ------------------------------------------------
# 7. Create FAISS index
# ------------------------------------------------

print("\nCreating FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("FAISS index created.")
print("Total vectors:", index.ntotal)
print("Vector dimension:", dimension)


# ------------------------------------------------
# 8. Save FAISS index
# ------------------------------------------------

faiss.write_index(
    index,
    "faiss_index.bin"
)

print("FAISS index saved successfully.")


# ------------------------------------------------
# 9. Save ticket data
# ------------------------------------------------

df.to_pickle("ticket_data.pkl")

print("Ticket data saved successfully.")

print("\n===================================")
print("INDEXING COMPLETED SUCCESSFULLY")
print("===================================")