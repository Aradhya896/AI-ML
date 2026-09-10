import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer


# -----------------------------
# 1. Load saved indexes/data
# -----------------------------

print("Loading indexes...")

with open("bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)

index = faiss.read_index("faiss_index.bin")

df = pd.read_pickle("ticket_data.pkl")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Indexes loaded successfully!")
print("Total tickets:", len(df))


# -----------------------------
# 2. Search function
# -----------------------------

def search_tickets(query, top_k=5):

    # ----- FAISS semantic search -----

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    # ----- Display results -----

    print("\n" + "=" * 70)
    print("QUERY:", query)
    print("=" * 70)

    for rank, idx in enumerate(indices[0], start=1):

        ticket = df.iloc[idx]

        print(f"\nResult #{rank}")
        print("-" * 50)

        print("Ticket ID:", ticket["ID"])
        print("Classification:", ticket["Classification"])
        print("Group:", ticket["Group"])
        print("Service Category:", ticket["Service Category"])
        print("Priority:", ticket["Priority"])

        print("\nDescription:")
        print(ticket["Description"])

        print("\nResolution:")
        print(ticket["Resolution Comments"])

        print("\nSimilarity Score:",
              round(float(scores[0][rank - 1]), 4))


# -----------------------------
# 3. Test query
# -----------------------------

query = input("\nEnter your ITSM problem: ")

search_tickets(query)