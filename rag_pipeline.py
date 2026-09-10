import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer


print("Loading RAG components...")


# 1. Load BM25
with open("bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)


# 2. Load FAISS index
faiss_index = faiss.read_index("faiss_index.bin")


# 3. Load ticket data
with open("ticket_data.pkl", "rb") as f:
    ticket_data = pickle.load(f)


# 4. Load SAME model used to create FAISS index
model = SentenceTransformer("BAAI/bge-m3")


print("RAG components loaded successfully!")


# 5. FAISS Retrieval
def retrieve_documents(query, top_k=5):

    # Convert query to embedding
    query_embedding = model.encode(
        [query],
        normalize_embeddings=False
    )

    # Convert to float32
    query_embedding = np.array(query_embedding).astype("float32")

    # Normalize query
    faiss.normalize_L2(query_embedding)

    # Search FAISS
    distances, indices = faiss_index.search(
        query_embedding,
        top_k
    )

    results = []

    # Get ticket information
    for idx in indices[0]:

        if idx != -1 and idx < len(ticket_data):

            ticket = ticket_data.iloc[idx]

            results.append({
                "ID": ticket["ID"],
                "Title": ticket["Title"],
                "Description": ticket["Description"],
                "Resolution": ticket["Resolution Comments"],
                "Priority": ticket["Priority"],
                "Status": ticket["Call Status"]
            })

    return results


# 6. Test query
query = "User has been unlocked"

print("\nSearching for:", query)


# 7. Retrieve top 5 tickets
results = retrieve_documents(query, top_k=5)


# 8. Display results
print("\nRetrieved Documents:")

for i, result in enumerate(results):

    print(f"\n========== Result {i + 1} ==========")

    print("ID:", result["ID"])
    print("Title:", result["Title"])
    print("Description:", result["Description"])
    print("Resolution:", result["Resolution"])
    print("Priority:", result["Priority"])
    print("Status:", result["Status"])