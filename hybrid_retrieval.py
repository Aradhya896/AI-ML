import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer


# -----------------------------------------
# 1. Load indexes
# -----------------------------------------

print("Loading indexes...")

with open("bm25_index.pkl", "rb") as f:
    bm25 = pickle.load(f)

faiss_index = faiss.read_index("faiss_index.bin")

df = pd.read_pickle("ticket_data.pkl")

model = SentenceTransformer("BAAI/bge-m3")

print("All indexes and model loaded successfully!")
print("Total tickets:", len(df))


# -----------------------------------------
# 2. Hybrid Search
# -----------------------------------------

def hybrid_search(query, top_k=5):

    # =====================================
    # BM25 SEARCH
    # =====================================

    tokenized_query = query.lower().split()

    bm25_scores = bm25.get_scores(tokenized_query)

    # Get more candidates
    bm25_candidates = np.argsort(bm25_scores)[::-1][:20]


    # =====================================
    # BGE-M3 + FAISS SEARCH
    # =====================================

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(query_embedding)

    faiss_scores, faiss_indices = faiss_index.search(
        query_embedding,
        20
    )


    # =====================================
    # NORMALIZE BM25 SCORES
    # =====================================

    bm25_selected_scores = bm25_scores[bm25_candidates]

    if bm25_selected_scores.max() > 0:
        bm25_normalized = (
            bm25_selected_scores /
            bm25_selected_scores.max()
        )
    else:
        bm25_normalized = bm25_selected_scores


    # =====================================
    # CREATE SCORE DICTIONARY
    # =====================================

    hybrid_scores = {}

    # BM25 contribution
    for i, index in enumerate(bm25_candidates):

        hybrid_scores[index] = (
            0.5 * bm25_normalized[i]
        )


    # FAISS contribution
    faiss_max = faiss_scores[0].max()

    for i, index in enumerate(faiss_indices[0]):

        vector_score = faiss_scores[0][i]

        if faiss_max > 0:
            vector_score = vector_score / faiss_max

        hybrid_scores[index] = (
            hybrid_scores.get(index, 0)
            + 0.5 * vector_score
        )


    # =====================================
    # FINAL RANKING
    # =====================================

    ranked_results = sorted(
        hybrid_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]


    # =====================================
    # DISPLAY RESULTS
    # =====================================

    print("\n" + "=" * 70)
    print("QUERY:", query)
    print("=" * 70)

    print("\nTop Hybrid Retrieval Results:\n")

    for rank, (index, score) in enumerate(
        ranked_results,
        start=1
    ):

        ticket = df.iloc[index]

        print(f"Result #{rank}")
        print("-" * 60)

        print("Hybrid Score:", round(score, 4))
        print("Ticket ID:", ticket["ID"])
        print("Classification:", ticket["Classification"])
        print("Group:", ticket["Group"])

        print("\nDescription:")
        print(ticket["Description"])

        print()


# -----------------------------------------
# 3. User Query
# -----------------------------------------

query = input("\nEnter your ITSM problem: ")

hybrid_search(query)