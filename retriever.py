import chromadb
from sentence_transformers import SentenceTransformer

embedder   = SentenceTransformer("all-MiniLM-L6-v2")
client     = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("my_docs")


def retrieve(query: str, top_k: int = 3) -> list[dict]:
    """
    Given a question, return the top_k most relevant document chunks.
    Each result has: title, content, and similarity score.
    """
    # 1. Embed the query using the same model used for indexing
    query_vector = embedder.encode(query).tolist()

    # 2. Search ChromaDB for similar vectors
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    # 3. Package results into a clean list
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "title":   results["metadatas"][0][i]["title"],
            "content": results["documents"][0][i],
            "score":   round(1 - results["distances"][0][i], 3)  # convert distance → similarity
        })

    return chunks


# Quick test — run this file directly to check retrieval works
if __name__ == "__main__":
    query = "How does similarity search work?"
    hits = retrieve(query)

    print(f"\nQuery: {query}\n")
    for i, hit in enumerate(hits, 1):
        print(f"[{i}] {hit['title']} (score: {hit['score']})")
        print(f"    {hit['content'][:120]}...")
        print()