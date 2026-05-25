import chromadb #vector database
from sentence_transformers import SentenceTransformer #convert text to vectors, used to find similar documents
from documents import DOCUMENTS

print("Starting indexer...")

embedder = SentenceTransformer("all-MiniLM-L6-v2") #convert each document’s content from DOCUMENTS into vectors before indexing them in ChromaDB 

print("setting up chromadb")


client = chromadb.PersistentClient(path="./chroma_db")  # saves to disk, stores data in ./chroma_db

# Delete old collection if re-running
try:
    client.delete_collection("my_docs")
except:
    pass

collection = client.create_collection(
    name="my_docs",
    metadata={"hnsw:space": "cosine"}  # use cosine similarity
)

print(f"Indexing {len(DOCUMENTS)} documents...")

# Embed all documents at once (faster than one by one)
texts = [doc["content"] for doc in DOCUMENTS]
embeddings = embedder.encode(texts, show_progress_bar=True)

collection.add(
    ids        = [doc["id"] for doc in DOCUMENTS],
    documents  = [doc["content"] for doc in DOCUMENTS],
    metadatas  = [{"title": doc["title"]} for doc in DOCUMENTS],
    embeddings = [emb.tolist() for emb in embeddings]
) #all of these info goes into the my_docs 

print(f"Done! {len(DOCUMENTS)} chunks indexed and saved to ./chroma_db")