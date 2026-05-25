DOCUMENTS = [
    {
        "id": "1",
        "title": "What is RAG",
        "content": """Retrieval-Augmented Generation (RAG) is a technique that improves
        LLM responses by first retrieving relevant documents from a knowledge base,
        then passing them as context to the language model before generating an answer.
        This grounds the model's response in real data rather than relying on training memory."""
    },
    {
        "id": "2",
        "title": "What are embeddings",
        "content": """Embeddings are dense numerical vectors that represent the meaning of text.
        Similar pieces of text have vectors that point in similar directions in vector space.
        Models like sentence-transformers/all-MiniLM-L6-v2 convert sentences into 384-dimensional
        vectors. These vectors are used to find semantically similar content during retrieval."""
    },
    {
        "id": "3",
        "title": "What is ChromaDB",
        "content": """ChromaDB is a free, open-source vector database that runs locally on your machine.
        It stores embeddings alongside their source text and metadata, and supports fast similarity
        search using cosine distance. You can use it in-memory (resets each run) or
        persist it to disk so the index survives between sessions."""
    },
    {
        "id": "4",
        "title": "Chunking strategy",
        "content": """Chunking means splitting long documents into smaller pieces before indexing.
        Good chunk size is usually 300-500 tokens. Too small loses context; too large dilutes
        relevance. A common technique is to overlap chunks by 10-20% so context is not
        lost at chunk boundaries. For example, a 500-token chunk with 50-token overlap."""
    },
    {
        "id": "5",
        "title": "Cosine similarity",
        "content": """Cosine similarity measures the angle between two vectors, returning a score
        from 0 (completely different) to 1 (identical direction). It is the standard way
        to compare text embeddings because it ignores vector magnitude and focuses purely
        on direction, which represents semantic meaning in embedding space."""
    },
    {
        "id": "6",
        "title": "What is a vector database",
        "content": """A vector database is a specialized database that stores and indexes vectors.
        It supports fast similarity search using cosine distance. ChromaDB is a free, open-source
        vector database that runs locally on your machine. It stores embeddings alongside their
        source text and metadata, and supports fast similarity search using cosine distance.
        You can use it in-memory (resets each run) or persist it to disk so the index survives
        between sessions."""
    }
]       