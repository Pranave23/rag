from groq import Groq
from retriever import retrieve

client = Groq()  # reads GROQ_API_KEY from environment


def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Combine retrieved chunks + question into a single prompt.
    This is the 'augmentation' step in RAG.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[{i}] {chunk['title']}:\n{chunk['content']}")

    context = "\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant. Answer the question using ONLY the context provided below.
If the context does not contain enough information to answer, say so clearly.

Context:
{context}

Question: {query}

Answer:"""

    return prompt


def generate(query: str, top_k: int = 3) -> dict:
    """
    Full RAG pipeline: retrieve → augment → generate.
    Returns the answer plus the retrieved chunks for transparency.
    """
    chunks = retrieve(query, top_k=top_k)
    prompt = build_prompt(query, chunks)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",  # or llama-3.1-8b-instant for faster/cheaper
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )

    answer = response.choices[0].message.content

    return {
        "query":  query,
        "chunks": chunks,
        "prompt": prompt,
        "answer": answer,
    }