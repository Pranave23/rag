import os
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🔍",
    layout="wide",
)

CHROMA_PATH = Path("./chroma_db")
SAMPLE_QUESTIONS = [
    "What is RAG?",
    "How do embeddings work?",
    "What is ChromaDB?",
    "What chunk size should I use?",
    "How does cosine similarity work?",
]


@st.cache_resource
def load_generate():
    from generator import generate

    return generate


def main():
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except (KeyError, FileNotFoundError):
        st.error("Add `GROQ_API_KEY` to `.streamlit/secrets.toml` or your Streamlit Cloud app secrets.")
        st.stop()

    os.environ["GROQ_API_KEY"] = api_key

    st.title("RAG Assistant")
    st.caption("Ask questions about RAG, embeddings, ChromaDB, and vector search.")

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Chunks to retrieve", min_value=1, max_value=6, value=3)

        st.divider()
        st.markdown("**Run locally:**")
        st.code("streamlit run app.py", language="bash")

    if not CHROMA_PATH.exists():
        st.error("Knowledge base not found. Run `python indexer.py` first to build `./chroma_db`.")
        st.stop()

    st.subheader("Try a sample question")
    cols = st.columns(len(SAMPLE_QUESTIONS))
    for col, question in zip(cols, SAMPLE_QUESTIONS):
        if col.button(question, use_container_width=True):
            st.session_state["query_input"] = question

    query = st.text_input(
        "Your question",
        placeholder="e.g. What is a vector database?",
        key="query_input",
    )

    ask = st.button("Ask", type="primary")

    if not ask:
        return

    query = query.strip()
    if not query:
        st.info("Enter a question first.")
        return

    with st.spinner("Searching knowledge base and generating answer..."):
        try:
            generate = load_generate()
            result = generate(query, top_k=top_k)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            return

    st.subheader("Answer")
    st.markdown(result["answer"])

    st.subheader("Retrieved chunks")
    for i, chunk in enumerate(result["chunks"], 1):
        with st.expander(f"[{i}] {chunk['title']} — similarity {chunk['score']}"):
            st.write(chunk["content"])

    with st.expander("Full prompt sent to the model"):
        st.code(result["prompt"], language="text")


if __name__ == "__main__":
    main()
