import os
from pathlib import Path
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from llm_client import generate_grounded_answer, DEFAULT_MODEL

st.set_page_config(page_title="🌾 Agricultural Knowledge Retrieval System with RAG", layout="wide")

RUN_ID = "20260209_185402"

# Use secret if provided
COLLECTION_NAME = st.secrets.get(
    "COLLECTION_NAME",
    f"agrigenius_{RUN_ID}"
)
# Always resolve path relative to this file
BASE_DIR = Path(__file__).resolve().parent
CHROMA_PATH = str(BASE_DIR / "chroma_db")

st.title("🌾 Agricultural Knowledge Retrieval System with RAG")
st.subheader("Shruti Project")

st.write(
    "Ask agriculture-related questions. The system retrieves relevant chunks from "
    "the agricultural knowledge base and generates a grounded answer using Gemini."
)

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_vectordb():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(name=COLLECTION_NAME)

model = load_model()

try:
    collection = load_vectordb()
except Exception as e:
    st.error(
        "Could not load the vector collection. Please verify that the collection name "
        "and chroma_db path are correct."
    )
    st.write("Resolved CHROMA_PATH:", CHROMA_PATH)
    st.exception(e)
    st.stop()
st.sidebar.success(f"Collection loaded: {collection.count()} chunks")
st.sidebar.write(f"Embedding model: {EMBED_MODEL_NAME}")
st.sidebar.write(f"Collection: {COLLECTION_NAME}")
st.sidebar.write(f"LLM: {DEFAULT_MODEL}")

top_k = st.sidebar.slider("Top-K retrieval results", min_value=1, max_value=10, value=5)
llm_context_k = st.sidebar.slider("Chunks sent to LLM", min_value=1, max_value=5, value=3)

query = st.text_input("Ask an agriculture question:")

examples = [
    "What is Minimum Support Price scheme?",
    "What irrigation schemes support farmers?",
    "What is the purpose of PMKSY irrigation scheme?",
    "What government support is available for farmers?",
    "What agricultural statistics are available from government sources?",
]
st.caption("Example queries:")
st.code("\n".join(examples), language="text")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("Retrieving evidence and generating grounded answer..."):
            query_embedding = model.encode(query).tolist()

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]

            st.subheader("Generated Answer")

            try:
                answer = generate_grounded_answer(
                    query=query,
                    docs=docs,
                    metas=metas,
                    max_chunks=llm_context_k,
                )
                st.success(answer)
            except Exception as e:
                st.error(f"LLM generation failed: {e}")
                st.info("Retrieved evidence is still shown below.")

            st.subheader("Retrieved Evidence")

            for i in range(len(docs)):
                similarity = 1 - float(distances[i]) if distances[i] is not None else None
                title = (
                    f"Result {i+1}"
                    + (f" | Distance: {float(distances[i]):.4f}" if distances[i] is not None else "")
                    + (f" | Similarity: {similarity:.4f}" if similarity is not None else "")
                )

                with st.expander(title, expanded=(i == 0)):
                    st.write(f"**Chunk ID:** {ids[i]}")
                    st.write(f"**Source Type:** {metas[i].get('source_type')}")
                    st.write(f"**Source Name:** {metas[i].get('source_name')}")
                    st.write(f"**File Name:** {metas[i].get('file_name')}")
                    st.write(f"**Chunk Index:** {metas[i].get('chunk_index_in_file')}")
                    st.write("**Retrieved Text:**")
                    st.write(docs[i])