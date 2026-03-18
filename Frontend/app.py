import os
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer

from llm_client import generate_grounded_answer, DEFAULT_MODEL


# ============================================================
# Streamlit Page Config
# ============================================================
st.set_page_config(
    page_title="Agricultural Knowledge Retrieval System with RAG",
    page_icon="🌾",
    layout="wide"
)


# ============================================================
# Safe Settings Helper
# ============================================================
def get_setting(name: str, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)


# ============================================================
# App Constants
# ============================================================
RUN_ID = get_setting("RUN_ID", "20260209_185402")
COLLECTION_NAME = get_setting("COLLECTION_NAME", f"agrigenius_{RUN_ID}")

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = BASE_DIR / "data" / "chunks.parquet"

# Set this to whichever folder actually exists in your final project
# Example options:
# CHROMA_PATH = str(BASE_DIR / "runtime_chroma_db")
# CHROMA_PATH = str(BASE_DIR / "chroma_db")
CHROMA_PATH = str(BASE_DIR / "runtime_chroma_db")

EMBED_MODEL_NAME = get_setting("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")


# ============================================================
# Advisory Options
# ============================================================
CROP_OPTIONS = [
    "Tomato",
    "Potato",
    "Brinjal",
    "Chili",
    "Okra",
    "Wheat",
    "Rice",
    "Maize",
    "Cotton",
    "Mustard",
]

SYMPTOM_OPTIONS = [
    "Yellow spots",
    "Brown spots",
    "White powder on leaves",
    "Leaf curling",
    "Leaf wilting",
    "Dry leaf edges",
    "Black patches",
    "Holes in leaves",
    "Yellowing of leaves",
    "Stunted growth",
]


# ============================================================
# App Header
# ============================================================
st.title("🌾 Agricultural Knowledge Retrieval System with RAG")
st.subheader("Shruti Project")

st.write(
    "Ask agriculture-related questions. The system retrieves relevant chunks from "
    "the agricultural knowledge base and generates a grounded answer using Gemini."
)


# ============================================================
# Cached Resources
# ============================================================
@st.cache_resource
def load_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource
def build_or_load_vectordb():
    """
    Loads the existing collection if available.
    If not available, builds it from chunks.parquet.
    This preserves the existing implementation behavior.
    """
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"chunks.parquet not found at: {CHUNKS_PATH}")

    chunks_df = pd.read_parquet(CHUNKS_PATH)
    chunks_df["chunk_text"] = chunks_df["chunk_text"].fillna("").astype(str)
    chunks_df = chunks_df[chunks_df["chunk_text"].str.strip() != ""].copy()

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    collections = client.list_collections()
    existing_names = [c.name if hasattr(c, "name") else str(c) for c in collections]

    if COLLECTION_NAME in existing_names:
        collection = client.get_collection(name=COLLECTION_NAME)
        return collection, len(chunks_df)

    collection = client.create_collection(name=COLLECTION_NAME)

    docs = chunks_df["chunk_text"].astype(str).tolist()
    ids = chunks_df["chunk_id"].astype(str).tolist()

    metadatas = []
    for _, row in chunks_df.iterrows():
        metadatas.append({
            "source_type": str(row.get("source_type", "")),
            "source_name": str(row.get("source_name", "")),
            "file_name": str(row.get("file_name", "")),
            "chunk_index_in_file": int(row.get("chunk_index_in_file", 0)),
            "chunk_words": int(row.get("chunk_words", 0)),
            "chunk_chars": int(row.get("chunk_chars", 0)),
        })

    model = load_model()
    batch_size = 64

    for start in range(0, len(docs), batch_size):
        end = min(start + batch_size, len(docs))

        batch_docs = docs[start:end]
        batch_ids = ids[start:end]
        batch_metas = metadatas[start:end]

        batch_embeddings = model.encode(batch_docs, show_progress_bar=False).tolist()

        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=batch_embeddings
        )

    return collection, len(chunks_df)


# ============================================================
# Utility Functions
# ============================================================
def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def build_leaf_advisory_query(crop: str, symptom: str) -> str:
    crop_norm = normalize_text(crop)
    symptom_norm = normalize_text(symptom)
    return f"treatment and prevention for {symptom_norm} on {crop_norm} leaves"


def retrieve_documents(
    query: str,
    top_k: int,
    model,
    collection
) -> Dict[str, List[Any]]:
    query_embedding = model.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    return {
        "docs": results.get("documents", [[]])[0],
        "metas": results.get("metadatas", [[]])[0],
        "distances": results.get("distances", [[]])[0],
        "ids": results.get("ids", [[]])[0],
    }


def run_rag_pipeline(
    query: str,
    top_k: int,
    llm_context_k: int,
    model,
    collection
) -> Dict[str, Any]:
    retrieved = retrieve_documents(
        query=query,
        top_k=top_k,
        model=model,
        collection=collection
    )

    docs = retrieved["docs"]
    metas = retrieved["metas"]
    distances = retrieved["distances"]
    ids = retrieved["ids"]

    answer: Optional[str] = None
    error_message: Optional[str] = None

    if docs:
        try:
            answer = generate_grounded_answer(
                query=query,
                docs=docs,
                metas=metas,
                max_chunks=llm_context_k,
            )
        except Exception as e:
            error_message = f"LLM generation failed: {e}"
            answer = "The system retrieved relevant evidence, but grounded answer generation failed."
    else:
        answer = "No relevant documents were retrieved for this query."

    return {
        "query": query,
        "answer": answer,
        "error_message": error_message,
        "docs": docs,
        "metas": metas,
        "distances": distances,
        "ids": ids,
    }


def render_generated_answer(answer: Optional[str], error_message: Optional[str]):
    st.subheader("Generated Answer")

    if answer:
        st.success(answer)

    if error_message:
        st.error(error_message)
        st.info("Retrieved evidence is still shown below.")


def render_retrieved_evidence(
    docs: List[str],
    metas: List[Dict[str, Any]],
    distances: List[Any],
    ids: List[str]
):
    st.subheader("Retrieved Evidence")

    if not docs:
        st.warning("No evidence found for this query.")
        return

    for i in range(len(docs)):
        distance_value = distances[i] if i < len(distances) else None
        similarity = None

        try:
            if distance_value is not None:
                similarity = 1 - float(distance_value)
        except Exception:
            similarity = None

        title = f"Result {i + 1}"
        if distance_value is not None:
            try:
                title += f" | Distance: {float(distance_value):.4f}"
            except Exception:
                pass
        if similarity is not None:
            title += f" | Similarity: {similarity:.4f}"

        with st.expander(title, expanded=(i == 0)):
            meta = metas[i] if i < len(metas) and metas[i] else {}

            st.write(f"**Chunk ID:** {ids[i] if i < len(ids) else 'N/A'}")
            st.write(f"**Source Type:** {meta.get('source_type', '')}")
            st.write(f"**Source Name:** {meta.get('source_name', '')}")
            st.write(f"**File Name:** {meta.get('file_name', '')}")
            st.write(f"**Chunk Index:** {meta.get('chunk_index_in_file', '')}")
            st.write("**Retrieved Text:**")
            st.write(docs[i])


def render_query_preview(query: str):
    st.subheader("Generated Advisory Query")
    st.code(query, language="text")


# ============================================================
# Load Shared Resources
# ============================================================
model = load_model()

try:
    collection, chunk_count = build_or_load_vectordb()
except Exception as e:
    st.error("Could not build or load the vector database.")
    st.exception(e)
    st.stop()


# ============================================================
# Sidebar
# ============================================================
st.sidebar.success(f"Collection ready: {chunk_count} chunks")
st.sidebar.write(f"Embedding model: {EMBED_MODEL_NAME}")
st.sidebar.write(f"Collection: {COLLECTION_NAME}")
st.sidebar.write(f"LLM: {DEFAULT_MODEL}")

top_k = st.sidebar.slider("Top-K retrieval results", min_value=1, max_value=10, value=5)
llm_context_k = st.sidebar.slider("Chunks sent to LLM", min_value=1, max_value=5, value=3)


# ============================================================
# Tabs
# ============================================================
tab1, tab2 = st.tabs(["🔎 General RAG Search", "🌿 Leaf Advisory System"])


# ============================================================
# Tab 1 - Existing Search
# ============================================================
with tab1:
    query = st.text_input("Ask an agriculture question:", key="general_query")

    examples = [
        "What is Minimum Support Price scheme?",
        "What irrigation schemes support farmers?",
        "What is the purpose of PMKSY irrigation scheme?",
        "What government support is available for farmers?",
        "What agricultural statistics are available from government sources?",
    ]

    st.caption("Example queries:")
    st.code("\n".join(examples), language="text")

    if st.button("Search", key="search_button"):
        if not query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Retrieving evidence and generating grounded answer..."):
                result = run_rag_pipeline(
                    query=query.strip(),
                    top_k=top_k,
                    llm_context_k=llm_context_k,
                    model=model,
                    collection=collection
                )

            render_generated_answer(
                answer=result["answer"],
                error_message=result["error_message"]
            )
            render_retrieved_evidence(
                docs=result["docs"],
                metas=result["metas"],
                distances=result["distances"],
                ids=result["ids"]
            )


# ============================================================
# Tab 2 - Leaf Advisory
# ============================================================
with tab2:
    st.write(
        "Use structured crop and symptom selection to generate a grounded advisory response "
        "using the same agricultural RAG backend."
    )

    left_col, right_col = st.columns([1, 1])

    with left_col:
        uploaded_image = st.file_uploader(
            "Upload leaf image (optional for now)",
            type=["jpg", "jpeg", "png"],
            key="leaf_image_upload"
        )

        if uploaded_image is not None:
            st.image(uploaded_image, caption="Uploaded Leaf Image", use_column_width=True)
    with right_col:
        st.info(
            "The uploaded image is currently stored only for UI and future extension. "
            "It is not used in retrieval yet. Later, you can plug in an image model here "
            "without changing the RAG backend."
        )

    crop = st.selectbox(
        "Select Crop",
        options=["Select Crop"] + CROP_OPTIONS,
        index=0,
        key="crop_select"
    )

    symptom = st.selectbox(
        "Select Symptom",
        options=["Select Symptom"] + SYMPTOM_OPTIONS,
        index=0,
        key="symptom_select"
    )

    if st.button("Generate Advisory", key="advisory_button"):
        if crop == "Select Crop":
            st.warning("Please select a crop.")
        elif symptom == "Select Symptom":
            st.warning("Please select a symptom.")
        else:
            advisory_query = build_leaf_advisory_query(crop, symptom)
            render_query_preview(advisory_query)

            with st.spinner("Retrieving evidence and generating grounded advisory..."):
                result = run_rag_pipeline(
                    query=advisory_query,
                    top_k=top_k,
                    llm_context_k=llm_context_k,
                    model=model,
                    collection=collection
                )

            render_generated_answer(
                answer=result["answer"],
                error_message=result["error_message"]
            )
            render_retrieved_evidence(
                docs=result["docs"],
                metas=result["metas"],
                distances=result["distances"],
                ids=result["ids"]
            )