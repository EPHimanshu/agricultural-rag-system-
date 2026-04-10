import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import streamlit as st
import chromadb
import tensorflow as tf
from PIL import Image
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
CHROMA_PATH = str(BASE_DIR / "runtime_chroma_db")
EMBED_MODEL_NAME = get_setting("EMBED_MODEL_NAME", "all-MiniLM-L6-v2")

MODELS_DIR = BASE_DIR / "models"
POTATO_MODEL_PATH = MODELS_DIR / "potato_classification_model.h5"
TOMATO_MODEL_PATH = MODELS_DIR / "tomato_classification_model.h5"
COTTON_MODEL_PATH = MODELS_DIR / "cotton_plant_disease_classifier.h5"

POTATO_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]

TOMATO_CLASSES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]

COTTON_CLASSES = [
    "Aphids",
    "Army worm",
    "Bacterial Blight",
    "Healthy leaf",
    "Powdery Mildew",
    "Target spot",
]


# ============================================================
# App Header
# ============================================================
st.title("🌾 Agricultural Knowledge Retrieval System with RAG")
st.subheader("Shruti Project")

st.write(
    "Ask agriculture-related questions in the RAG tab. "
    "Use the Leaf Advisory tab for potato and tomato leaf disease prediction. "
    "Use the Cotton Disease tab for cotton leaf disease prediction."
)


# ============================================================
# Cached Resources
# ============================================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource
def build_or_load_vectordb():
    """
    Loads the existing collection if available.
    If not available, builds it from chunks.parquet.
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

    model = load_embedding_model()
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


@st.cache_resource
def load_leaf_models():
    """
    Loads legacy .h5 leaf disease models with compatibility patches
    for dtype policy and preprocessing / augmentation layer deserialization.
    """
    if not POTATO_MODEL_PATH.exists():
        raise FileNotFoundError(f"Potato model not found at: {POTATO_MODEL_PATH}")

    if not TOMATO_MODEL_PATH.exists():
        raise FileNotFoundError(f"Tomato model not found at: {TOMATO_MODEL_PATH}")

    if not COTTON_MODEL_PATH.exists():
        raise FileNotFoundError(f"Cotton model not found at: {COTTON_MODEL_PATH}")

    def patch_dtype(kwargs):
        dtype_cfg = kwargs.get("dtype")
        if isinstance(dtype_cfg, dict):
            kwargs["dtype"] = dtype_cfg.get("config", {}).get("name", "float32")
        return kwargs

    def patch_common_kwargs(kwargs):
        kwargs.pop("data_format", None)
        kwargs.pop("pad_to_aspect_ratio", None)
        kwargs.pop("fill_mode", None)
        kwargs.pop("fill_value", None)
        kwargs.pop("antialias", None)
        kwargs = patch_dtype(kwargs)
        return kwargs

    class PatchedInputLayer(tf.keras.layers.InputLayer):
        def __init__(self, *args, **kwargs):
            if "batch_shape" in kwargs and "batch_input_shape" not in kwargs:
                kwargs["batch_input_shape"] = kwargs.pop("batch_shape")
            kwargs = patch_common_kwargs(kwargs)
            super().__init__(*args, **kwargs)

    class PatchedResizing(tf.keras.layers.Resizing):
        def __init__(self, *args, **kwargs):
            kwargs = patch_common_kwargs(kwargs)
            super().__init__(*args, **kwargs)

    class PatchedRescaling(tf.keras.layers.Rescaling):
        def __init__(self, *args, **kwargs):
            kwargs = patch_common_kwargs(kwargs)
            super().__init__(*args, **kwargs)

    class PatchedRandomFlip(tf.keras.layers.RandomFlip):
        def __init__(self, *args, **kwargs):
            kwargs = patch_common_kwargs(kwargs)
            super().__init__(*args, **kwargs)

    class PatchedRandomRotation(tf.keras.layers.RandomRotation):
        def __init__(self, *args, **kwargs):
            kwargs = patch_common_kwargs(kwargs)
            super().__init__(*args, **kwargs)

    class PatchedRandomZoom(tf.keras.layers.RandomZoom):
        def __init__(self, *args, **kwargs):
            kwargs = patch_common_kwargs(kwargs)
            super().__init__(*args, **kwargs)

    class PatchedRandomContrast(tf.keras.layers.RandomContrast):
        def __init__(self, *args, **kwargs):
            kwargs = patch_common_kwargs(kwargs)
            super().__init__(*args, **kwargs)

    class PatchedRandomTranslation(tf.keras.layers.RandomTranslation):
        def __init__(self, *args, **kwargs):
            kwargs = patch_common_kwargs(kwargs)
            super().__init__(*args, **kwargs)

    policy_cls = tf.keras.mixed_precision.Policy

    custom_objects = {
        "InputLayer": PatchedInputLayer,
        "Resizing": PatchedResizing,
        "Rescaling": PatchedRescaling,
        "RandomFlip": PatchedRandomFlip,
        "RandomRotation": PatchedRandomRotation,
        "RandomZoom": PatchedRandomZoom,
        "RandomContrast": PatchedRandomContrast,
        "RandomTranslation": PatchedRandomTranslation,
        "DTypePolicy": policy_cls,
        "Policy": policy_cls,
    }

    with tf.keras.utils.custom_object_scope(custom_objects):
        potato_model = tf.keras.models.load_model(
            POTATO_MODEL_PATH,
            custom_objects=custom_objects,
            compile=False
        )

        tomato_model = tf.keras.models.load_model(
            TOMATO_MODEL_PATH,
            custom_objects=custom_objects,
            compile=False
        )

        cotton_model = tf.keras.models.load_model(
            COTTON_MODEL_PATH,
            custom_objects=custom_objects,
            compile=False
        )

    return {
        "potato": potato_model,
        "tomato": tomato_model,
        "cotton": cotton_model,
    }


# ============================================================
# RAG Utility Functions
# ============================================================
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


# ============================================================
# Leaf Advisory Utility Functions
# ============================================================
def preprocess_leaf_image(uploaded_file, target_size=(256, 256)):
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array


def predict_with_single_model(model, img_array, class_names, crop_name):
    prediction = model.predict(img_array, verbose=0)
    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100.0
    predicted_class = class_names[predicted_index]

    return {
        "crop": crop_name,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 2)
    }


def predict_leaf_disease(uploaded_file, models_dict):
    display_image, img_array = preprocess_leaf_image(uploaded_file)

    potato_result = predict_with_single_model(
        model=models_dict["potato"],
        img_array=img_array,
        class_names=POTATO_CLASSES,
        crop_name="Potato"
    )

    tomato_result = predict_with_single_model(
        model=models_dict["tomato"],
        img_array=img_array,
        class_names=TOMATO_CLASSES,
        crop_name="Tomato"
    )

    all_results = [potato_result, tomato_result]
    best_result = max(all_results, key=lambda x: x["confidence"])

    return display_image, best_result, all_results


def predict_cotton_disease(uploaded_file, models_dict):
    display_image, img_array = preprocess_leaf_image(
        uploaded_file,
        target_size=(180, 180)
    )

    cotton_result = predict_with_single_model(
        model=models_dict["cotton"],
        img_array=img_array,
        class_names=COTTON_CLASSES,
        crop_name="Cotton"
    )

    return display_image, cotton_result


def format_prediction_label(predicted_class: str) -> str:
    return predicted_class.replace("___", " - ").replace("_", " ")


# ============================================================
# Load Shared Resources
# ============================================================
embedding_model = load_embedding_model()

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
tab1, tab2, tab3 = st.tabs([
    "🔎 General RAG Search",
    "🌿 Leaf Advisory System",
    "🧵 Cotton Disease Prediction"
])


# ============================================================
# Tab 1 - Existing RAG Search
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
                    model=embedding_model,
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
# Tab 2 - Leaf Advisory (Potato + Tomato)
# ============================================================
with tab2:
    st.write(
        "Upload a leaf image to predict disease directly using the trained potato and tomato classification models. "
        "This tab does not use RAG."
    )

    uploaded_image = st.file_uploader(
        "Upload leaf image",
        type=["jpg", "jpeg", "png"],
        key="leaf_image_upload"
    )

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Leaf Image", width=350)

        if st.button("Predict Disease", key="predict_leaf_button"):
            with st.spinner("Loading leaf models and running prediction..."):
                try:
                    leaf_models = load_leaf_models()

                    _, best_result, all_results = predict_leaf_disease(
                        uploaded_file=uploaded_image,
                        models_dict=leaf_models
                    )

                    st.subheader("Prediction Result")
                    st.success(
                        f"Predicted Class: {format_prediction_label(best_result['predicted_class'])}"
                    )
                    st.info(f"Predicted Crop: {best_result['crop']}")
                    st.info(f"Confidence: {best_result['confidence']}%")

                    st.subheader("Model-wise Confidence Comparison")
                    comparison_rows = []
                    for item in all_results:
                        comparison_rows.append({
                            "Model": item["crop"],
                            "Predicted Class": format_prediction_label(item["predicted_class"]),
                            "Confidence (%)": item["confidence"]
                        })

                    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)

                except Exception as e:
                    st.error(f"Leaf models could not be loaded or prediction failed: {e}")
    else:
        st.info("Upload a leaf image to run prediction.")


# ============================================================
# Tab 3 - Cotton Disease Prediction
# ============================================================
with tab3:
    st.write(
        "Upload a cotton leaf image to predict disease directly using the trained cotton classification model. "
        "This tab does not use RAG."
    )

    uploaded_cotton_image = st.file_uploader(
        "Upload cotton leaf image",
        type=["jpg", "jpeg", "png"],
        key="cotton_image_upload"
    )

    if uploaded_cotton_image is not None:
        st.image(uploaded_cotton_image, caption="Uploaded Cotton Leaf Image", width=350)

        if st.button("Predict Cotton Disease", key="predict_cotton_button"):
            with st.spinner("Loading cotton model and running prediction..."):
                try:
                    leaf_models = load_leaf_models()

                    _, cotton_result = predict_cotton_disease(
                        uploaded_file=uploaded_cotton_image,
                        models_dict=leaf_models
                    )

                    st.subheader("Cotton Prediction Result")
                    st.success(
                        f"Predicted Class: {format_prediction_label(cotton_result['predicted_class'])}"
                    )
                    st.info(f"Predicted Crop: {cotton_result['crop']}")
                    st.info(f"Confidence: {cotton_result['confidence']}%")

                    comparison_rows = [{
                        "Model": cotton_result["crop"],
                        "Predicted Class": format_prediction_label(cotton_result["predicted_class"]),
                        "Confidence (%)": cotton_result["confidence"]
                    }]

                    st.subheader("Prediction Summary")
                    st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)

                except Exception as e:
                    st.error(f"Cotton model could not be loaded or prediction failed: {e}")
    else:
        st.info("Upload a cotton leaf image to run cotton disease prediction.")