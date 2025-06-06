import streamlit as st
import pandas as pd
from caching import load_sentence_model, get_chroma_collection, load_metadata_csv, clear_all_cache
from dashboard_semantic_search import DashboardSemanticSearch

# --- Search Implementations ---
def cosine_search(query):
    model = load_sentence_model()
    collection = get_chroma_collection()
    df = load_metadata_csv()

    def search_dashboards(query, top_n=10):
        query_vec = model.encode([query])[0]
        results = collection.query(
            query_embeddings=[query_vec],
            n_results=top_n,
            include=["documents", "metadatas", "distances"]
        )

        matches = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            meta["score"] = 1 - dist
            matches.append(meta)

        return pd.DataFrame(matches).sort_values("score", ascending=False)

    return search_dashboards(query)

def groq_llm_search(query):
    import os
    import pandas as pd
    import numpy as np
    import faiss
    import requests
    from sentence_transformers import SentenceTransformer

    # Paths
    MODEL_NAME = "Sentencemmodel"
    INDEX_FILE = "faiss_index.bin"
    EMBEDDING_FILE = "faiss_embeddings.npy"
    DF_PATH = "metadata.csv"

    try:
        # Load model and data
        df = pd.read_csv(DF_PATH)
        model = SentenceTransformer(MODEL_NAME)

        # Describe dataframe
        def describe_dataframe(df):
            description = f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns.\n"
            description += "Columns include:\n"
            for col in df.columns:
                description += f"- {col}: {df[col].dtype}, {df[col].nunique()} unique values\n"
            description += "\nBasic statistics:\n"
            description += df.describe().to_string()
            return description

        # Load or create FAISS index
        def load_or_create_faiss_index():
            if os.path.exists(INDEX_FILE) and os.path.exists(EMBEDDING_FILE):
                index = faiss.read_index(INDEX_FILE)
                embeddings = np.load(EMBEDDING_FILE)
            else:
                sentences = df.astype(str).apply(lambda row: ' | '.join(row), axis=1).tolist()
                embeddings = model.encode(sentences, convert_to_numpy=True)
                index = faiss.IndexFlatL2(embeddings.shape[1])
                index.add(embeddings)
                faiss.write_index(index, INDEX_FILE)
                np.save(EMBEDDING_FILE, embeddings)
            return index

        # Ollama API call
        def ollama_chat(prompt, model="llama3"):
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False}
            )
            if response.status_code != 200:
                raise RuntimeError(f"Ollama API error: {response.text}")
            return response.json()["response"].strip()

        # Embed query and search
        index = load_or_create_faiss_index()
        q_embedding = model.encode([query], convert_to_numpy=True)
        distances, indices = index.search(q_embedding, k=5)
        relevant_rows = df.iloc[indices[0]]

        # Construct prompt
        prompt = f"""
        You are a data analyst.
        
        Here is a summary of the dataset:
        {describe_dataframe(df)}
        
        Here are the 5 most relevant rows related to the user's question:
        {relevant_rows.to_string(index=False)}
        
        User Question:
        {query}
        
        Answer:"""

        response = ollama_chat(prompt)
        return response

    except Exception as e:
        return f"🚨 FAISS + Ollama error: {e}"

def elasticsearch_semantic_search(query):
    try:
        # Download metadata
        df = load_metadata_csv()

        dashboard = DashboardSemanticSearch()
        if not dashboard.connect_elasticsearch(): return "❌ Elasticsearch connection failed."
        if not dashboard.load_model(): return "❌ Model loading failed."
        if not dashboard.load_data("metadata.csv"): return "❌ Metadata loading failed."
        if not dashboard.prepare_data(): return "❌ Embedding preparation failed."
        if not dashboard.create_index(): return "❌ Index creation failed."
        if not dashboard.index_data(): return "❌ Indexing failed."

        results = dashboard.semantic_search(query, size=10)
        if not results:
            return "🔍 No results found."

        df = pd.DataFrame([{
            "Name": r["data"].get("name"),
            "Owner": r["data"].get("owner"),
            "Description": r["data"].get("description"),
            "Size (MB)": r["data"].get("file size(in Mb)"),
            "Modified": r["data"].get("last modified date"),
            "Similarity": round(r["score"], 3)
        } for r in results])

        return df
    except Exception as e:
        return f"🚨 Error: {e}"

# --- Search Methods Mapping ---
search_methods = {
    "Cosine Similarity": cosine_search,
    "LLM-Based Search (llama3)": groq_llm_search,
    "ElasticSearch Semantic Search": elasticsearch_semantic_search,
}

st.set_page_config(page_title="Customizable Multi-Search", layout="wide")

# Display logo and title
col1, col2 = st.columns([1, 6])
with col1:
    st.image("iocl_logo.png", width=80)  # <-- Replace with the path to your logo file
with col2:
    st.title("Intelligent Dashboard Search Assistant")

# Sidebar - cache reset
st.sidebar.header("Options")
if st.sidebar.button("Clear Cache"):
    clear_all_cache()
    st.sidebar.success("✅ Cache cleared. Please rerun the app.")

# Query input
query = st.text_input("Enter your search query:")

# Search method selection
selected_methods = st.multiselect(
    "Select search method(s) to apply:",
    options=list(search_methods.keys()),
    default=["Cosine Similarity"]
)

# Run search
if st.button("Run Search") and query.strip() and selected_methods:
    with st.spinner("Running selected searches..."):
        results = {name: search_methods[name](query) for name in selected_methods}
    st.success("✅ Search complete!")

    for method, result in results.items():
        st.subheader(method)
        if isinstance(result, pd.DataFrame):
            st.dataframe(result)
        else:
            st.markdown(result)
        st.markdown("----")
elif not selected_methods:
    st.warning("⚠️ Please select at least one search method.")
elif not query.strip():
    st.info("ℹ️ Please enter a query to start searching.")
