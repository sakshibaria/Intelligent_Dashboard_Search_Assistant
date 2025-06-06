<h1 align="center">📊 Qlik Sense Dashboard Inventory & Semantic Search 🔍</h1>

<p align="center">
  Automate your Qlik metadata collection, store it smartly, and search semantically using embeddings and LLMs.
</p>

---

## 🚀 Project Overview

This project automates the collection of **Qlik Sense dashboard metadata**, generates **semantic embeddings** using HuggingFace, and enables **natural language search** via a **Streamlit UI**. You can plug in **ChromaDB**, **ElasticSearch**, or even **Groq LLM** for lightning-fast semantic querying!

---

## 🛠️ Tech Stack

| Layer            | Tool / Technology                          |
|------------------|---------------------------------------------|
| Metadata Source  | Qlik QRS API (on-prem)                      |
| Embeddings       | `HuggingFace Transformers` 🤗               |
| Vector DB        | `ChromaDB` or `ElasticSearch` 🔍            |
| Object Storage   | `MinIO` (S3-compatible) 🪣                   |
| Frontend         | `Streamlit` + `Groq LLM` 🌐                 |
| Automation       | `Apache Airflow` 🛫                          |

---
## ⚙️ Features

- ✅ Automatic metadata extraction from Qlik dashboards
- 📦 Storage of cleaned metadata as CSV in MinIO
- 🤖 Semantic embeddings via HuggingFace Transformers
- 🧠 Fast semantic search using ChromaDB or ElasticSearch
- 🗣️ Natural language querying with Groq LLM (optional)
- ⏰ Full automation with Airflow DAGs

---
## 📊 Architecture Diagram

<details>
<summary>Click to view Mermaid Diagram</summary>

```mermaid
graph TD
    A[Qlik QRS API] --> B[Metadata Extractor (Python Script)]
    B --> C[CSV Export]
    C --> D[Upload to MinIO Storage]
    C --> E[Embedding Generator (HuggingFace)]
    E --> F[Vector DB (ChromaDB or ElasticSearch)]
    F --> G[Streamlit UI]
    G --> H[LLM (Groq API)]

    subgraph Airflow DAG
        B
        C
        E
        F
    end
       
