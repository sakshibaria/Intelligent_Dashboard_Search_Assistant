#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
df = pd.read_csv("metadata.csv")  # Replace with actual filename
df["description"] = df["description"].fillna("No description provided.")
df["text"] = (
        "Name: " + df["name"].astype(str) + ", " +
        "Owner: " + df["owner"].astype(str) + ", " +
        "Description: " + df["description"].astype(str) + ", " +
        "File Size (MB): " + df["file size(in Mb)"].astype(str) + ", " +
        "Last Modified: " + df["last modified date"].astype(str)
    )
from sentence_transformers import SentenceTransformer
import numpy as np
# Load from local folder
model = SentenceTransformer("Sentencemmodel")
df["embedding"] = df["text"].apply(lambda x: model.encode(x).tolist())
import chromadb
from chromadb.config import Settings
client = chromadb.PersistentClient(path="./chroma_store")
collection = client.get_or_create_collection(name="dashboard_search")
collection.add(
    ids=df["id"].astype(str).tolist(),
    documents=df["text"].tolist(),
    embeddings=df["embedding"].tolist(),
    metadatas=df.drop(columns=["embedding", "text"]).to_dict("records")
        )         


# In[ ]:




