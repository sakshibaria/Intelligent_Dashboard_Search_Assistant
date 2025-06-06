#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from minio import Minio
import chromadb
from chromadb.config import Settings
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="details.env")
port=os.getenv("ports")
access=os.getenv("ak")
secret=os.getenv("sk")

# Cache SentenceTransformer model
@st.cache_resource
def load_sentence_model():
    return SentenceTransformer("./Sentencemmodel" , device="cpu")

# Cache Chroma collection
@st.cache_resource
def get_chroma_collection():
    client = chromadb.PersistentClient(path="./chroma_store")
    return client.get_or_create_collection(name="dashboard_search")

# Cache MinIO CSV fetch + dataframe construction
@st.cache_data
def load_metadata_csv():
    client=Minio(
        f"{port}",
        access_key=f"{access}",
        secret_key=f"{secret}",
        secure=False
    )
    client.fget_object("metadata-chatbot", "metadata.csv", "metadata.csv")
    df = pd.read_csv("metadata.csv")
    df["description"] = df["description"].fillna("No description provided.")
    df["text"] = (
        "Name: " + df["name"].astype(str) + ", " +
        "Owner: " + df["owner"].astype(str) + ", " +
        "Description: " + df["description"].astype(str) + ", " +
        "File Size (MB): " + df["file size(in Mb)"].astype(str) + ", " +
        "Last Modified: " + df["last modified date"].astype(str)
    )
    return df

# Cache clear utility (optional for app UI)
def clear_all_cache():
    st.cache_data.clear()
    st.cache_resource.clear()

