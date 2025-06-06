#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# dashboard_semantic_search.py

import os
import pandas as pd
import logging
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DashboardSemanticSearch:
    def __init__(self,
                 es_host: str = "localhost",
                 es_port: int = 9200,
                 model_name: str = "./Sentencemmodel",  # <-- Update if needed
                 index_name: str = "dashboard_search",
                 cache_path: str = "cached_data.parquet"):

        self.es_host = es_host
        self.es_port = es_port
        self.model_name = model_name
        self.index_name = index_name
        self.cache_path = cache_path

        self.es = None
        self.model = None
        self.df = None

    def connect_elasticsearch(self):
        try:
            self.es = Elasticsearch([f"http://{self.es_host}:{self.es_port}"])
            if self.es.ping():
                logger.info("✅ Connected to Elasticsearch.")
                return True
            else:
                logger.error("❌ Failed to ping Elasticsearch.")
                return False
        except Exception as e:
            logger.error(f"❌ Elasticsearch connection failed: {e}")
            return False

    def load_model(self):
        try:
            self.model = SentenceTransformer(self.model_name)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False

    # --- Modified: Always load fresh CSV and skip parquet cache ---
    def load_data_force_refresh(self, csv_path: str):
        """
        Always load fresh data from CSV, ignoring any cached parquet.
        """
        try:
            self.df = pd.read_csv(csv_path)
            self.df["last modified date"] = pd.to_datetime(
                self.df.get("last modified date", pd.NaT), errors="coerce"
            ).dt.tz_localize(None)
            self.df = self.df.fillna("")
            logger.info("✅ Loaded fresh CSV data (cache ignored).")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            return False

    # --- Modified: Always recreate embeddings and save parquet ---
    def prepare_data_force_refresh(self):
        """
        Always generate fresh embeddings, then save to parquet.
        """
        try:
            self.df["combined_text"] = self.df.astype(str).agg(' '.join, axis=1)
            self.df["embedding"] = self.df["combined_text"].apply(lambda x: self.model.encode(x).tolist())
            self.df.to_parquet(self.cache_path, index=False)
            logger.info(f"✅ Generated new embeddings and saved to {self.cache_path}.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to prepare data: {e}")
            return False

    def create_index(self):
        try:
            if self.es.indices.exists(index=self.index_name):
                self.es.indices.delete(index=self.index_name)
                logger.info(f"ℹ️ Deleted existing index: {self.index_name}")

            mapping = {
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0
                },
                "mappings": {
                    "properties": {
                        "name": {"type": "text"},
                        "id": {"type": "keyword"},
                        "owner": {"type": "keyword"},
                        "description": {"type": "text"},
                        "file size(in Mb)": {"type": "float"},
                        "last modified date": {"type": "date"},
                        "combined_text": {"type": "text"},
                        "embedding": {
                            "type": "dense_vector",
                            "dims": 384,
                            "index": True,
                            "similarity": "cosine"
                        }
                    }
                }
            }

            self.es.indices.create(index=self.index_name, body=mapping)
            logger.info(f"✅ Created index: {self.index_name}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to create index: {e}")
            return False

    def index_data(self):
        try:
            records = self.df.to_dict("records")

            def doc_generator():
                for i, record in enumerate(records):
                    yield {
                        "_index": self.index_name,
                        "_id": i,
                        "_source": record
                    }

            bulk(self.es, doc_generator(), chunk_size=100, request_timeout=60)
            self.es.indices.refresh(index=self.index_name)
            logger.info(f"✅ Indexed {len(records)} documents.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to index data: {e}")
            return False

    def semantic_search(self, query: str, size: int = 5):
        try:
            query_embedding = self.model.encode(query).tolist()

            search_body = {
                "size": size,
                "_source": {"excludes": ["embedding"]},
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": query_embedding}
                        }
                    }
                }
            }

            response = self.es.search(index=self.index_name, body=search_body)

            results = []
            for hit in response['hits']['hits']:
                results.append({
                    'id': hit['_id'],
                    'score': hit['_score'],
                    'data': hit['_source']
                })
            return results
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []

    # --- New combined method to refresh embeddings in one call ---
    def refresh_embeddings(self, csv_path: str):
        if not self.load_data_force_refresh(csv_path):
            return False
        if not self.prepare_data_force_refresh():
            return False
        return True


def elasticsearch_semantic_search(query):
    try:
        # Example placeholder for your metadata CSV loading function
        # Replace or modify if you have a custom loader
        csv_file = "metadata.csv"

        dashboard = DashboardSemanticSearch()
        if not dashboard.connect_elasticsearch(): return "❌ Elasticsearch connection failed."
        if not dashboard.load_model(): return "❌ Model loading failed."
        if not dashboard.refresh_embeddings(csv_file): return "❌ Embedding refresh failed."
        if not dashboard.create_index(): return "❌ Index creation failed."
        if not dashboard.index_data(): return "❌ Indexing failed."

        results = dashboard.semantic_search(query, size=5)
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


if __name__ == "__main__":
    # Example run:
    query_text = "random"
    result = elasticsearch_semantic_search(query_text)
    print(result)

