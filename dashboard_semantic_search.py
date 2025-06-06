#!/usr/bin/env python
# coding: utf-8
# %%

# %%


# dashboard_semantic_search.py

import os
import pandas as pd
import logging
from typing import List, Dict
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer
from caching import load_sentence_model, get_chroma_collection, load_metadata_csv, clear_all_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DashboardSemanticSearch:
    def __init__(self,
                 es_host: str = "localhost",
                 es_port: int = 9200,
                 model_name: str = "./Sentencemmodel",
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
            self.model = SentenceTransformer(self.model_name,device="cpu")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            return False

    def load_data(self, csv_path: str):
        try:
            if os.path.exists(self.cache_path):
                self.df = pd.read_parquet(self.cache_path)
                return True

            self.df = pd.read_csv(csv_path)
            self.df["last modified date"] = pd.to_datetime(
                self.df.get("last modified date", pd.NaT), errors="coerce"
            ).dt.tz_localize(None)
            self.df = self.df.fillna("")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load data: {e}")
            return False

    def prepare_data(self):
        try:
            if "embedding" in self.df.columns:
                return True

            self.df["combined_text"] = self.df.astype(str).agg(' '.join, axis=1)
            self.df["embedding"] = self.df["combined_text"].apply(lambda x: self.model.encode(x).tolist())
            self.df.to_parquet(self.cache_path, index=False)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to prepare data: {e}")
            return False

    def create_index(self):
        try:
            if self.es.indices.exists(index=self.index_name):
                self.es.indices.delete(index=self.index_name)

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


# %%




