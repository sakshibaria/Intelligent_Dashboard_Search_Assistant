#!/usr/bin/env python
# coding: utf-8
# %%

# %%


#uploading to minio
from minio import Minio
from minio.error import S3Error
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path="details.env")
port=os.getenv("ports")
access=os.getenv("ak")
secret=os.getenv("sk")


client=Minio(
    f"{port}",
    access_key=f"{access}",
    secret_key=f"{secret}",
    secure=False
)
bucket_name="metadata-chatbot"
object_name="metadata.csv"
file_path="metadata.csv"
try:
    client.fput_object(
        bucket_name,
        object_name,
        file_path,
        content_type="text/csv"
    )
except S3Error as e:
    print(f"error:{e}")


# %%




