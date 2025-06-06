#!/usr/bin/env python
# coding: utf-8
# %%

# %%


from minio import Minio
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

  
client.fget_object("metadata-chatbot", "metadata.csv", "metadata.csv")


# %%




