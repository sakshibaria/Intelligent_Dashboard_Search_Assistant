#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import os
import requests
import csv
from dotenv import load_dotenv
import pandas as pd
os.environ.pop("HTTP_PROXY",None)
os.environ.pop("http_PROXY",None)
os.environ.pop("HTTPs_PROXY",None)
os.environ.pop("https_PROXY",None)
load_dotenv(dotenv_path="details.env")
xrfkey=os.getenv("X_Qlik_Xrfkey")
qrscust=os.getenv("x_qlik_qrscust")
url=os.getenv("URL")
para=os.getenv("param")
proxies={
    "http":"",
    "https":""
}


# %%


headers={
    "X-Qlik-Xrfkey": f"{xrfkey}",
    "x-qlik-qrscust": f"{qrscust}"
}
response=requests.get(url,headers=headers,params=para,proxies=proxies)
data=response.json()
cobj=open("metadata.csv","w",newline='')
writer = csv.writer(cobj)
frow=["name","id","owner","description","file size(in Mb)","last modified date"]
writer.writerow(frow)
for meta in data:
    appname=meta["name"]
    mid=meta["id"]
    owner=meta["owner"]["name"]
    description=meta["description"]
    filesize=round(meta["fileSize"]/1024)
    lastdate=meta["modifiedDate"]
    alist=[appname,mid,owner,description,filesize,lastdate]
    writer.writerow(alist)

