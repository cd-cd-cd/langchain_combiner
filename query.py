from pymilvus import MilvusClient, connections, Collection
import numpy as np
import json

CLUSTER_ENDPOINT = "https://in05-c395bc82883c484.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn"
TOKEN = "e769f024891860623ab86426431422cb34e57edd152377121d40cbced6348d19fcb0f253223bf65efe3d401c667423971c0b118a"
COLLECTION = "TieTa_rag"

client = MilvusClient(
    uri=CLUSTER_ENDPOINT,
    token=TOKEN,
    user="db_c395bc82883c484",
    password="tietaTIETA123"
)

res = client.query(
    collection_name = COLLECTION,
    # highlight-start
    output_fields =[" count(*)"],
    # partition_names=["content"],
)

# # 查询分区
# res = client.list_partitions(collection_name=COLLECTION)
print(res)

connections.disconnect("default")