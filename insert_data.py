from pymilvus import MilvusClient
import numpy as np
import json

# 填充数据
CLUSTER_ENDPOINT = "https://in03-c37b3b2d61f1d1f.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn"
TOKEN = "e769f024891860623ab86426431422cb34e57edd152377121d40cbced6348d19fcb0f253223bf65efe3d401c667423971c0b118a"
COLLECTION = "TieTa_rag"

client = MilvusClient(
    uri=CLUSTER_ENDPOINT,
    token=TOKEN 
)

client.create_partition(
    collection_name=COLLECTION,
    partition_name="num"
)

client.create_partition(
    collection_name=COLLECTION,
    partition_name="content"
)
    
file_path1 = '/amax/home/chendian/WEI_project/MM-main/data_num.json'
file_path2 = '/amax/home/chendian/WEI_project/MM-main/data_content.json'
with open(file_path1, 'r', encoding='utf-8') as file:  # 指定utf-8编码以正确读取汉字
    data1 = json.load(file)
    
with open(file_path2, 'r', encoding='utf-8') as file:  # 指定utf-8编码以正确读取汉字
    data2 = json.load(file)

client.insert(
    collection_name=COLLECTION,
    data=data1,
    partition_name="num"
)

client.insert(
    collection_name=COLLECTION,
    data=data2,
    partition_name="content"
)
