from pymilvus import MilvusClient, DataType

## 创建vilvus

CLUSTER_ENDPOINT = "https://in03-c37b3b2d61f1d1f.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn"
TOKEN = "e769f024891860623ab86426431422cb34e57edd152377121d40cbced6348d19fcb0f253223bf65efe3d401c667423971c0b118a"

# 1. Set up a Milvus client
client = MilvusClient(
    uri=CLUSTER_ENDPOINT,
    token=TOKEN 
)

# 3. Create a collection in customized setup mode

# 3.1. Create schema
schema = MilvusClient.create_schema(
    auto_id=False,
    enable_dynamic_field=True,
)

# 3.2. Add fields to schema
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True)
schema.add_field(field_name="tag_class", datatype=DataType.VARCHAR, max_length=512)
schema.add_field(field_name="tag_class_vector", datatype=DataType.FLOAT_VECTOR, dim=512)
schema.add_field(field_name="image_vector", datatype=DataType.FLOAT_VECTOR, dim=512)
schema.add_field(field_name="label", datatype=DataType.VARCHAR, max_length=512)
schema.add_field(field_name="label_vector", datatype=DataType.FLOAT_VECTOR, dim=512)
schema.add_field(field_name="fusion_vector", datatype=DataType.FLOAT_VECTOR, dim=512)


# 3.3. Prepare index parameters
index_params = client.prepare_index_params()

# 3.4. Add indexes
index_params.add_index(
    field_name="id"
)

index_params.add_index(
    field_name="tag_class"
)

index_params.add_index(
    field_name="tag_class_vector",
    metric_type="COSINE"
)

index_params.add_index(
    field_name="image_vector",
    metric_type="COSINE"
)

index_params.add_index(
    field_name="label"
)

index_params.add_index(
    field_name="label_vector",
    metric_type="COSINE"
)


index_params.add_index(
    field_name="fusion_vector", 
    index_type="AUTOINDEX",
    metric_type="COSINE"
)

# 3.5. Create a collection
client.create_collection(
    collection_name="TieTa_rag",
    schema=schema,
    index_params=index_params
)