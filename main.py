import torch
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from pathlib import Path
from cn_clip.clip.utils import image_transform
import os
from cn_clip.clip.model import convert_weights, CLIP
import json
import numpy as np
from tqdm import tqdm
from pymilvus import MilvusClient
# from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.messages import HumanMessage

os.environ["QIANFAN_AK"] = "LXs3eEm55mPryp9n5PbgmBUS"
os.environ["QIANFAN_SK"] = "0um0VZ8adtNgQWClOMwJZ0OXKnqV9SIA"

CLUSTER_ENDPOINT = "https://in03-c37b3b2d61f1d1f.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn"
TOKEN = "e769f024891860623ab86426431422cb34e57edd152377121d40cbced6348d19fcb0f253223bf65efe3d401c667423971c0b118a"

COLLECTION = "TieTa_rag"
# USER_NAME = "db_c395bc82883c484"
# PASSWORD = "tietaTIETA123"

# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']
detection_types = {'Equipment_cabinet': '设备机柜', 'switching_power_supply': '电源', 'switch_box': "配电箱", 'null': '无' }

device = "cuda:5"

vision_model = "ViT-B-16"
text_model = "RoBERTa-wwm-ext-base-chinese"

pre_image_path = "/amax/home/chendian/WEI_project/Multimodal_Annotated_Dataset/625/switch_box/2wLi7_1718075856993_00000041060346200097_00000041060346200097_00000000001310000002_20240608111718307.jpg"
# 一个配电箱：箱门打开,指示灯亮,无杂物,油机未接入
unchecked_image_path = "/amax/home/chendian/WEI_project/Multimodal_Annotated_Dataset/625/switch_box/3sQmG_1718075794870_00000041919246200279_00000041919246200279_00000000001310000001_20240608111624503.jpg"
# 一个配电箱：箱门关闭,指示灯亮,无杂物,油机未接入
# unchecked_image_path = "/amax/home/chendian/WEI_project/Multimodal_Annotated_Dataset/625/switch_box/4hTo6_1718075309438_00000041092646200188_00000041092646200188_00000000001310000003_20240608110824577.jpg"
# # 无
tag_class = "配电箱"


resume = "/amax/home/chendian/WEI_project/MM-main/experiments/all_dataset_train/lr_5e-5_bs_8_epochs_100_contex_100/checkpoints/epoch_latest.pt"

# llm = ChatOpenAI(
#     openai_api_key='sk-OTuR9FaiCwrjDvjW0ofqgQMmp423VUxKbFHcfgnqNe5axNL7',
#     base_url='https://api.chatanywhere.tech/v1',
#     model='gpt-3.5-turbo'
#     )

llm = QianfanChatEndpoint(
    streaming=True,
)

# llm = Ollama(base_url="http://10.126.62.69:11434",
#              model="llava:13b")

prompt = ChatPromptTemplate.from_messages([
    ("system", "现在有content1和content2，如果content1和content2一致，则输出无异常，否则输出表面异常。注意你只需要输出：无异常/表面异常"),
    ("user", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

def getNum(context):
    if context == '无':
        return 0
    else:
        return int(context[0])

    
if __name__=="__main__":
    vision_model_config_file = Path(__file__).parent / "cn_clip" / f"clip/model_configs/{vision_model.replace('/', '-')}.json"
    print('Loading vision model config from', vision_model_config_file)
    assert os.path.exists(vision_model_config_file)
    
    text_model_config_file = Path(__file__).parent / "cn_clip" / f"clip/model_configs/{text_model.replace('/', '-')}.json"
    print('Loading text model config from', text_model_config_file)
    assert os.path.exists(text_model_config_file)
    
    with open(vision_model_config_file, 'r') as fv, open(text_model_config_file, 'r') as ft:
        model_info = json.load(fv)
        if isinstance(model_info['vision_layers'], str):
            model_info['vision_layers'] = eval(model_info['vision_layers'])        
        for k, v in json.load(ft).items():
            model_info[k] = v

    model = CLIP(**model_info).to(device)
    checkpoint = torch.load(resume, map_location={'cuda:0':'cuda:5'})
    sd = checkpoint["state_dict"]
    if next(iter(sd.items()))[0].startswith('module'):
        sd = {k[len('module.'):]: v for k, v in sd.items() if "bert.pooler" not in k}
    model.load_state_dict(sd)
    print(f"=> loaded checkpoint '{resume}' (epoch {checkpoint['epoch']} @ {checkpoint['step']} steps)")

    model.eval()

    preprocess = image_transform()
    
    
    pre_image = preprocess(Image.open(pre_image_path)).unsqueeze(0).to(device)
    unchecked_image = preprocess(Image.open(unchecked_image_path)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        client = MilvusClient(
            uri=CLUSTER_ENDPOINT,
            token=TOKEN,
            # username=USER_NAME,
            # password=PASSWORD
        )
        tag_emd = model(None, clip.tokenize(tag_class).to(device)).squeeze().cpu().numpy().tolist()
        pre_image_embed = model(pre_image, None).squeeze().cpu().numpy().tolist()
        unchecked_image_embed = model(unchecked_image, None).squeeze().cpu().numpy().tolist()
        
        res1 = client.search(
            collection_name=COLLECTION,
            data=[pre_image_embed],
            anns_field="fusion_vector",
            limit=1,
            search_params={"metric_type": "COSINE", "params": {}},
            partition_names=["num"],
            filter=f'(tag_class == "无") or (tag_class like "%{tag_class}%")',
            output_fields=["label"]
        )
        
        res2 = client.search(
            collection_name=COLLECTION,
            data=[unchecked_image_embed],
            anns_field="fusion_vector",
            limit=1,
            search_params={"metric_type": "COSINE", "params": {}},
            partition_names=["num"],
            filter=f'(tag_class == "无") or (tag_class like "%{tag_class}%")',
            output_fields=["label"]
        )
        
        context1 = res1[0][0]["entity"]["label"]
        context2 = res2[0][0]["entity"]["label"]
        num1 = getNum(context1)
        num2 = getNum(context2)
        if num1 == 0 and num2 == 0:
            print("无异常")
        elif num1 < num2:
            print("数量增加")
        elif num1 > num2:
            print("数量减少")
        else:
            res_content1 = client.search(
                collection_name=COLLECTION,
                data=[pre_image_embed],
                anns_field="fusion_vector",
                limit=1,
                search_params={"metric_type": "COSINE", "params": {}},
                partition_names=["content"],
                filter=f'(tag_class == "无") or (tag_class like "%{tag_class}%")',
                output_fields=["label"]
                )
            
            res_content2 = client.search(
                collection_name=COLLECTION,
                data=[unchecked_image_embed],
                anns_field="fusion_vector",
                limit=5,
                search_params={"metric_type": "COSINE", "params": {}},
                partition_names=["content"],
                filter=f'(tag_class == "无") or (tag_class like "%{tag_class}%")',
                output_fields=["label"]
                )
            content_1 = res_content1[0][0]["entity"]["label"]
            content_2 = res_content2[0][0]["entity"]["label"]
            print(content_1)
            print(content_2)
            text = chain.invoke({"input": f"content1:{content_1}, content2:{content_2}"})
            print(text)