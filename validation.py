import torch
from PIL import Image
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models
from pathlib import Path
from cn_clip.clip.utils import image_transform
import os
from cn_clip.clip.model import CLIP
import json
import numpy as np
from tqdm import tqdm
from pymilvus import MilvusClient
# from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.messages import HumanMessage
import re
from tqdm import tqdm

os.environ["QIANFAN_AK"] = "LXs3eEm55mPryp9n5PbgmBUS"
os.environ["QIANFAN_SK"] = "0um0VZ8adtNgQWClOMwJZ0OXKnqV9SIA"

CLUSTER_ENDPOINT = "https://in03-c37b3b2d61f1d1f.serverless.ali-cn-hangzhou.cloud.zilliz.com.cn"
TOKEN = "e769f024891860623ab86426431422cb34e57edd152377121d40cbced6348d19fcb0f253223bf65efe3d401c667423971c0b118a"
USER_NAME = "db_c37b3b2d61f1d1f"
PASSWORD = "12345678cdCD"
COLLECTION = "TieTa_rag"

# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']
detection_types = {'Equipment_cabinet': '设备机柜', 'switching_power_supply': '电源', 'switch_box': "配电箱", 'null': '无' }

device = "cuda:5"

vision_model = "ViT-B-16"
text_model = "RoBERTa-wwm-ext-base-chinese"

tag_class = "配电箱"

client = MilvusClient(
            uri=CLUSTER_ENDPOINT,
            token=TOKEN,
            username=USER_NAME,
            password=PASSWORD
        )

resume = "/amax/home/chendian/WEI_project/MM-main/experiments/all_dataset_train/lr_5e-5_bs_8_epochs_100_contex_100/checkpoints/epoch_latest.pt"

llm = QianfanChatEndpoint(
    model="ERNIE-Speed-128K",
    streaming=True,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "现在有content1和content2，如果content1和content2一致，则输出无异常，否则输出表面异常。只用输出“无异常”或者“表面异常”, 请不要说多余的文本，包括空格等"),
    ("user", "{input}")
])

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

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


def getNum(context):
    if context == '无':
        return 0
    else:
        return int(context[0])
    
# 得到gt的text
def get_gt_text(label2, label1, num1, num2):
    if num1 == 0 and num2 == 0:
        return "无异常"
    elif num2 < num1:
        return "数量增加"
    elif num2 > num1:
        return "数量减少"
    else:
        if label1 == label2:
            return "无异常"
        else:
            return "表面异常"

def get_result(tag_class, pre_image_path, unchecked_image_path):
    pre_image = preprocess(Image.open(pre_image_path)).unsqueeze(0).to(device)
    unchecked_image = preprocess(Image.open(unchecked_image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        tag_emd = model(None, clip.tokenize(tag_class).to(device)).squeeze().cpu().numpy().tolist()
        pre_image_embed = model(pre_image, None).squeeze().cpu().numpy().tolist()
        unchecked_image_embed = model(unchecked_image, None).squeeze().cpu().numpy().tolist()
        
        # res1 = client.search(
        #     collection_name=COLLECTION,
        #     data=[pre_image_embed],
        #     anns_field="fusion_vector",
        #     limit=1,
        #     search_params={"metric_type": "COSINE", "params": {}},
        #     partition_names=["num"],
        #     filter=f'(tag_class == "无") or (tag_class like "%{tag_class}%")',
        #     output_fields=["label"]
        # )
        
        # res2 = client.search(
        #     collection_name=COLLECTION,
        #     data=[unchecked_image_embed],
        #     anns_field="fusion_vector",
        #     limit=1,
        #     search_params={"metric_type": "COSINE", "params": {}},
        #     partition_names=["num"],
        #     filter=f'(tag_class == "无") or (tag_class like "%{tag_class}%")',
        #     output_fields=["label"]
        # )
        
        res1 = client.search(
            collection_name=COLLECTION,
            data=[pre_image_embed],
            anns_field="label_vector",
            limit=1,
            search_params={"metric_type": "COSINE", "params": {}},
            partition_names=["num"],
            filter=f'(tag_class == "无。") or (tag_class like "%{tag_class}%")',
            output_fields=["label"]
        )
        
        res2 = client.search(
            collection_name=COLLECTION,
            data=[unchecked_image_embed],
            anns_field="label_vector",
            limit=1,
            search_params={"metric_type": "COSINE", "params": {}},
            partition_names=["num"],
            filter=f'(tag_class == "无。") or (tag_class like "%{tag_class}%")',
            output_fields=["label"]
        )
        
        context1 = res1[0][0]["entity"]["label"]
        context2 = res2[0][0]["entity"]["label"]
        num1 = getNum(context1)
        num2 = getNum(context2)
        text = ""
        if num1 == 0 and num2 == 0:
            text = "无异常"
        elif num1 < num2:
            text = "数量增加"
        elif num1 > num2:
            text = "数量减少"
        else:
            
            # fusion_vector label_vector
            
            res_content1 = client.search(
                collection_name=COLLECTION,
                data=[pre_image_embed],
                anns_field="fusion_vector",
                limit=1,
                search_params={"metric_type": "COSINE", "params": {}},
                partition_names=["content"],
                filter=f'(tag_class == "无。") or (tag_class like "%{tag_class}%")',
                output_fields=["label"]
                )
            
            res_content2 = client.search(
                collection_name=COLLECTION,
                data=[unchecked_image_embed],
                anns_field="fusion_vector",
                limit=1,
                search_params={"metric_type": "COSINE", "params": {}},
                partition_names=["content"],
                filter=f'(tag_class == "无。") or (tag_class like "%{tag_class}%")',
                output_fields=["label"]
                )
            
            content_1 = res_content1[0][0]["entity"]["label"]
            content_2 = res_content2[0][0]["entity"]["label"]
            text = chain.invoke({"input": f"content1:{content_1}, content2:{content_2}"})
            return text, context1, context2, content_1, content_2
        return text, context1, context2, "", ""
    
if __name__=="__main__":
    json_file_path = "/amax/home/chendian/WEI_project/Multimodal_Annotated_Dataset/929/test_data.json"
    data_dir_path = "/amax/home/chendian/WEI_project/Multimodal_Annotated_Dataset/929/distribution_box0929-200"
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    results = []  # 创建一个空列表来存储结果
    ground_truth = []
    
    detail_results = []
    # 遍历JSON数组
    for index, item in tqdm(enumerate(data), total=len(data), desc="Processing"):
        detail_result = {}
        # 提取name1和name2
        path_unchecked_image = os.path.join(data_dir_path, item['name1'] + '.jpg')
        path_pre_image = os.path.join(data_dir_path, item['name2'] + '.jpg')
        
        # 调用处理方法
        # context1 原始图像 context2 待检测图像 content_1原始图像
        result, context1, context2, content_1, content_2 = get_result(tag_class, path_pre_image, path_unchecked_image)
        
        gt = get_gt_text(item['label2'], item['label1'], item['num1'], item['num2'])
        
        detail_result['id'] = index
        detail_result['result'] = result
        detail_result['context1'] = context1 
        detail_result['context2'] = context2
        detail_result['content_1'] = content_1
        detail_result['content_2'] = content_2
        detail_result['raw_image_name'] = item['name2']
        detail_result['unchecked_image_name'] = item['name1']
        
        detail_results.append(detail_result)
        results.append(result)
        ground_truth.append(gt)
    
    results_file_path = "/amax/home/chendian/WEI_project/langchain_project/result.json"
    gt_file_path = "/amax/home/chendian/WEI_project/langchain_project/gt.json"
    detail_file_path = "/amax/home/chendian/WEI_project/langchain_project/detail.json"
    
    with open(results_file_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    with open(gt_file_path, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=4)
        
    with open(detail_file_path, 'w', encoding='utf-8') as f:
        json.dump(detail_results, f, ensure_ascii=False, indent=4)
    
    correct_count = 0
    total_count = len(results)

    # 遍历results和ground_truth
    for result, truth in zip(results, ground_truth):
        if result == truth:
            correct_count += 1

    # 计算准确率
    accuracy = correct_count / total_count

    # 打印准确率
    print(f"准确率: {accuracy:.2f} 或 {(correct_count / total_count) * 100:.2f}%")