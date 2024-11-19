import json
import random
from sklearn.model_selection import train_test_split
import os
from PIL import Image

class Dataset:
    def __init__(self, json_path, img_path, transforms, test_split_size=0.2, train=True):
        self.json_path = json_path
        self.img_path = img_path
        self.data = None
        self.train = train
        self.test_split_size = test_split_size
        self.transforms = transforms
        
        self.ref_imgs = []
        self.target_imgs = []
        self.target_labels = []
        self.target_nums = []
        self.ref_names = []
        
        with open(self.json_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)
            
        self.init_data(self.data)
        
        if self.train:
            self.init_train()
        else:
            self.init_test()
        
    def init_data(self, data):
        if self.data is None:
            raise ValueError("数据未加载，请先调用 load_data 方法")
        self.dataset = []
        for item in self.data:
            new_item1 = {
                "img_name": item["name1"],
                "ref_img": item["name2"],
                "target_label": item["label2"],
                "target_img": item["name1"],
                "target_num": item["num2"]
            }
            new_item2 = {
                "img_name": item["name2"],
                "ref_img": item["name1"],
                "target_label": item["label1"],
                "target_img": item["name2"],
                "target_num": item["num1"]
            }
            
            self.dataset.append(new_item1)
            self.dataset.append(new_item2)
            
        random.shuffle(self.dataset)
        self.dataset_train, self.dataset_test = train_test_split(self.dataset, test_size=self.test_split_size)
        
    def init_train(self):
        for item in self.dataset_train:
            ref_img_path = os.path.join(self.img_path, item['ref_img'] + ".jpg")
            self.ref_imgs.append(ref_img_path)
            target_img_path = os.path.join(self.img_path, item['target_img'] + ".jpg")
            self.target_imgs.append(target_img_path)
            self.target_labels.append(item['target_label'])
            self.target_nums.append(item['target_num'])
            self.ref_names.append(item['img_name'])
            
    def init_test(self):
        for item in self.dataset_test:
            ref_img_path = os.path.join(self.img_path, item['ref_img'] + ".jpg")
            self.ref_imgs.append(ref_img_path)
            target_img_path = os.path.join(self.img_path, item['target_img'] + ".jpg")
            self.target_imgs.append(target_img_path)
            self.target_labels.append(item['target_label'])
            self.target_nums.append(item['target_num'])
            self.ref_names.append(item['img_name'])
    
    def return_data(self, idx):
        ref_img_path = str(self.ref_imgs[idx])
        ref_img = self.transforms(Image.open(ref_img_path))
        target_img_path = str(self.target_imgs[idx])
        target_img = self.transforms(Image.open(target_img_path))
        target_label = self.target_labels[idx]
        target_num = self.target_nums[idx]
        img_name = self.ref_names[idx]
        return ref_img, target_img, target_label, target_num, img_name
    
    def __getitem__(self, idx):
        return self.return_data(idx)
    
    def __len__(self):
        if self.train:
            return len(self.dataset_train)
        else:
            return len(self.dataset_test)

# 使用示例
if __name__ == "__main__":
    dataset_path = '/amax/home/chendian/WEI_project/Multimodal_Annotated_Dataset/929/test_data.json'
    img_path = "/amax/home/chendian/WEI_project/Multimodal_Annotated_Dataset/929/10.12_distribution_after_mod"
    
    dataset = Dataset(dataset_path)
    dataset.load_data()
    dataset1, dataset2 = dataset.split_data()
    
    dataset.save_data(dataset1, output_path1)
    dataset.save_data(dataset2, output_path2)