# Senticap，划分数据集，计算并保留图像特征
import sys
import torch
import skimage.io as io
sys.path.append("/home/liwc/wxp/refercode/GeDi_Final")
import clip
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
import re
import random
from utils import caption_process

random.seed(42)

def process_senticap_for_train(dataset, preprocess, clip_model, device):
    dataset_return = []
    idx = 0
    for i in tqdm(range(len(dataset))):
        # 判断是否为图像名称
        img_text = dataset[i]
        if os.path.exists(f"/home/liwc/wxp/dataset/MSCOCO/train2014/" + img_text["filename"]):
            filename = f"/home/liwc/wxp/dataset/MSCOCO/train2014/" + img_text["filename"]
        elif os.path.exists(f"/home/liwc/wxp/dataset/MSCOCO/val2014/" + img_text["filename"]):
            filename = f"/home/liwc/wxp/dataset/MSCOCO/val2014/" + img_text["filename"]
        else:
            continue

        # 图像embedding
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()


        for j in range(len(img_text['sentences'])):
            img_caption_info = {}
            img_caption_info["idx"] = idx
            img_caption_info["filename"] = filename
            img_caption_info["prefix"] = prefix
            img_caption_info["caption"] = caption_process(img_text['sentences'][j]['raw'])
            img_caption_info["style"] = "positive" if img_text['sentences'][j]['sentiment'] == 1 else "negative"
            img_caption_info["imgid"] = img_text['imgid']
            dataset_return.append(img_caption_info)
            
            idx = idx + 1

    return dataset_return

def process_senticap_for_test(dataset, preprocess, clip_model, device):
    dataset_return = []
    idx = 0
    for i in tqdm(range(len(dataset))):
        # 判断是否为图像名称
        img_text = dataset[i]
        if os.path.exists(f"/home/liwc/wxp/dataset/MSCOCO/train2014/" + img_text["filename"]):
            filename = f"/home/liwc/wxp/dataset/MSCOCO/train2014/" + img_text["filename"]
        elif os.path.exists(f"/home/liwc/wxp/dataset/MSCOCO/val2014/" + img_text["filename"]):
            filename = f"/home/liwc/wxp/dataset/MSCOCO/val2014/" + img_text["filename"]
        else:
            continue

        # 图像embedding
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()

        for style in ['positive', 'negative']:
            img_caption_info = {}
            # 图像序号
            img_caption_info["idx"] = idx
            # 图像名称
            img_caption_info["filename"] = filename
            # 图像embedding
            img_caption_info["prefix"] = prefix
            # 图像image_id
            img_caption_info["imgid"] = img_text['imgid']
            # 图像风格
            img_caption_info["style"] = style
            # 图像描述
            img_caption_info["caption"] = []
            for j in range(len(img_text['sentences'])):
                if img_caption_info["style"] == 'positive' and img_text['sentences'][j]['sentiment'] == 1:
                    img_caption_info["caption"].append(caption_process(img_text['sentences'][j]['raw']))
                if img_caption_info["style"] == 'negative' and img_text['sentences'][j]['sentiment'] == 0:
                    img_caption_info["caption"].append(caption_process(img_text['sentences'][j]['raw']))
            # 保存并计数
            if len(img_caption_info["caption"]):
                dataset_return.append(img_caption_info)
                idx = idx + 1
    return dataset_return


if __name__ == '__main__':
    # 获取数据
    with open('/home/liwc/wxp/dataanno/Senticap/senticap_dataset.json', 'r') as f:
        data = json.load(f)
        data = data['images']

    # 划分数据集
    train_data = [item for item in data if item["split"] == "train"]
    val_data = [item for item in data if item["split"] == "val"]
    test_data = [item for item in data if item["split"] == "test"]
    random.shuffle(val_data)
    train_data = train_data+val_data[100:]
    val_data = val_data[:100]
    print("Senticap split - num of train data: " + str(len(train_data)))
    print("Senticap split - num of test data : " + str(len(test_data)))
    print("Senticap split - num of val data : " + str(len(val_data)))

    # clip图像编码器
    device = torch.device("cuda:0")
    clip_model_type = "ViT-L/14"
    clip_model_name = clip_model_type.replace('/', '_')
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)

    # 数据处理
    train_data = process_senticap_for_train(train_data, preprocess, clip_model, device)
    test_data = process_senticap_for_test(test_data, preprocess, clip_model, device)
    val_data = process_senticap_for_test(val_data, preprocess, clip_model, device)
    print("Senticap - num of train data processed: " + str(len(train_data)))
    print("Senticap - num of test data processed: " + str(len(test_data)))
    print("Senticap - num of val data processed: " + str(len(val_data)))

    # 保存数据
    out_path_train = f"/home/liwc/wxp/refercode/GeDi_Final/data/Senticap/Senticap_{clip_model_name}_train.pkl"
    out_path_test = f"/home/liwc/wxp/refercode/GeDi_Final/data/Senticap/Senticap_{clip_model_name}_test.pkl"
    out_path_val = f"/home/liwc/wxp/refercode/GeDi_Final/data/Senticap/Senticap_{clip_model_name}_val.pkl"
    with open(out_path_train, "wb") as file:
        pickle.dump(train_data, file)
    with open(out_path_test, "wb") as file:
        pickle.dump(test_data, file)
    with open(out_path_val, "wb") as file:
        pickle.dump(val_data, file)



