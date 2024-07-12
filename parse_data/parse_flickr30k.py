# Flickr30k，划分数据集，计算并保留图像特征
import json
import re
from tqdm import tqdm
import os
import torch
import skimage.io as io
import sys
sys.path.append("/home/liwc/wxp/refercode/GeDi_Final")
import clip
from PIL import Image
import pickle
from utils import caption_process

def process_senticap_for_train(dataset, preprocess, clip_model, device):
    dataset_return = []
    for i in tqdm(range(len(dataset))):
        # 判断是否为图像名称
        img_text = dataset[i]
        if os.path.exists(f"/home/liwc/wxp/dataset/flickr30k-images/" + img_text["filename"]):
            filename = f"/home/liwc/wxp/dataset/flickr30k-images/" + img_text["filename"]
        else:
            continue

        # 图像embedding
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()

        for j in range(len(img_text['sentences'])):
            img_caption_info = {}
            img_caption_info["idx"] = img_text['sentences'][j]['sentid']
            img_caption_info["filename"] = filename
            img_caption_info["prefix"] = prefix
            img_caption_info["caption"] = caption_process(img_text['sentences'][j]['raw'])
            img_caption_info["imgid"] = img_text['imgid']
            dataset_return.append(img_caption_info)
        

        # if i > 10:
        #     break

    return dataset_return

def process_senticap_for_test(dataset, preprocess, clip_model, device):
    dataset_return = []
    for i in tqdm(range(len(dataset))):
        # 判断是否为图像名称
        img_text = dataset[i]
        if os.path.exists(f"/home/liwc/wxp/dataset/flickr30k-images/" + img_text["filename"]):
            filename = f"/home/liwc/wxp/dataset/flickr30k-images/" + img_text["filename"]
        else:
            continue

        # 图像embedding
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()

        img_caption_info = {}
        img_caption_info["filename"] = filename
        img_caption_info["prefix"] = prefix
        img_caption_info["imgid"] = img_text['imgid']
        img_caption_info["caption"] = []
        for j in range(len(img_text['sentences'])):
            img_caption_info["caption"].append(caption_process(img_text['sentences'][j]['raw']))

        # 保存并计数
        if len(img_caption_info["caption"]):
            dataset_return.append(img_caption_info)

        # if i > 10:
        #     break

    return dataset_return


if __name__ == "__main__":
    # 获取Flickr30k数据
    with open("/home/liwc/wxp/dataanno/Flickr30k/dataset_flickr30k.json", 'r') as f:
        data = json.load(f)
        data = data["images"]
    # 数据集划分
    train_dataset = []
    test_dataset = []
    val_dataset = []
    restval_dataset = []
    for i in range(len(data)):
        if data[i]["split"] == "train":
            train_dataset.append(data[i])
        elif data[i]["split"] == "test":
            test_dataset.append(data[i])
        elif data[i]["split"] == "val":
            val_dataset.append(data[i])
    print("Flickr30k split - num of train data: " + str(len(train_dataset)))
    print("Flickr30k split - num of test data : " + str(len(test_dataset)))
    print("Flickr30k split - num of val data : " + str(len(val_dataset)))

    # clip图像编码器
    device = torch.device("cuda:0")
    clip_model_type = "ViT-L/14"
    clip_model_name = clip_model_type.replace('/', '_')
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)

    # 数据处理
    train_dataset_processed = process_senticap_for_train(train_dataset, preprocess, clip_model, device)
    test_dataset_processed = process_senticap_for_test(test_dataset, preprocess, clip_model, device)
    val_dataset_processed = process_senticap_for_test(val_dataset, preprocess, clip_model, device)
    print("Flickr30k - num of train data processed: " + str(len(train_dataset_processed)))
    print("Flickr30k - num of test data processed: " + str(len(test_dataset_processed)))
    print("Flickr30k - num of val data processed: " + str(len(val_dataset_processed)))

    # 保存数据
    out_path_train = f"/home/liwc/wxp/refercode/GeDi_Final/data/Flickr30k/Flickr30k_{clip_model_name}_train.pkl"
    out_path_test = f"/home/liwc/wxp/refercode/GeDi_Final/data/Flickr30k/Flickr30k_{clip_model_name}_test.pkl"
    out_path_val = f"/home/liwc/wxp/refercode/GeDi_Final/data/Flickr30k/Flickr30k_{clip_model_name}_val.pkl"
    with open(out_path_train, "wb") as file:
        pickle.dump(train_dataset_processed, file)
    with open(out_path_test, "wb") as file:
        pickle.dump(test_dataset_processed, file)
    with open(out_path_val, "wb") as file:
        pickle.dump(val_dataset_processed, file)
