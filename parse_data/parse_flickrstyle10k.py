# FlickrStyle10k，划分数据集，计算并保留图像特征
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

def process_flickrstyle_for_train(dataset, preprocess, clip_model, device):
    dataset_return = []
    idx = 0
    # 风格
    styles = ["romantic", "humorous", "factual"]

    for i in tqdm(range(len(dataset))):
        # 图像名称
        img_text = dataset[i]
        filename = "/home/liwc/wxp/dataset/flickr30k-images/" + img_text["filename"]

        # 图像embedding
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()

        # 0 humor 1 roman
        for j in range(len(styles)):
            img_caption_info = {}
            # 图像序号
            img_caption_info["idx"] = idx
            # 图像名称
            img_caption_info["filename"] = filename

            # 图像描述
            img_caption_info["caption"] = caption_process(img_text["caption_"+styles[j]])
            # 图像embedding
            img_caption_info["prefix"] = prefix

            # 图像风格
            img_caption_info["style"] = styles[j]
            # 图像image_id
            img_caption_info["imgid"] = 0

            if not img_caption_info["caption"] == "":
                dataset_return.append(img_caption_info)
                idx = idx + 1

    return dataset_return

def process_flickrstyle_for_test(dataset, preprocess, clip_model, device):
    dataset_return = []
    idx = 0
    # 风格
    styles = ["romantic", "humorous", "factual"]

    for i in tqdm(range(len(dataset))):
        # 图像名称
        img_text = dataset[i]
        filename = "/home/liwc/wxp/dataset/flickr30k-images/" + img_text["filename"]

        # 图像embedding
        image = io.imread(filename)
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            prefix = clip_model.encode_image(image).cpu()

        # 0 humor 1 roman
        for j in range(len(styles)):
            img_caption_info = {}
            # 图像序号
            img_caption_info["idx"] = idx
            # 图像名称
            img_caption_info["filename"] = filename
            # 图像embedding
            img_caption_info["prefix"] = prefix
            # 图像描述
            img_caption_info["caption"] = [caption_process(img_text["caption_"+styles[j]])]
            # 图像风格
            img_caption_info["style"] = styles[j]
            # 图像image_id
            img_caption_info["imgid"] = 0

            if not img_caption_info["caption"][0] == "":
                dataset_return.append(img_caption_info)
                idx = idx + 1

    return dataset_return


def main(clip_model_type: str, device='cpu'):
    # 准备
    device = torch.device(device)
    clip_model_name = clip_model_type.replace('/', '_')
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)

    # 读取事实描述/风格描述
    data_caption_factual = '/home/liwc/wxp/dataanno/Flickr30k/dataset_flickr30k.json'
    with open(data_caption_factual, 'r') as f:
        file_caption_factual = json.load(f)
        file_caption_factual = file_caption_factual["images"]
    data_caption_humorous = "/home/liwc/wxp/dataanno/FlickrStyle10k/humor/funny_train.txt"
    with open(data_caption_humorous, "r") as file:
        file_caption_humorous = file.read().splitlines()
    data_caption_romantic = "/home/liwc/wxp/dataanno/FlickrStyle10k/romantic/romantic_train.txt"
    with open(data_caption_romantic, "r") as file:
        file_caption_romantic = file.read().splitlines()

    # 读取文件名
    file_names = []
    with open('/home/liwc/wxp/dataanno/FlickrStyle10k/humor/train.p', 'r') as f:
        file_img = f.read().splitlines()
        for i in range(len(file_img)):
            img_name = file_img[i]
            if not img_name.endswith(".jpg"):
                continue
            img_name = img_name.split('_')[0]
            img_name = ''.join(re.findall(r'\d+', img_name)) + '.jpg'
            file_names.append(img_name)

    # 数据合在一起
    dataset = []
    for i in range(len(file_names)):
        data_sample = {}
        data_sample["filename"] = file_names[i]
        data_sample["caption_humorous"] = file_caption_humorous[i]
        data_sample["caption_romantic"] = file_caption_romantic[i]
        data_sample["caption_factual"] = ""
        for j in range(len(file_caption_factual)):
            if file_caption_factual[j]["filename"] == data_sample["filename"]:
                data_sample["caption_factual"] = file_caption_factual[j]["sentences"][0]['raw']
                break
        dataset.append(data_sample)

        # if i > 100:
        #     break

    # 划分数据集
    random.shuffle(dataset)
    train_data = dataset[0:6000]
    test_data = dataset[6000:]

    # 处理数据集格式，训练集将ref拆到多个样本中，测试集、验证集则不拆
    train_data = process_flickrstyle_for_train(train_data, preprocess, clip_model, device)
    test_data = process_flickrstyle_for_test(test_data, preprocess, clip_model, device)
    print("num of train data : " + str(len(train_data)))
    print("num of test data : " + str(len(test_data)))

    out_path_train = f"/home/liwc/wxp/refercode/GeDi_Final/data/FlickrStyle10k/FlickrStyle10k_{clip_model_name}_train.pkl"
    out_path_test = f"/home/liwc/wxp/refercode/GeDi_Final/data/FlickrStyle10k/FlickrStyle10k_{clip_model_name}_test.pkl"
    with open(out_path_train, "wb") as file:
        pickle.dump(train_data, file)
    with open(out_path_test, "wb") as file:
        pickle.dump(test_data, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clip_model_type', default="ViT-L/14", choices=('RN50', 'RN101', 'RN50x4', 'ViT-B/32'))
    parser.add_argument('--device', default="cuda:1")
    args = parser.parse_args()
    exit(main(args.clip_model_type, args.device))