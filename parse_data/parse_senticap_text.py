# 将senticap训练集中每个样本的图像特征改为文本特征
import pickle
import sys

import torch
from tqdm import tqdm
sys.path.append("/home/liwc/wxp/refercode/GeDi_Final")
import clip


if __name__ == "__main__":
    # 打开senticap的训练集
    data_senticap_train_path = "/home/liwc/wxp/refercode/GeDi_Final/data/Senticap/Senticap_ViT-L_14_train.pkl"
    with open(data_senticap_train_path, "rb") as f:
        data_senticap_train = pickle.load(f)

    # CLIP
    clip_model_type = "ViT-L/14"
    device = "cuda:1"
    clip_model_name = clip_model_type.replace('/', '_')
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)

    # 处理
    with torch.no_grad():
        for i in tqdm(range(len(data_senticap_train))):
            caption = data_senticap_train[i]["caption"]
            try:
                caption_tokens = clip.tokenize(caption).to(device)
            except:
                caption_tokens = clip.tokenize(caption[:100]).to(device)
            prefix = clip_model.encode_text(caption_tokens).cpu()
            data_senticap_train[i]["prefix"] = prefix

    # 保存
    train_data_text = data_senticap_train
    out_path_train_text = "/home/liwc/wxp/refercode/GeDi_Final/data/dataset_train/Senticap_text_ViT-L_14_train.pkl"
    with open(out_path_train_text, "wb") as f:
        pickle.dump(train_data_text, f)