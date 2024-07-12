# romantic/humorous factual

import pickle
import sys

import torch
from tqdm import tqdm
sys.path.append("/home/liwc/wxp/refercode/GeDi_Final")
import clip

if __name__ == "__main__":
    # 打开flickrstyle10k的训练集
    data_flickrstyle10k_train_path = "/home/liwc/wxp/refercode/GeDi_Final/data/FlickrStyle10k/FlickrStyle10k_ViT-L_14_train.pkl"
    with open(data_flickrstyle10k_train_path, "rb") as f:
        data_flickrstyle10k_train = pickle.load(f)

    # CLIP
    clip_model_type = "ViT-L/14"
    device = "cuda:1"
    clip_model_name = clip_model_type.replace('/', '_')
    clip_model, preprocess = clip.load(clip_model_type, device=device, jit=False)


    # 处理
    with torch.no_grad():
        for i in tqdm(range(len(data_flickrstyle10k_train))):
            caption = data_flickrstyle10k_train[i]["caption"]
            try:
                caption_tokens = clip.tokenize(caption).to(device)
            except:
                caption_tokens = clip.tokenize(caption[:100]).to(device)
            prefix = clip_model.encode_text(caption_tokens).cpu()
            data_flickrstyle10k_train[i]["prefix"] = prefix


    train_data_fu_text = []
    train_data_ro_text = []
    for i in range(len(data_flickrstyle10k_train)):
        if data_flickrstyle10k_train[i]["style"] == "romantic":
            train_data_ro_text.append(data_flickrstyle10k_train[i])
        elif data_flickrstyle10k_train[i]["style"] == "humorous":
            train_data_fu_text.append(data_flickrstyle10k_train[i])
        else:
            train_data_ro_text.append(data_flickrstyle10k_train[i])
            train_data_fu_text.append(data_flickrstyle10k_train[i])


    # 保存
    out_path_train_ro_text = "/home/liwc/wxp/refercode/GeDi_Final/data/dataset_train/Flickrstyle10k_ro_text_ViT-L_14_train.pkl"
    out_path_train_fu_text = "/home/liwc/wxp/refercode/GeDi_Final/data/dataset_train/Flickrstyle10k_fu_text_ViT-L_14_train.pkl"
    with open(out_path_train_ro_text, "wb") as f:
        pickle.dump(train_data_ro_text, f)
    with open(out_path_train_fu_text, "wb") as f:
        pickle.dump(train_data_fu_text, f)