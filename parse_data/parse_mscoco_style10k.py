import pickle
import sys

import torch
sys.path.append("/home/liwc/wxp/refercode/GeDi_Final")
from utils import noise_injection

if __name__ == "__main__":
    # 是否加噪音、是否归一化
    add_noise = True
    do_norm = True
    device = "cuda:0"
    remove = False
    if add_noise:
        do_norm = True

    # 打开数据
    if remove:
        data_mscoco_removed_train_path = "/home/liwc/wxp/refercode/GeDi_Final/data/dataset_train/MSCOCO_removed_ViT-L_14_train.pkl"
    else:
        data_mscoco_removed_train_path = "/home/liwc/wxp/refercode/GeDi_Final/data/MSCOCO/MSCOCO_ViT-L_14_train.pkl"
    with open(data_mscoco_removed_train_path, "rb") as f:
        data_mscoco_removed_train = pickle.load(f)


    data_flickrstyle10k_text_train_path = "/home/liwc/wxp/refercode/GeDi_Final/data/dataset_train/Flickrstyle10k_text_ViT-L_14_train.pkl"
    with open(data_flickrstyle10k_text_train_path, "rb") as f:
        data_flickrstyle10k_text_train = pickle.load(f)


    data_flickrstyle10k_text_train_add = []

    with torch.no_grad():
        # MSCOCO，事实描述，图像特征，归一化处理
        if do_norm:
            for i in range(len(data_mscoco_removed_train)):
                prefix = data_mscoco_removed_train[i]["prefix"]
                prefix = prefix / prefix.norm(2, -1, keepdim=True)
                data_mscoco_removed_train[i]["prefix"] = prefix
            

        # FlickrStyle10k，风格描述，文本特征，归一化处理，并加入噪音
        for i in range(len(data_flickrstyle10k_text_train)):
            item = data_flickrstyle10k_text_train[i].copy()  
            del item["style"]

            if do_norm:
                prefix = item["prefix"].to(device)
                prefix = prefix / prefix.norm(2, -1, keepdim=True)
                if add_noise:
                    prefix = noise_injection(prefix, 0.016, modality_offset=None, uniform_noise=False, dont_norm=False)
                prefix = prefix.cpu()
                item["prefix"] = prefix
            
            data_flickrstyle10k_text_train_add.append(item)

    data_mscoco_style10k_train = data_mscoco_removed_train + data_flickrstyle10k_text_train_add

    if remove:
        if add_noise:
            data_mscoco_style10k_train_path = "/home/liwc/wxp/refercode/GeDi_Final/data/dataset_train/MSCOCO_style10k_norm_noise_ViT-L_14_train.pkl"
        elif do_norm:
            data_mscoco_style10k_train_path = "/home/liwc/wxp/refercode/GeDi_Final/data/dataset_train/MSCOCO_style10k_norm_ViT-L_14_train.pkl"
        else:
            data_mscoco_style10k_train_path = "/home/liwc/wxp/refercode/GeDi_Final/data/dataset_train/MSCOCO_style10k_ViT-L_14_train.pkl"
    else:
        if add_noise:
            data_mscoco_style10k_train_path = "/home/liwc/wxp/refercode/GeDi_Final/data/dataset_train/MSCOCO_style10k_norm_noise_notremove_ViT-L_14_train.pkl"
        elif do_norm:
            data_mscoco_style10k_train_path = "/home/liwc/wxp/refercode/GeDi_Final/data/dataset_train/MSCOCO_style10k_norm_notremove_ViT-L_14_train.pkl"
        else:
            data_mscoco_style10k_train_path = "/home/liwc/wxp/refercode/GeDi_Final/data/dataset_train/MSCOCO_style10k_notremove_ViT-L_14_train.pkl"
    
    
    with open(data_mscoco_style10k_train_path, "wb") as f:
        pickle.dump(data_mscoco_style10k_train, f)