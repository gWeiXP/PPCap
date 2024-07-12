
# romantic humorous

import pickle


if __name__ == "__main__":
    data_fu_train_path = "/home/liwc/wxp/refercode/GeDi_Final/data/dataset_train/Flickrstyle10k_fu_text_ViT-L_14_train.pkl"
    with open(data_fu_train_path, "rb") as f:
        data_fu_train = pickle.load(f)
    data_ro_train_path = "/home/liwc/wxp/refercode/GeDi_Final/data/dataset_train/Flickrstyle10k_ro_text_ViT-L_14_train.pkl"
    with open(data_ro_train_path, "rb") as f:
        data_ro_train = pickle.load(f)
    
    data_flickrstyle10k_train = []
    for i in range(len(data_fu_train)):
        if data_fu_train[i]["style"] == "humorous":
            data_flickrstyle10k_train.append(data_fu_train[i])
    for i in range(len(data_ro_train)):
        if data_ro_train[i]["style"] == "romantic":
            data_flickrstyle10k_train.append(data_ro_train[i])

    data_flickrstyle10k_train_path = "/home/liwc/wxp/refercode/GeDi_Final/data/dataset_train/Flickrstyle10k_text_ViT-L_14_train.pkl"
    with open(data_flickrstyle10k_train_path, "wb") as f:
        pickle.dump(data_flickrstyle10k_train, f)