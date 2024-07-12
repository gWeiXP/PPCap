# MSCOCO， 去掉训练集中senticap的测试集
import pickle

if __name__ == "__main__":
    data_mscoco_train_path = "/home/liwc/wxp/refercode/GeDi_Final/data/MSCOCO/MSCOCO_ViT-L_14_train.pkl"
    with open(data_mscoco_train_path, "rb") as f:
        data_mscoco_train = pickle.load(f)
    
    data_senticap_test_path = "/home/liwc/wxp/refercode/GeDi_Final/data/Senticap/Senticap_ViT-L_14_test.pkl"
    with open(data_senticap_test_path, "rb") as f:
        data_senticap_test = pickle.load(f)

    filenames_in_senticap_test = []
    for i in range(len(data_senticap_test)):
        filenames_in_senticap_test.append(data_senticap_test[i]["filename"])

    
    data_mscoco_train_save = []
    for i in range(len(data_mscoco_train)):
        if not data_mscoco_train[i]["filename"] in filenames_in_senticap_test:
            data_mscoco_train_save.append(data_mscoco_train[i])

    
    data_mscoco_train_save_path = "/home/liwc/wxp/refercode/GeDi_Final/data/dataset_train/MSCOCO_removed_ViT-L_14_train.pkl"
    with open(data_mscoco_train_save_path, "wb") as f:
        pickle.dump(data_mscoco_train_save, f)