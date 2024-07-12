import argparse
import math
import os
import pickle
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Tuple, Optional, Union
from torch.utils.tensorboard import SummaryWriter


from tqdm import tqdm


from transformers import GPT2Tokenizer, AdamW, get_linear_schedule_with_warmup

from torch.utils.data import Dataset, DataLoader

random.seed(1)
torch.manual_seed(1)

# log_dir = "./result/writer3"  # 日志保存路径
# writer = SummaryWriter(log_dir)

class textCNN(nn.Module):
    def __init__(self, kernel_num, vocab_size, kernel_size, embed_dim, dropout, class_num):
        super(textCNN, self).__init__()
        ci = 1  # input chanel size
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.conv11 = nn.Conv2d(ci, kernel_num, (kernel_size[0], embed_dim))
        self.conv12 = nn.Conv2d(ci, kernel_num, (kernel_size[1], embed_dim))
        self.conv13 = nn.Conv2d(ci, kernel_num, (kernel_size[2], embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_size) * kernel_num, class_num)

    def init_embed(self, embed_matrix):
        self.embed.weight = nn.Parameter(torch.Tensor(embed_matrix))

    @staticmethod
    def conv_and_pool(x, conv):
        # x: (batch, 1, sentence_length,  )
        x = conv(x)
        # x: (batch, kernel_num, H_out, 1)
        x = F.relu(x.squeeze(3))
        # x: (batch, kernel_num, H_out)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        #  (batch, kernel_num)
        return x

    def forward(self, x):
        # x: (batch, sentence_length)
        x = self.embed(x)
        # x: (batch, sentence_length, embed_dim)
        # TODO init embed matrix with pre-trained
        x = x.unsqueeze(1)
        # x: (batch, 1, sentence_length, embed_dim)
        x1 = self.conv_and_pool(x, self.conv11)  # (batch, kernel_num)
        x2 = self.conv_and_pool(x, self.conv12)  # (batch, kernel_num)
        x3 = self.conv_and_pool(x, self.conv13)  # (batch, kernel_num)
        x = torch.cat((x1, x2, x3), 1)  # (batch, 3 * kernel_num)
        x = self.dropout(x)
        logit = self.fc1(x)
        return logit

    def get_batch_captions_style_scores(self, captions, tokenizer, device):
        input_ids = tokenizer.batch_encode_plus(captions, padding=True)['input_ids']
        input_ids_ = torch.tensor(input_ids).to(device)
        logits = self.forward(input_ids_)
        probs = F.softmax(logits, dim=-1)
        predicts = logits.argmax(-1)

        return probs, predicts

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class ClassfierDataset(Dataset):
    def __len__(self) -> int:
        return len(self.text_tokens)
    
    def pad_tokens(self, item: int):
        tokens = self.text_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            # tokens = torch.cat((tokens, torch.ones(padding, dtype=torch.int64)*50256))
            self.text_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.text_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = 50256
        return tokens

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        tokens = self.pad_tokens(item)
        label = self.labels[item]
        return tokens, label

    def __init__(self, data_path,  gpt2_type, max_length):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        
        # # 训练集+测试集
        # if data_path == "./dataset/FlickrStyle/oscar_split_ViT-L_14_train_classfier.pkl":
        #     with open("./dataset/FlickrStyle/oscar_split_ViT-L_14_test_classfier.pkl", 'rb') as f:
        #         test_data = pickle.load(f)
        #         all_data = all_data + test_data



        print("Data size is %0d" % len(all_data))
        sys.stdout.flush()

        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.text_tokens, self.labels = pickle.load(f)
        else:
            self.text_tokens = []
            self.labels = []
            for i in range(len(all_data)):
                self.text_tokens.append(torch.tensor(self.tokenizer.encode(all_data[i]["text"]), dtype=torch.int64))
                self.labels.append(all_data[i]["label"])
            with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                pickle.dump([self.text_tokens, self.labels], f)

        self.seq_len = []
        for i in range(len(self.text_tokens)):
            self.seq_len.append(self.text_tokens[i].shape[0])


        self.max_seq_len = max_length

def classfier_sentiment(text_caption, model, tokenizer, device):
    tokens = torch.tensor(tokenizer.encode(text_caption), dtype=torch.int64)

    padding = 24 - tokens.shape[0]
    if padding > 0:
        tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        # tokens = torch.cat((tokens, torch.ones(padding, dtype=torch.int64)*50256))
    elif padding < 0:
        tokens = tokens[:24]
    mask = tokens.ge(0)  # mask is zero where we out of sequence
    tokens[~mask] = 50256

    tokens = tokens.unsqueeze(0).to(device)
    outputs_logit = model(tokens)

    _, predicted_labels = torch.max(outputs_logit, dim=1)

    return predicted_labels

def train(dataset: ClassfierDataset, model: textCNN, args, dataset_test:ClassfierDataset,
          lr: float = 1e-4, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = ""):

    device = torch.device(args.device)
    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    # save_config(args)

    for epoch in tqdm(range(epochs)):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        loss_epoch = 0
        num_loss = 0
        for idx, (tokens, labels) in enumerate(train_dataloader):
            model.zero_grad()
            tokens, labels = tokens.to(device), labels.to(device)
            outputs_logit = model(tokens) 
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(outputs_logit.view(-1, outputs_logit.size(-1)), labels.view(-1))
            loss = loss.view(tokens.shape[0], -1)

            loss_epoch = loss_epoch + torch.sum(loss)
            num_loss = num_loss + loss.shape[0]

            loss = torch.sum(loss) / loss.shape[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            #break # test

        # eval
        acc_max = 0
        loss_epoch_train = loss_epoch / num_loss
        loss_epoch_test, acc = eval_model(model, test_dataloader, device=device)
        if acc >=  acc_max:
            acc_max = acc
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}_maxacc.pt"),
            )
        print("epoch:" + str(epoch + 1) + 
              "\ntrain loss:" + str(loss_epoch_train.item()) +
              "\ntest loss:" + str(loss_epoch_test.item()) +
              "\nacc:" + str(acc)
              )
        
        # writer.add_scalar("Loss/train", loss_epoch_train.item(), epoch)
        # writer.add_scalar("Loss/test", loss_epoch_test.item(), epoch)
        # writer.add_scalar("acc/test", acc, epoch)


    return model

def eval_model(model, test_dataloader, device):
    loss_epoch = 0
    num_loss = 0
    acc_epoch = 0
    model.eval()
    for idx, (tokens, labels) in enumerate(test_dataloader):
        tokens, labels = tokens.to(device), labels.to(device)
        outputs_logit = model(tokens) 
        # 统计损失
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(outputs_logit.view(-1, outputs_logit.size(-1)), labels.view(-1))
        loss = loss.view(tokens.shape[0], -1) 
        loss_epoch = loss_epoch + torch.sum(loss)
        num_loss = num_loss + loss.shape[0]
        # 统计acc
        _, predicted_labels = torch.max(outputs_logit, dim=1)
        acc_epoch = acc_epoch + (predicted_labels == labels).sum().item()

    # 计算损失
    loss_epoch_test = loss_epoch / num_loss
    # 计算acc
    acc = acc_epoch / num_loss

    model.train()
    return loss_epoch_test, acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default="./dataset/FlickrStyle/oscar_split_ViT-L_14_train_classfier.pkl")
    parser.add_argument('--data_path_test', default="./dataset/FlickrStyle/oscar_split_ViT-L_14_test_classfier.pkl")
    parser.add_argument('--max_length', default=24)
    parser.add_argument('--device', default="cuda:1")
    parser.add_argument('--bs', default=20)
    parser.add_argument('--epochs', default=50)
    parser.add_argument('--out_dir', default="./result/result_classfier/flickrstyle_classfier/classfier1024/")
    parser.add_argument('--kernel_num', default=100)
    parser.add_argument('--vocab_size', default=50257)
    parser.add_argument('--embed_dim', default=1024)
    parser.add_argument('--dropout', default=0.3)
    parser.add_argument('--class_num', default=2)
    args = parser.parse_args()

    args.kernel_size = [1, 2, 3]

    if 1:
        # 数据集
        dataset = ClassfierDataset(data_path=args.data_path,  gpt2_type="gpt2", max_length=args.max_length)
        dataset_test = ClassfierDataset(data_path=args.data_path_test,  gpt2_type="gpt2", max_length=args.max_length)
        # 模型
        args.vocab_size = dataset.tokenizer.vocab_size
        model = textCNN(args.kernel_num, args.vocab_size, args.kernel_size, args.embed_dim, args.dropout, args.class_num)
        model.init_weight()
        # 训练
        model = train(dataset, model, args, dataset_test, output_dir=args.out_dir)
        # writer.close()
        
    return 0

if __name__ == "__main__":
    main()