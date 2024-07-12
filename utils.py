import math
import os
import re
import pickle
import random
import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm, trange
import sys

from pytorch_pretrained_bert.modeling import BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from classier_train import textCNN, classfier_sentiment
from classier_test_input import BertLSTM, sentence_style_cls

def caption_process(caption):
    caption = caption.replace("\n", "").replace("\"", "").replace("\r", "").replace("\t", "").lower()
    caption = re.sub(r'[^\x00-\x7F]', '', caption)
    caption = caption.replace(" .", ".")
    return caption

class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return self.prefixes.shape[0]

    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
            # tokens = torch.cat((tokens, torch.ones(padding, dtype=torch.int64)*50256))
            self.captions_tokens[item] = tokens
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[item] = tokens
        mask = tokens.ge(0)  # mask is zero where we out of sequence
        tokens[~mask] = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)  # adding prefix mask
        return tokens, mask

    def __getitem__(self, item: int):
        prefix = self.prefixes[item]
        # prefix = prefix / prefix.norm(2, -1)

        if self.train_or_test == 'train':
            tokens, mask = self.pad_tokens(item)
            style_token = self.style_tokens[item]
            match_label = self.match_labels[item]
            return tokens, mask, prefix, style_token, match_label
            
        if self.train_or_test == 'test':
            style = self.style[item]
            caption = '\n'.join(self.captions[item])
            # caption = self.captions[item]
            imgpath = self.imgpath[item]
            idx = self.idxs[item]
            return prefix, style, caption, imgpath, idx

    def __init__(self, args, data_path, train_or_test='train', tokenizer=None):
        self.train_or_test = train_or_test
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = '[PAD]'
        self.prefix_length = args.prefix_length # 10 vs 4
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)

        # 数据处理
        self.prefixes = []  # num * 768
        captions_raw = []
        self.style = []
        self.match_labels = []
        self.imgpath = []
        self.idxs = []

        if args.code_0 == " factual" or args.code_1 == " factual":
            nagetive_map = {
                    "positive": "negative",
                    "negative": "positive",
                    "humorous": " factual",
                    "romantic": " factual",
                    "factual": " "+args.teststyle}
        else:
            nagetive_map = {
                    "positive": "negative",
                    "negative": "positive",
                    "humorous": " romantic",
                    "romantic": " humorous"}

        for i in range(len((all_data))):
            if train_or_test == "train" or all_data[i]['style'] == args.teststyle:
                # 图像
                image_path = all_data[i]['filename']
                if os.path.exists(image_path):
                    filename = image_path
                if os.path.exists(f"/home/liwc/wxp/dataset/MSCOCO/train2014/" + image_path):
                    filename = f"/home/liwc/wxp/dataset/MSCOCO/train2014/" + image_path
                elif os.path.exists(f"/home/liwc/wxp/dataset/MSCOCO/val2014/" + image_path):
                    filename = f"/home/liwc/wxp/dataset/MSCOCO/val2014/" + image_path

                # 正样本
                self.prefixes.append(all_data[i]['prefix'])
                # captions_raw.append("this image shows " + all_data[i]['caption'])
                captions_raw.append(all_data[i]['caption'])
                if all_data[i]['style'] in ["positive", "negative"]:
                    self.style.append(all_data[i]['style'])
                elif all_data[i]['style'] in ["humorous", "romantic", "factual"]:
                    self.style.append(" "+all_data[i]['style'])
                self.match_labels.append(1)
                self.imgpath.append(filename)
                if train_or_test == "test":
                    self.idxs.append(all_data[i]['idx'])
                if train_or_test == "train":
                    # 负样本-风格
                    self.prefixes.append(all_data[i]['prefix'])
                    # captions_raw.append("this image shows " + all_data[i]['caption'])
                    captions_raw.append(all_data[i]['caption'])
                    self.style.append(nagetive_map[all_data[i]['style']])
                    self.match_labels.append(0)
                    self.imgpath.append(filename)

        self.prefixes = torch.cat(self.prefixes, dim=0)
        self.captions = captions_raw

        if train_or_test == "train":
            if os.path.isfile(f"{data_path[:-4]}_tokens.pkl") and not args.overwrite_cache:
                with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                    self.captions_tokens, self.caption2embedding, self.max_seq_len, self.style_tokens = pickle.load(f)
            else:
                self.captions_tokens = []
                self.caption2embedding = []
                self.style_tokens = []
                max_seq_len = 0
                for i in range(len(captions_raw)):
                    self.captions_tokens.append(torch.tensor(self.tokenizer.encode(captions_raw[i]), dtype=torch.int64))
                    self.style_tokens.append(torch.tensor(self.tokenizer.encode(self.style[i])[0], dtype=torch.int64))
                    self.caption2embedding.append(i)
                    max_seq_len = max(max_seq_len, self.captions_tokens[-1].shape[0])
                with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                    pickle.dump([self.captions_tokens, self.caption2embedding, max_seq_len, self.style_tokens], f)
        self.max_seq_len = args.max_length


class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens, prefix, mask):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)

        # T/F + style_embedding + prefix_projections + embedding_text
        inputs_embeds1 = embedding_text[:, 0:1, :]
        inputs_embeds2 = embedding_text[:, 1:, :]
        embedding_cat = torch.cat((inputs_embeds1, prefix_projections, inputs_embeds2), 1)

        # 标签
        labels = tokens[:, 1:]
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask, prefix_length=self.prefix_length)
        return out

    # gedi直接生成
    def generate(self, prefix, args, tokenizer, style):
        # 图像embedding batchsize*4*1024
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        # positive/negative batchsize*1*1024
        seq_a = torch.tensor([tokenizer.encode(style_pn)[0] for style_pn in style]).reshape(-1, 1).to(device=args.device, dtype=torch.int64)
        embedding_text_style = self.gpt.transformer.wte(seq_a)
        # cat batchsize*5*1024
        embedding_cat = torch.cat((embedding_text_style, prefix_projections), dim=1)

        # else
        if args.class_bias is None:
            args.class_bias = 0.0

        generated_sequence = self.gpt.generate(input_ids=None,
                                                pad_lens=None,
                                                max_length=args.max_length-1,
                                                do_sample=args.do_sample,
                                                temperature=args.temperature,
                                                top_k=args.top_k,
                                                top_p=args.top_p,
                                                repetition_penalty=args.repetition_penalty,
                                                rep_penalty_scale=0,
                                                pad_token_id=tokenizer.eos_token_id,
                                                eos_token_ids=tokenizer.eos_token_id,
                                                penalize_cond=args.penalize_cond,
                                                gedi_model=None,
                                                tokenizer=tokenizer,
                                                disc_weight=0,
                                                filter_p=args.filter_p,
                                                target_p=args.target_p,
                                                class_bias=args.class_bias,
                                                attr_class=1,
                                                code_0="false",
                                                code_1="true",
                                                prefix_sequence=embedding_cat)
        return generated_sequence

    def __init__(self, tokenizer, gpt, prefix_length, prefix_size):
        super(ClipCaptionModel, self).__init__()
        self.tokenizer = tokenizer
        self.gpt = gpt

        self.prefix_length = prefix_length
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) //2,
                                     self.gpt_embedding_size * prefix_length))

class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes, bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


def noise_injection(x, variance=0.001, modality_offset=None, uniform_noise=False, dont_norm=False):
    if variance == 0.0:
        return x
    std = math.sqrt(variance)
    if not dont_norm:
        x = torch.nn.functional.normalize(x, dim=1)
    if uniform_noise:
        # x = x + get_uniform_ball_noise(x.shape, radius=std)
        print(1)
    else:
        x = x + (torch.randn(x.shape, device=x.device) * std)  # todo by some conventions multivraiance noise should be devided by sqrt of dim
    if modality_offset is not None:
        x = x + modality_offset
    return torch.nn.functional.normalize(x, dim=1)


def add_sep(batch, sep_id):

    batch[0]
    len_list = (batch[1].sum(dim=1) - batch[2].sum(dim=1)).tolist()

    left_chunk = [x[:len_] for x,len_ in zip(batch[0],len_list)]
    right_chunk= [x[len_:] for x,len_ in zip(batch[0],len_list)]



    mid_chunk = [torch.Tensor(sep_id).type_as(x) for x in batch[0]]


    tensor_list = [torch.cat((left,mid,right)) for (left,mid,right) in
                   zip(left_chunk, mid_chunk, right_chunk)]
    return torch.stack(tensor_list)[:,:-1]

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def eval_ppl(out_txt_dir, desiered_style, ppl_out_path):
    map_style_pplLM = {"romantic":"/home/liwc/wxp/refercode/GeDi_Final/PPL/LM_ro",
                       "humorous":"/home/liwc/wxp/refercode/GeDi_Final/PPL/LM_fu",
                       "positive":"/home/liwc/wxp/refercode/GeDi_Final/PPL/LM_pos",
                       "negative":"/home/liwc/wxp/refercode/GeDi_Final/PPL/LM_neg"}
    os.system('ngram -ppl ' + out_txt_dir + ' -order 3 -lm '+ map_style_pplLM[desiered_style] + ' > ' + ppl_out_path)
    with open(ppl_out_path, 'rb') as f:
        while True:
            line = f.readline()
            line = line.decode('utf-8')
            if not line:
                break
            last_line = line
    tokens = last_line.split()
    ppl = float(tokens[-3])
    # ppl1 = float(tokens[-1])
    return ppl

def eval_acc(out_txt_dir, teststyle, device, tokenizer, file_error_path):
    with open(out_txt_dir, "r") as f:
        captions = [line.strip() for line in f.readlines()]

    acc = {"match":0, "total":0, "acc":0.0}
    acc_map = {"factual":0, "positive":1, "negative":1, "romantic":1, "humorous":1}
    # model_path_map = {"positive":'cls_pos_2', "negative":'cls_neg_2',
                    #   "romantic":'cls_ro_2', "humorous":'cls_fu_2'}
    
    if teststyle == "positive" or teststyle == "negative":
        acc_model = textCNN(kernel_num=100, vocab_size=50257, kernel_size=[1, 2, 3], embed_dim=1024, dropout=0, class_num=2)
        model_path = './classfier/cls_pos_2.pt' if teststyle == "positive" else './classfier/cls_neg_2.pt'
        state_dict = torch.load(model_path, map_location="cpu")
        acc_model.load_state_dict(state_dict)
        acc_model = acc_model.to(device)
        acc_model.eval()
    elif teststyle == "romantic" or teststyle == "humorous" or teststyle == "factual":
        config_path = './classfier/cls_fu_2/config.json' if teststyle == "humorous" else './classfier/cls_ro_2/config.json'
        model_path = './classfier/cls_fu_2/pytorch_model.bin' if teststyle == "humorous" else './classfier/cls_ro_2/pytorch_model.bin'
    # if True: 
        # config_path = "./classfier/" + model_path_map[teststyle] + "/config.json"
        # model_path = "./classfier/" + model_path_map[teststyle] + "/pytorch_model.bin"
        bert_config = BertConfig(config_path)
        acc_model = BertLSTM(config=bert_config, num_labels=2, rnn_hidden_size=300, num_layers=2, bidirectional=True, dropout=0.2)
        acc_model.load_state_dict(torch.load(model_path, map_location="cpu"))
        acc_model = acc_model.to(device)
        acc_model.eval()
        acc_tokenizer = BertTokenizer.from_pretrained("/home/liwc/wxp/refercode/DataTestProcess/bert-base-uncased/vocab.txt", do_lower_case=True)

    file_error = open(file_error_path, "w")
    for generated_text in captions:
        if teststyle == "positive" or teststyle == "negative":
            predicted_label = classfier_sentiment(generated_text, acc_model, tokenizer, device)
        elif teststyle == "romantic" or teststyle == "humorous" or teststyle == "factual":
        # if True:
            predicted_label = sentence_style_cls(generated_text, acc_tokenizer, 21, acc_model, device)

        true_label = acc_map[teststyle]
        if predicted_label == true_label:
            acc["match"] = acc["match"] + 1
        else:
            # print(generated_text+"\t"+str(predicted_label.item())+"\t"+str(true_label))
            file_error.write(generated_text+"\t"+str(predicted_label.item())+"\t"+str(true_label) + "\n")
        acc["total"] = acc["total"] + 1
    file_error.close()

    acc["acc"] = acc["match"] / acc["total"]

    return acc["acc"]