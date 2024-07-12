import argparse
import os
import numpy as np
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME, BertModel, BertPreTrainedModel
from pytorch_pretrained_bert.tokenization import BertTokenizer

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F



def get_device(gpu_id):
    device = torch.device("cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if torch.cuda.is_available():
        print("device is cuda, # cuda is: ", n_gpu)
    else:
        print("device is cpu, not recommend")
    return device, n_gpu

def get_args():
    parser = argparse.ArgumentParser(description='BERT Baseline')
    parser.add_argument("--model_name", default="BertLSTM", type=str, help="the name of model")
    parser.add_argument("--output_dir",  default=".flickrstyle_3label_output/BertLSTM/", type=str)
    parser.add_argument("--bert_vocab_file", default="/home/liwc/wxp/refercode/DataTestProcess/bert-base-uncased/vocab.txt", type=str)
    parser.add_argument("--bert_model_dir", default="/home/liwc/wxp/refercode/DataTestProcess/bert-base-uncased", type=str)
    parser.add_argument("--do_lower_case", default=True, type=bool, help="Set this flag if you are using an uncased model.")
    parser.add_argument("--max_seq_length", default=21, type=int)
    parser.add_argument("--hidden_size", default=300, type=int, help="隐层特征维度")
    parser.add_argument('--num_layers', default=2, type=int, help='RNN层数')
    parser.add_argument("--bidirectional", default=True, type=bool)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--gpu_ids", type=str, default="0", help="gpu 的设备id")
    parser.add_argument("--save_name", default="BertLSTM", type=str, help="the name file of model")
    parser.add_argument("--label_list", default=['0', '1'])

    config = parser.parse_args()
    return config

class BertLSTM(BertPreTrainedModel):

    def __init__(self, config, num_labels, rnn_hidden_size, num_layers, bidirectional, dropout):
        super(BertLSTM, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.rnn = nn.LSTM(config.hidden_size, rnn_hidden_size, num_layers,bidirectional=bidirectional, batch_first=True, dropout=dropout)
        self.classifier = nn.Linear(rnn_hidden_size * 2, num_labels)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoded_layers, _ = self.bert(
            input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        encoded_layers = self.dropout(encoded_layers)
        # encoded_layers: [batch_size, seq_len, bert_dim]

        _, (hidden, cell) = self.rnn(encoded_layers)
        # outputs: [batch_size, seq_len, rnn_hidden_size * 2]
        hidden = self.dropout(
            torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))  # 连接最后一层的双向输出

        logits = self.classifier(hidden)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

def sentence_style_cls(sentence, tokenizer, max_seq_length, model, device):
    token = tokenizer.tokenize(sentence)
    if len(token) > max_seq_length - 2:
        token = token[:(max_seq_length - 2)]
    token = ["[CLS]"] + token + ["[SEP]"]
    segment_ids = [0] * len(token)

    input_ids = tokenizer.convert_tokens_to_ids(token)
    input_mask = [1] * len(input_ids)
    padding = [0] * (max_seq_length - len(input_ids))

    input_ids += padding
    input_mask += padding
    segment_ids += padding


    input_ids = torch.tensor(input_ids).unsqueeze(0)
    input_mask = torch.tensor(input_mask).unsqueeze(0)
    segment_ids = torch.tensor(segment_ids).unsqueeze(0)
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)

    with torch.no_grad():
        logits = model(input_ids, segment_ids, input_mask, labels=None)

    preds = logits.detach().cpu().numpy()
    outputs = np.argmax(preds, axis=1)
    return outputs[0]


if __name__  == "__main__":

    config = get_args()
    output_config_file = os.path.join(config.output_dir, config.save_name, CONFIG_NAME)
    bert_config = BertConfig(output_config_file)

    # label_list = ['0', '1']
    num_labels = len(config.label_list)
    model = BertLSTM(bert_config, num_labels, config.hidden_size, config.num_layers, config.bidirectional, config.dropout)
    output_model_file = os.path.join(config.output_dir, config.save_name, WEIGHTS_NAME) 
    model.load_state_dict(torch.load(output_model_file))

    gpu_ids = [int(device_id) for device_id in config.gpu_ids.split()]
    device, n_gpu = get_device(gpu_ids[0])  
    if n_gpu > 1:
        n_gpu = len(gpu_ids)
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(
        config.bert_vocab_file, do_lower_case=config.do_lower_case)
    
    label_map = {label: i for i, label in enumerate(config.label_list)}

    while(1):
        sentence = input()
        outputs = sentence_style_cls(sentence, tokenizer, config.max_seq_length, model, device)
        print(outputs)