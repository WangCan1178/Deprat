import re
import json
import torch
import random
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, emb_dim)  # 词嵌入
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = (src len, batch size)
        embedded = self.dropout(self.embedding(src))
        # embedded = (src len, batch size, emb dim)
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = (src len, batch size, hid dim * n directions)
        # hidden = (n layers * n directions, batch size, hid dim)
        # cell = (n layers * n directions, batch size, hid dim)
        # rnn的输出总是来自顶部的隐藏层
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout,bidirectional=True)
        self.fc_out = nn.Linear(hid_dim*2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # 各输入的形状
        # input = (batch size)
        # hidden = (n layers * n directions, batch size, hid dim)
        # cell = (n layers * n directions, batch size, hid dim)

        # LSTM是单向的  ==> n directions == 1
        # hidden = (n layers, batch size, hid dim)
        # cell = (n layers, batch size, hid dim)

        input = input.unsqueeze(0)  # (batch size)  --> [1, batch size)

        embedded = self.dropout(self.embedding(input))  # (1, batch size, emb dim)

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # LSTM理论上的输出形状
        # output = (seq len, batch size, hid dim * n directions)
        # hidden = (n layers * n directions, batch size, hid dim)
        # cell = (n layers * n directions, batch size, hid dim)

        # 解码器中的序列长度 seq len == 1
        # 解码器的LSTM是单向的 n directions == 1 则实际上
        # output = (1, batch size, hid dim)
        # hidden = (n layers, batch size, hid dim)
        # cell = (n layers, batch size, hid dim)

        prediction = self.fc_out(output.squeeze(0))

        # prediction = (batch size, output dim)

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, input_word_count, output_word_count, encode_dim, decode_dim, hidden_dim, n_layers,
                 encode_dropout, decode_dropout, device):
        """

        :param input_word_count:    英文词表的长度     34737
        :param output_word_count:   中文词表的长度     4015
        :param encode_dim:          编码器的词嵌入维度
        :param decode_dim:          解码器的词嵌入维度
        :param hidden_dim:          LSTM的隐藏层维度
        :param n_layers:            采用n层LSTM
        :param encode_dropout:      编码器的dropout概率
        :param decode_dropout:      编码器的dropout概率
        :param device:              cuda / cpu
        """
        super().__init__()
        self.encoder = Encoder(input_word_count, encode_dim, hidden_dim, n_layers, encode_dropout)
        self.decoder = Decoder(output_word_count, decode_dim, hidden_dim, n_layers, decode_dropout)
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src = (src len, batch size)
        # trg = (trg len, batch size)
        # teacher_forcing_ratio 定义使用Teacher Forcing的比例
        # 例如 if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim  # 实际上就是中文词表的长度
        # 初始化保存解码器输出的Tensor
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # 编码器的隐藏层输出将作为i解码器的第一个隐藏层输入
        hidden, cell = self.encoder(src)

        # 解码器的第一个输入应该是起始标识符<sos>
        input = trg[0, :]  # 取trg的第“0”行所有列  “0”指的是索引
        # 从the_collate_fn函数中可以看出trg的第“0”行全是0，也就是起始标识符对应的ID

        for t in range(1, trg_len): # 从 trg的第"1"行开始遍历
            # 解码器的输入包括：起始标识符的词嵌入input; 编码器输出的 hidden and cell states
            # 解码器的输出包括：输出张量(predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            # 保存每次预测结果于outputs
            # outputs (trg_len, batch_size, trg_vocab_size)
            # output  (batch size, trg_vocab_size)
            outputs[t] = output

            # 随机决定是否使用Teacher Forcing
            teacher_force = random.random() < teacher_forcing_ratio

            # output  (batch size, trg_vocab_size)  沿dim=1取最大值索引
            top1 = output.argmax(dim=1)  # (batch size, )

            # if teacher forcing, 以真实值作为下一个输入 否则 使用预测值
            input = trg[t] if teacher_force else top1

        return outputs

def read_json(file_name):
# 逐行读取json
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    inf, lab = [], []
    for line in lines:
        tmp = []
        tmplst = [""]+re.split('([ |:|#])', json.loads(line)['information'])
        for w in ["".join(i) for i in zip(tmplst[0::2],tmplst[1::2])]:
            w = w.replace('.', '').replace(',', '').replace(':', '').replace('。', '').replace('，', '').replace('《', '').replace('》', '').replace('：', '')  # 去掉跟单词连着的标点符号
            w = w.lower()  # 统一单词大小写
            if w:
                tmp.append(w)
        inf.append(tmp.copy())
        tmp = []
        tmplst = [""]+re.split('([ |:|#])', json.loads(line)['label'])
        for w in ["".join(i) for i in zip(tmplst[0::2],tmplst[1::2])]:
            w = w.replace('.', '').replace(',', '').replace(':', '').replace('。', '').replace('，', '').replace('《', '').replace('》', '').replace('：', '')  # 去掉跟单词连着的标点符号
            w = w.lower()  # 统一单词大小写
            if w:
                tmp.append(w)
        lab.append(tmp.copy())
    return inf, lab

tsrc, ttrg = read_json('/wangcan/LSTM/api2_dataset/dataset_train.json')
vsrc, vtrg = read_json('/wangcan/LSTM/api2_dataset/dataset_valid.json')
tescr, tetrg = read_json('/wangcan/LSTM/api2_dataset/dataset_test.json')
print(len(tsrc),len(ttrg))

raw_words = set()  # 初始化集合对象  自动去重
for sen in tsrc + ttrg + vsrc +vtrg:
    for w in sen:  # 英文内容按照空格字符进行分词
        raw_words.add(w)
# print('postwageprojectionbasedondefaultclaconfigurationbyemploymentid' in list(raw_words))
print(len(raw_words))

words = ['<sos>', '<eos>', '<pad>'] + list(raw_words)

pad_id =2
word2id = {}
for i, w in enumerate(words):  # 遍历枚举类型对象实现此功能
    word2id[w] = i

# print(len(word2id),word2id)

import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

source_word_count = len(words)  # 词表的长度
target_word_count = len(words)  # 词表的长度

encode_dim = 512   # 编码器的词嵌入维度
decode_dim = 512    # 解码器的词嵌入维度
hidden_dim = 512    # LSTM的隐藏层维度
n_layers = 2        # 采用n层LSTM
encode_dropout = 0.3    # 编码器的dropout概率
decode_dropout = 0.3    # 编码器的dropout概率
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # GPU可用 用GPU
device =  torch.device('cpu')  # GPU可用 用GPU



import torch

batch_size = 1
data_workers = 2  # 子进程数 

class MyDataSet(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        s1 = example[0][:1024]
        s2 = example[1][:512]
        l1 = len(s1)
        l2 = len(s2)
        return s1, l1, s2, l2, index  # 英文句子  英文句子长度  中文句子  中文句子长度 当前数据在数据集中的索引

def the_collate_fn(batch):
    src = [[0] * len(batch)]  # src ---> source 缩写   该任务中 源句子指的是英文句子  # 每个样本的开头都是0(起始标识符的编码)
    tar = [[0] * len(batch)]  # tar ---> target 缩写           目标句子指的是中文句子
    src_max_l = 0  # 初始化英文句子最大长度  方便计算需要填充的个数
    for b in batch: # 每个batch的数据有五个信息 分别是: 英文句子  英文句子长度  中文句子  中文句子长度 当前数据在数据集中的索引
        src_max_l = max(src_max_l, b[1])  # b[1] 即英文句子的长度
    tar_max_l = 0
    for b in batch:
        tar_max_l = max(tar_max_l, b[3])  # b[3] 即中文句子的长度
    for i in range(src_max_l):
        l = []
        for x in batch:
            if i < x[1]:
                l.append(pad_id if x[0][i] not in word2id else word2id[x[0][i]])
            else:
                l.append(pad_id)  # 如果句子长度小于最大句子长度，进行填充
        src.append(l)
        # l记录的是每个句子的第 i 个词  有多少个句子？ batch size个，因此len(l) == batch_size == 句子的数量
        # src记录的是每个 l  总共多少个l？ src_max_l个，因此len(src) == src_max_l == 句子的最大长度
        # len(src) == 句子的最大长度    len(src[0]) == 句子的数量
        # [len(src), len(src[0])] ==> [src len, batch size]

    for i in range(tar_max_l):  # 注释参考上面
        l = []
        for x in batch:
            if i < x[3]:
                l.append(pad_id if x[2][i] not in word2id else word2id[x[2][i]])
            else:
                l.append(pad_id)  # 如果句子长度小于最大句子长度，进行填充
        tar.append(l)
    indexs = [b[4] for b in batch]  # b[4] 记录的是 当前数据在数据集中的索引
    src.append([1] * len(batch))  # 终止标识符的编码为1 所以src和tar在句子的最后把终止符加上
    tar.append([1] * len(batch))
    # print(src)
    s1 = torch.LongTensor(src) 
    s2 = torch.LongTensor(tar)
    return s1, s2, indexs

# valid_set = []
# for idx in range(len(vsrc)):
#     valid_set.append([vsrc[idx], vtrg[idx]])

# dev_dataset = MyDataSet(valid_set)

test_set = []
for idx in range(len(tescr)):
    test_set.append([tescr[idx], tetrg[idx]])
test_dataset = MyDataSet(test_set)

# dev_data_loader = torch.utils.data.DataLoader(
#     dev_dataset,
#     batch_size=batch_size,
#     shuffle= False,
#     num_workers=data_workers,
#     collate_fn=the_collate_fn,
# )
test_data_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle= False,
    num_workers=data_workers,
    collate_fn=the_collate_fn,
)
print("Create model...")
# Seq2Seq模型实例化
model = Seq2Seq(source_word_count, target_word_count, encode_dim, decode_dim, hidden_dim, n_layers, encode_dropout,
                decode_dropout, device).to(device)

model.load_state_dict(torch.load("best_model_copy.pth"))


import torch.optim as optim
from tqdm import tqdm

titles = []
# 验证策略
def evaluate(model, iterator, criterion):
    model.eval() # 切换到验证模式
    epoch_loss = 0
    with torch.no_grad():  # 不计算梯度
        # for idx, batch in enumerate(iterator):
        for idx, batch in tqdm(enumerate(iterator)):
            src = batch[0].to(device)
            trg = batch[1].to(device)
            output = model(src, trg, teacher_forcing_ratio=0)  # 验证时禁用Teacher Forcing
            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]
            output_dim = output.shape[-1]
            # print("output size", output.shape)
            output = output[1:].view(-1, output_dim)
            tmp = []
            for i in torch.argmax(output,1):
                if i!=1:
                   tmp.append(words[i])
                else:
                    break
            titles.append(" ".join(tmp))
            # print(len(titles))
            trg = trg[1:].view(-1)
            # print([words[i] for i in trg])
            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            if idx>=30:
                break
    return epoch_loss / len(iterator)

# 定义损失函数
criterion = nn.CrossEntropyLoss(ignore_index=pad_id).to(device)  # 忽略填充标识符的索引

import math

valid_loss = evaluate(model, test_data_loader, criterion)
print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


with open('lstmnew.txt','a') as f:
    for title in titles:
        f.write(title+'\n')