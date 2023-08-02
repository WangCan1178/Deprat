#!/usr/bin/env python
# coding: utf-8

# # Training Abstract2Title


import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import wandb
import random
import json
import tqdm
# from datasets import load_from_disk, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, \
                         DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer




model = AutoModelForSeq2SeqLM.from_pretrained('/wangcan/BART/model-t5-base/checkpoint-75000')
tokenizer = AutoTokenizer.from_pretrained('/wangcan/BART/model-t5-base/checkpoint-75000')

temperature = 0.9
num_beams = 4
max_gen_length = 512


with open('/wangcan/GPT/dataset/test.json') as f:
    datasets = f.readlines()
# datas = random.choices(datasets,k=10)
for data in tqdm.tqdm(datasets):
    data = json.loads(data)
    abstract, label = data['information'], data['label']
    inputs = tokenizer([abstract], max_length=1024, return_tensors='pt')

    title_ids = model.generate(
        inputs['input_ids'], 
        num_beams=num_beams, 
        temperature=temperature, 
        max_length=max_gen_length, 
        early_stopping=True
    )
    title = tokenizer.decode(title_ids[0].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
    with open('bart.txt','a') as f:
        f.write(title+'\n')
    #     for genl in gen_label:
    #         for l in genl:
    #             cnt+=1
    #             f.write(tokenizer.decode(l).replace(' [ space ] ',' ').split('[SEP]')[1].strip(' ')+'\n')
    # print('abstract: '+abstract+'\nlabel: '+label+'\ntitle: '+title)
    # print('\n')