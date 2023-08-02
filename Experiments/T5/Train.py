#!/usr/bin/env python
# coding: utf-8

# # Training Abstract2Title


import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import wandb
from datasets import load_from_disk, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, \
                         DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer



# replace with your weights and biases username otherwise comment this
wandb.init(project="abstract-to-title")


# ## Preprocess Data


# Initialize T5-base tokenizer
tokenizer = AutoTokenizer.from_pretrained('t5-base')



# Load the processed data
dataset = load_from_disk('api2param_dataset')




MAX_SOURCE_LEN = 1024
MAX_TARGET_LEN = 512




def preprocess_data(example):
    
    model_inputs = tokenizer(example['information'], max_length=MAX_SOURCE_LEN, padding=True, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(example['label'], max_length=MAX_TARGET_LEN, padding=True, truncation=True)

    # Replace all pad token ids in the labels by -100 to ignore padding in the loss
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]

    model_inputs['labels'] = labels["input_ids"]

    return model_inputs


# Apply preprocess_data() to the whole dataset
processed_dataset = dataset.map(
    preprocess_data,
    batched=True,
    remove_columns=['information', 'label'],
    desc="Running tokenizer on dataset",
)



batch_size = 2
num_epochs = 1
learning_rate = 5.6e-5
weight_decay = 0.01
log_every = 100
eval_every = 5000
lr_scheduler_type = "linear"




# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="model-t5-base",
    evaluation_strategy="steps",
    eval_steps=eval_every,
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=weight_decay,
    save_steps=500,
    save_total_limit=3,
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    logging_steps=log_every,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
    resume_from_checkpoint=True,
)


# ## Train

# Initialize T5-base model
model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')


# Define ROGUE metrics on evaluation data
metric = load_metric("/wangcan/T5/rouge.py")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
    
    # Compute ROUGE scores and get the median scores
    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    return {k: round(v, 4) for k, v in result.items()}


# Dynamic padding in batch using a data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define the trainer
trainer = Seq2SeqTrainer(
    model,
    training_args,
    train_dataset=processed_dataset["train"],
    eval_dataset=processed_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


trainer.train()


