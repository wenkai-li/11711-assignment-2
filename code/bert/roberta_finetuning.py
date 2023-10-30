import torch
import evaluate
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import load_dataset, Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import torch.nn.functional as F
import os
from accelerate import Accelerator


metric = evaluate.load("f1")
accelerator = Accelerator()

# 1. load dataset
def load_custom_dataset(file_path):
    sentences = []
    ner_tags = []
    tokens = []
    tags = []
    all_tags = []
    with open(file_path, 'r') as f:
        for line in f:
            split_line = line.strip().split()
            if len(split_line) > 1:    
                tokens.append(split_line[0])
                tags.append(split_line[-1].replace("B-", "").replace("I-", ""))
                all_tags.append(split_line[-1].replace("B-", "").replace("I-", ""))
                if split_line[0] in {'.', '!', '?'}:
                    sentences.append(tokens)
                    ner_tags.append(tags)
                    tokens, tags = [], []  # Reset for next sentence
        if len(tokens)!=0 and len(tags)!=0:
            sentences.append(tokens)
            ner_tags.append(tags)
    return {"tokens": sentences, "ner_tags": ner_tags}, all_tags

# load all the dataset, need to specify
file_path = ""
dataset = {
    'tokens': [],
    'ner_tags': []
}
tags = []
for i in os.listdir(file_path):
    data, all_tags = load_custom_dataset(os.path.join(file_path, i))
    dataset = {key: dataset[key] + data[key] for key in dataset}
    tags = tags + all_tags

label_list = sorted(np.unique(tags).tolist())
label_to_id = {
    'O': 0,
    'MethodName': 1,
    'DatasetName': 2,
    'HyperparameterName': 3,
    'TaskName': 4,
    'MetricValue': 5,
    'MetricName': 6,
    'HyperparameterValue': 7
}
# roberta file_path
model_directory_path = ""
tokenizer = AutoTokenizer.from_pretrained(model_directory_path, add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained(model_directory_path, num_labels=len(label_list))
data_collator = DataCollatorForTokenClassification(tokenizer)
# Split the dataset into train, validation, and test sets
def split_dataset(dataset):
    train_data, test_data = train_test_split(dataset['tokens'], test_size=0.1, random_state=42)
    train_tags, test_tags = train_test_split(dataset['ner_tags'], test_size=0.1, random_state=42)
    train_data, valid_data = train_test_split(train_data, test_size=0.15, random_state=42)
    train_tags, valid_tags = train_test_split(train_tags, test_size=0.15, random_state=42)
    return train_data, valid_data, test_data, train_tags, valid_tags, test_tags

def tokenize_function(examples):
    labels = [[label_to_id[label] for label in label_sequence] for label_sequence in examples['ner_tags']]
    input_ids = tokenizer(examples['tokens'], is_split_into_words=True, padding='max_length', truncation=True)['input_ids']
    return {
        "input_ids": input_ids,
        "labels": labels
    }

# 3. data preprocessing
# Create datasets for each split
train_data, valid_data, test_data, train_tags, valid_tags, test_tags = split_dataset(dataset)
tokenized_train_dataset = Dataset.from_dict({'tokens': train_data, 'ner_tags': train_tags}).map(tokenize_function, batched=True)
tokenized_valid_dataset = Dataset.from_dict({'tokens': valid_data, 'ner_tags': valid_tags}).map(tokenize_function, batched=True)
tokenized_test_dataset = Dataset.from_dict({'tokens': test_data, 'ner_tags': test_tags}).map(tokenize_function, batched=True)

# 4. prepare training arguments and trainer
training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=32,
    output_dir='./finetune_align',
    num_train_epochs=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    save_total_limit=2,
    logging_steps=100,
    do_train=True,
    do_eval=True,
    load_best_model_at_end=True,
    learning_rate=1e-5,
    seed = 42
)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [label_to_id[i] for item in true_labels for i in item]
    true_predictions = [label_to_id[i] for item in true_predictions  for i in item]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels, average = "macro")
    return {
        "f1": all_metrics,
    }


trainer = accelerator.prepare(Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
)
# 6. training
trainer.train()

# 7. evaluate
trainer.evaluate(tokenized_test_dataset)

# 8. save model
model.save_pretrained('finetune_align')
tokenizer.save_pretrained('finetune_align')