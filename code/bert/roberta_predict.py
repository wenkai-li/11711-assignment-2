import torch
import evaluate
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import canva
import numpy as np
import torch.nn.functional as F
import os


metric = evaluate.load("f1")

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

def load_test_dataset(file_path):
    sentences = []
    ner_tags = []
    tokens = []
    tags = []
    with open(file_path, 'r') as f:
        next(f)
        t = 0
        for line in f:
            split_line = line.strip().split(',')   
            tokens.append(split_line[1])
            tags.append('O')
            if split_line[1] in {'-DOCSTART-','.', '!', '?'}:
                sentences.append(tokens)
                ner_tags.append(tags)
                tokens, tags = [], []  # Reset for next sentence
        if len(tokens)!=0 and len(tags)!=0:
            sentences.append(tokens)
            ner_tags.append(tags)
    return {"tokens": sentences, "ner_tags": ner_tags}

# load all the dataset

file_name = "test.conll"
dataset, all_tags = load_custom_dataset(file_name)
file_name = "test-2.csv"
dataset = load_test_dataset(file_name)
label_list = np.unique(all_tags).tolist()
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
id_to_label = {i: label  for label,i  in label_to_id.items()}
# 2. load model and tokenizer
model_directory_path = 'finetune_align/checkpoint'
tokenizer = AutoTokenizer.from_pretrained(model_directory_path, add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained(model_directory_path, num_labels=len(label_list))
data_collator = DataCollatorForTokenClassification(tokenizer)

def tokenize_function(examples):
    max_length = 512
    t = 0
    for label_sequence in examples['ner_tags']:
        for _ in label_sequence:
            t += 1
    print(t)
    labels = [[label_to_id[label] for label in label_sequence] for label_sequence in examples['ner_tags']]
    tokenized_inputs = tokenizer(examples['tokens'], is_split_into_words=True, padding='max_length', truncation=True, max_length=max_length, return_attention_mask=True)
    for label_sequence in labels:
        if len(label_sequence)>512:
            assert(1<0)
    # Pad or truncate the labels to the same length
    labels = [label_sequence + [-100]*(max_length-len(label_sequence)) for label_sequence in labels]
    labels = [label_sequence[:max_length] for label_sequence in labels]
    tokenized_inputs["labels"] = labels
    
    return tokenized_inputs

class CustomDataCollator(DataCollatorForTokenClassification):
    def __call__(self, features):
        label_name = "labels" if "labels" in features[0].keys() else "label_ids"
        labels = [f[label_name] for f in features]
        for f in features:
            f.pop(label_name)

        batch = self.tokenizer.pad(features, return_tensors=self.return_tensors)
        batch[label_name] = torch.tensor(labels, dtype=torch.int64)

        return batch

data_collator = CustomDataCollator(tokenizer)



tokenized_test_dataset = Dataset.from_dict({'tokens': dataset['tokens'], 'ner_tags': dataset['ner_tags']}).map(tokenize_function, batched=True)

# do not use
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    return {
        "f1": 1.0,
    }


trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 6. predict and evaluate
results = trainer.predict(tokenized_test_dataset)
predict = results.predictions
predict = np.array(predict)
predict = np.argmax(predict,axis=-1)
print(results.label_ids)
t = 0
output_path ='test_output_2.csv'
with open(output_path, 'w') as f:
    print("id,target",file=f)
    for p_sentence, l_sentence in zip(predict,results.label_ids):
        for p,l in zip(p_sentence, l_sentence):
            if l != -100:
                t += 1
                print(str(t)+','+id_to_label[p],file=f)