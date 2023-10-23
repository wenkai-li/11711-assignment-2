import torch
import evaluate
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import torch.nn.functional as F
import os


metric = evaluate.load("seqeval")

# class Training:
    # def __init__(self):

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
                tags.append(split_line[-1])
                all_tags.append(split_line[-1])
                if split_line[0] in {'.', '!', '?'}:
                    sentences.append(tokens)
                    ner_tags.append(tags)
                    tokens, tags = [], []  # Reset for next sentence
    return {"tokens": sentences, "ner_tags": ner_tags}, all_tags

# load all the dataset
file_path = "/home/zw3/11711-assignment-2/annotated"
dataset = []
tags = []
for i in os.listdir(file_path):
    data, all_tags = load_custom_dataset(os.path.join(file_path, i))
    dataset.extend(data)
    tags = tags + all_tags

# dataset = Dataset.from_dict(dataset)
label_list = np.unique(tags).tolist()
label_to_id = {label: i for i, label in enumerate(label_list)}
# 2. load model and tokenizer
model_directory_path = '/home/zw3/11711-assignment-2/checkpoints_linebyline_1000e'
tokenizer = AutoTokenizer.from_pretrained(model_directory_path, add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained(model_directory_path, num_labels=len(label_list))
data_collator = DataCollatorForTokenClassification(tokenizer)
# Split the dataset into train, validation, and test sets
def split_dataset(dataset):
    train_data, test_data = train_test_split(dataset['tokens'], test_size=0.2, random_state=42)
    train_tags, test_tags = train_test_split(dataset['ner_tags'], test_size=0.2, random_state=42)
    train_data, valid_data = train_test_split(train_data, test_size=0.1, random_state=42)
    train_tags, valid_tags = train_test_split(train_tags, test_size=0.1, random_state=42)
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
    per_device_train_batch_size=64,
    per_device_eval_batch_size=32,
    output_dir='./results_2',
    num_train_epochs=600,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    save_total_limit=2,
    logging_steps=100,
    do_train=True,
    do_eval=True,
    load_best_model_at_end=True,
    learning_rate=1e-5,
    label_names=["labels"],
    seed = 42
)


def compute_metrics(self, eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[self.label_classes[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [self.label_classes[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
# 6. training
trainer.train()

# 7. evaluate
trainer.evaluate(tokenized_test_dataset)

# 8. save model
model.save_pretrained('fine-tuned-roberta-1output')
tokenizer.save_pretrained('fine-tuned-roberta-1output')
