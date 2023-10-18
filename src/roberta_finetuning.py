import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from datasets import load_dataset
import numpy as np
import torch.nn.functional as F

# 1. load dataset
dataset = load_dataset("conll2003")
label_list = [i for i in range (9)]
# 2. load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("roberta-base", add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained("roberta-base", num_labels=9)
data_collator = DataCollatorForTokenClassification(tokenizer)

# 3. data preprocessing
def tokenize_function(examples):
      return tokenizer(examples['tokens'], is_split_into_words=True, padding=True, truncation=True, max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 4. prepare dataset for pytorch
class NerDataset(Dataset):
    
    def __init__(self, tokenized_dataset):
        self.tokenized_dataset = tokenized_dataset

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.tokenized_dataset['input_ids'][idx])
        attention_mask = torch.tensor(self.tokenized_dataset['attention_mask'][idx])
        labels = torch.tensor(self.tokenized_dataset['ner_tags'][idx], dtype=torch.long)
        # max_length = 128  # or set it to another value

        # # Apply padding
        # input_ids = F.pad(input_ids, pad=(0, max_length - len(input_ids)))
        # attention_mask = F.pad(attention_mask, pad=(0, max_length - len(attention_mask)))
        # labels = F.pad(labels, pad=(0, max_length - len(labels)))

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    def __len__(self):
        return len(self.tokenized_dataset['input_ids'])

train_dataset = NerDataset(tokenized_dataset["train"])
valid_dataset = NerDataset(tokenized_dataset["validation"])
test_dataset = NerDataset(tokenized_dataset["test"])

# 5. prepare training arguments and trainer
training_args = TrainingArguments(
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    output_dir='./results',
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=100,
    do_train=True,
    do_eval=True,
    no_cuda=False,
    load_best_model_at_end=True,
)

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = precision_recall_fscore_support(true_labels, true_predictions)
    return {
        "precision": results[0],
        "recall": results[1],
        "f1": results[2],
        "num_samples": len(true_labels),
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 6. training
trainer.train()

# 7. evaluate
trainer.evaluate()

# 8. save model
model.save_pretrained('fine-tuned-roberta-conll2003')
tokenizer.save_pretrained('fine-tuned-roberta-conll2003')
