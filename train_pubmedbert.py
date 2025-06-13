import os
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import numpy as np

MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'

# Load data
csv_path = 'PUBMEDQA_dataset.csv'
df = pd.read_csv(csv_path)

# Use the numeric labels already provided
# Combine question and context into a single text field
texts = df['question'] + ' ' + df['context']
labels = df['labels'].astype(int)

# Build a Hugging Face Dataset
raw_dataset = Dataset.from_dict({'text': texts, 'label': labels})

# Split into train/valid/test (70/15/15)
train_valid, test_dataset = raw_dataset.train_test_split(test_size=0.15, stratify_by_column='label', seed=42).values()
train_dataset, valid_dataset = train_valid.train_test_split(test_size=0.1765, stratify_by_column='label', seed=42).values()  # 0.1765 * 0.85 ~ 0.15

# Tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# Tokenization function
max_length = 512
def preprocess(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=max_length)

train_dataset = train_dataset.map(preprocess, batched=True)
valid_dataset = valid_dataset.map(preprocess, batched=True)
test_dataset = test_dataset.map(preprocess, batched=True)

for ds in (train_dataset, valid_dataset, test_dataset):
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

args = TrainingArguments(
    output_dir='pubmedbert-finetuned',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='macro_f1'
)

from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'macro_f1': f1_score(labels, preds, average='macro')
    }

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model('pubmedbert-finetuned')
tokenizer.save_pretrained('pubmedbert-finetuned')

metrics = trainer.evaluate(test_dataset)
print(metrics)
