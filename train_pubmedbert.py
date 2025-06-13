import os
import pandas as pd
from datasets import Dataset, ClassLabel, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import numpy as np

MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'

# Load the full PubMedQA dataset (211k examples)
print("Loading full PubMedQA dataset...")
print("Loading pqa_artificial (211k examples)...")
dataset = load_dataset("pubmed_qa", "pqa_artificial")
df = dataset['train'].to_pandas()
print(f"Loaded full dataset with {len(df)} examples")

print(f"Dataset size: {len(df)} examples")

# Better text preprocessing
def clean_text(text):
    # Remove extra whitespace and normalize
    text = ' '.join(text.split())
    # Ensure reasonable length (truncate very long contexts)
    if len(text.split()) > 400:  # ~500 tokens after tokenization
        text = ' '.join(text.split()[:400])
    return text

# Map labels to integers if needed
label_mapping = {'yes': 1, 'no': 0, 'maybe': 2}
if df['final_decision'].dtype == 'object':
    labels = df['final_decision'].map(label_mapping).astype(int)
else:
    labels = df['final_decision'].astype(int)

# Simple data augmentation techniques
def augment_question(question):
    """Simple question paraphrasing using common medical question patterns"""
    augmentations = []
    
    # Original question
    augmentations.append(question)
    
    # Add question variants
    if question.endswith('?'):
        # "Does X cause Y?" -> "Can X cause Y?"
        if question.startswith('Does '):
            augmentations.append(question.replace('Does ', 'Can ', 1))
        elif question.startswith('Is '):
            augmentations.append(question.replace('Is ', 'Can ', 1))
        elif question.startswith('Can '):
            augmentations.append(question.replace('Can ', 'Does ', 1))
    
    return augmentations

# Combine question and context with augmentation
texts = []
augmented_labels = []

print("Applying data augmentation...")
for idx, row in df.iterrows():
    if idx % 10000 == 0:
        print(f"Processed {idx}/{len(df)} examples")
    
    question = clean_text(str(row['question']))
    context = clean_text(str(row['context']))
    label = labels.iloc[idx]
    
    # Get question variants
    question_variants = augment_question(question)
    
    # Create training examples for each variant
    for variant in question_variants:
        combined_text = f"Question: {variant} Context: {context}"
        texts.append(combined_text)
        augmented_labels.append(label)

print(f"Augmented dataset size: {len(texts)} examples (from {len(df)})")

# Build a Hugging Face Dataset with proper ClassLabel feature
raw_dataset = Dataset.from_dict({'text': texts, 'label': augmented_labels})
# Convert label column to ClassLabel type for stratification
raw_dataset = raw_dataset.cast_column('label', ClassLabel(num_classes=3, names=['0', '1', '2']))

# For large dataset, use smaller test/validation sets to speed up training
# Split into train/valid/test (90/5/5) for efficiency with large dataset
train_valid, test_dataset = raw_dataset.train_test_split(test_size=0.05, stratify_by_column='label', seed=42).values()
train_dataset, valid_dataset = train_valid.train_test_split(test_size=0.053, stratify_by_column='label', seed=42).values()  # 0.053 * 0.95 ~ 0.05

print(f"Training set: {len(train_dataset)} examples")
print(f"Validation set: {len(valid_dataset)} examples") 
print(f"Test set: {len(test_dataset)} examples")

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
    output_dir='pubmedbert-finetuned-full',
    eval_strategy='steps',
    eval_steps=5000,  # Evaluate every 5000 steps for large dataset
    save_strategy='steps',
    save_steps=5000,
    learning_rate=2e-5,  # Standard learning rate for large dataset
    per_device_train_batch_size=8,  # Larger batch size for efficiency
    per_device_eval_batch_size=16,
    num_train_epochs=3,  # Fewer epochs needed with large dataset
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='macro_f1',
    warmup_steps=1000,  # More warmup for large dataset
    logging_steps=500,  # Log every 500 steps
    gradient_accumulation_steps=2,  # Effective batch size of 16
    fp16=True,  # Enable mixed precision for faster training
    dataloader_num_workers=0,  # Disable multiprocessing for Windows
    remove_unused_columns=False,
)

from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'macro_f1': f1_score(labels, preds, average='macro')
    }

if __name__ == "__main__":
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()

    trainer.save_model('pubmedbert-finetuned-full')
    tokenizer.save_pretrained('pubmedbert-finetuned-full')

    metrics = trainer.evaluate(test_dataset)
    print(metrics)
