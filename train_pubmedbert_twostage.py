import os
import pandas as pd
from datasets import Dataset, ClassLabel, load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import numpy as np

MODEL_NAME = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'

def clean_text(text):
    # Remove extra whitespace and normalize
    text = ' '.join(text.split())
    # Ensure reasonable length (truncate very long contexts)
    if len(text.split()) > 400:  # ~500 tokens after tokenization
        text = ' '.join(text.split()[:400])
    return text

def prepare_dataset(df, augment=True):
    """Prepare dataset with optional augmentation"""
    # Map labels to integers if needed
    label_mapping = {'yes': 1, 'no': 0, 'maybe': 2}
    if df['final_decision'].dtype == 'object':
        labels = df['final_decision'].map(label_mapping).astype(int)
    else:
        labels = df['final_decision'].astype(int)

    texts = []
    processed_labels = []
    
    for idx, row in df.iterrows():
        if idx % 10000 == 0:
            print(f"Processed {idx}/{len(df)} examples")
        
        question = clean_text(str(row['question']))
        context = clean_text(str(row['context']))
        label = labels.iloc[idx]
        
        # Original example
        combined_text = f"Question: {question} Context: {context}"
        texts.append(combined_text)
        processed_labels.append(label)
        
        # Simple augmentation for labeled data only
        if augment and len(df) < 5000:  # Only augment small datasets
            if question.startswith('Does '):
                aug_question = question.replace('Does ', 'Can ', 1)
                aug_text = f"Question: {aug_question} Context: {context}"
                texts.append(aug_text)
                processed_labels.append(label)
            elif question.startswith('Is '):
                aug_question = question.replace('Is ', 'Can ', 1)
                aug_text = f"Question: {aug_question} Context: {context}"
                texts.append(aug_text)
                processed_labels.append(label)

    return texts, processed_labels

def create_trainer(model, tokenizer, train_dataset, val_dataset, output_dir, learning_rate=2e-5, epochs=3):
    """Create trainer with appropriate settings"""
    
    # Tokenization function
    max_length = 512
    def preprocess(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=max_length)

    train_dataset = train_dataset.map(preprocess, batched=True)
    val_dataset = val_dataset.map(preprocess, batched=True)

    for ds in (train_dataset, val_dataset):
        ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy='steps',
        eval_steps=2000,
        save_strategy='steps',
        save_steps=2000,
        learning_rate=learning_rate,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        warmup_steps=500,
        logging_steps=200,
        gradient_accumulation_steps=2,
        fp16=True,
        dataloader_num_workers=0,  # Windows compatibility
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

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    return trainer

if __name__ == "__main__":
    print("=== TWO-STAGE TRAINING FOR PUBMEDQA ===")
    
    # STAGE 1: Pre-train on artificial dataset (211k examples)
    print("\n🚀 STAGE 1: Pre-training on artificial dataset...")
    print("Loading pqa_artificial (211k examples)...")
    
    artificial_dataset = load_dataset("pubmed_qa", "pqa_artificial")
    artificial_df = artificial_dataset['train'].to_pandas()
    print(f"Loaded artificial dataset: {len(artificial_df)} examples")
    
    # Prepare artificial dataset (no augmentation for large dataset)
    artificial_texts, artificial_labels = prepare_dataset(artificial_df, augment=False)
    
    # Create dataset
    artificial_raw_dataset = Dataset.from_dict({'text': artificial_texts, 'label': artificial_labels})
    artificial_raw_dataset = artificial_raw_dataset.cast_column('label', ClassLabel(num_classes=3, names=['0', '1', '2']))
    
    # Split artificial dataset (90/10 for efficiency)
    artificial_train, artificial_val = artificial_raw_dataset.train_test_split(test_size=0.1, stratify_by_column='label', seed=42).values()
    
    print(f"Stage 1 - Training: {len(artificial_train)}, Validation: {len(artificial_val)}")
    
    # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    
    # Train on artificial dataset
    stage1_trainer = create_trainer(
        model=model, 
        tokenizer=tokenizer,
        train_dataset=artificial_train,
        val_dataset=artificial_val,
        output_dir='pubmedbert-stage1-artificial',
        learning_rate=2e-5,
        epochs=2  # Fewer epochs for pre-training
    )
    
    print("Starting Stage 1 training...")
    stage1_trainer.train()
    
    # Save stage 1 model
    stage1_trainer.save_model('pubmedbert-stage1-artificial')
    tokenizer.save_pretrained('pubmedbert-stage1-artificial')
    print("✅ Stage 1 complete!")
    
    # STAGE 2: Fine-tune on labeled dataset (1k examples)
    print("\n🔥 STAGE 2: Fine-tuning on high-quality labeled dataset...")
    print("Loading pqa_labeled (1k examples)...")
    
    labeled_dataset = load_dataset("pubmed_qa", "pqa_labeled")
    labeled_df = labeled_dataset['train'].to_pandas()
    print(f"Loaded labeled dataset: {len(labeled_df)} examples")
    
    # Prepare labeled dataset (with augmentation)
    labeled_texts, labeled_labels = prepare_dataset(labeled_df, augment=True)
    
    # Create dataset
    labeled_raw_dataset = Dataset.from_dict({'text': labeled_texts, 'label': labeled_labels})
    labeled_raw_dataset = labeled_raw_dataset.cast_column('label', ClassLabel(num_classes=3, names=['0', '1', '2']))
    
    # Split labeled dataset (80/10/10)
    labeled_train_val, labeled_test = labeled_raw_dataset.train_test_split(test_size=0.1, stratify_by_column='label', seed=42).values()
    labeled_train, labeled_val = labeled_train_val.train_test_split(test_size=0.11, stratify_by_column='label', seed=42).values()
    
    print(f"Stage 2 - Training: {len(labeled_train)}, Validation: {len(labeled_val)}, Test: {len(labeled_test)}")
    
    # Load the pre-trained model from Stage 1
    print("Loading Stage 1 model for fine-tuning...")
    stage2_model = AutoModelForSequenceClassification.from_pretrained('pubmedbert-stage1-artificial', num_labels=3)
    
    # Fine-tune on labeled dataset
    stage2_trainer = create_trainer(
        model=stage2_model,
        tokenizer=tokenizer,
        train_dataset=labeled_train,
        val_dataset=labeled_val,
        output_dir='pubmedbert-stage2-labeled',
        learning_rate=1e-5,  # Lower learning rate for fine-tuning
        epochs=5  # More epochs for fine-tuning
    )
    
    print("Starting Stage 2 fine-tuning...")
    stage2_trainer.train()
    
    # Save final model
    stage2_trainer.save_model('pubmedbert-final-twostage')
    tokenizer.save_pretrained('pubmedbert-final-twostage')
    
    # Final evaluation
    print("\n📊 FINAL EVALUATION:")
    
    # Test on held-out labeled data
    max_length = 512
    def preprocess(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=max_length)
    
    labeled_test = labeled_test.map(preprocess, batched=True)
    labeled_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    final_metrics = stage2_trainer.evaluate(labeled_test)
    print("Final Test Results:")
    print(f"Accuracy: {final_metrics['eval_accuracy']:.3f}")
    print(f"Macro F1: {final_metrics['eval_macro_f1']:.3f}")
    
    print("\n🎉 Two-stage training complete!")
    print("Final model saved as: pubmedbert-final-twostage") 