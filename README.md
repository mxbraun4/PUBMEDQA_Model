# PubMedQA PubMedBERT Fine-tuning

This repository contains the `PUBMEDQA_dataset.csv` file with ~1000 question–abstract pairs from the PubMedQA dataset and a script to fine‑tune PubMedBERT for classifying answers as **yes**, **no**, or **maybe**.

## Requirements
Install dependencies (for example, using pip):

```bash
pip install transformers datasets torch scikit-learn pandas
```

## Training
Run the training script from the repository root:

```bash
python train_pubmedbert.py
```

The script splits the data into train/validation/test sets, fine‑tunes the model for a few epochs, and saves the results under `pubmedbert-finetuned/`.

After training finishes, the `pubmedbert-finetuned` directory will contain the trained model and tokenizer.
