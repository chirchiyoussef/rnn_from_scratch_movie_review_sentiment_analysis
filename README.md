# Movie Review Sentiment — RNN From Scratch to Improved RNN

This repo builds a movie-review sentiment classifier in PyTorch.  
It starts with a simple BiGRU baseline (trained from scratch) and then improves it with stronger regularization and better pooling. A Streamlit app is included for quick testing.

## What it does
Type a movie review → the model predicts:
- ✅ Liked it (positive)
- ❌ Didn’t like it (negative)

## Dataset
This project uses the Kaggle dataset:
https://www.kaggle.com/datasets/ashukr/rnnsentiment-data/code

Download it and place the files inside the `data/` folder.

## Key ideas
- Word-level tokenization (regex)
- Vocabulary built from TRAIN only (avoids leakage)
- Sequence model (BiGRU) to use word order/context
- Improved model uses EmbeddingDropout + Mean+Max pooling

## Project structure
.
├── data/
├── notebooks/
├── app.py
└── model_weights/
├── sentiment_bigru_meanmax_state_dict.pt
└── sentiment_bigru_meanmax_bundle.pt

## Installation
Install dependencies from `requirements.txt`:

pip install -r requirements.txt
Models
Baseline model (starter)

Architecture:

Embedding (trainable lookup table)

BiGRU (bidirectional GRU)

Masked mean pooling (ignores <pad>)

Linear layer → logit

Improved model (final)

Improvements:

EmbeddingDropout (regularizes embeddings)

Larger dimensions: emb_dim=200, hid_dim=256

Mean + Max pooling (overall signal + strongest cue)

More dropout to reduce overfitting

Architecture:

Embedding → EmbeddingDropout → BiGRU → masked mean pooling + masked max pooling → concat → Linear → logit

Saved model files

Two formats are saved:

1) Weights only

model_weights/sentiment_bigru_meanmax_state_dict.pt

2) Full bundle (recommended for inference)

model_weights/sentiment_bigru_meanmax_bundle.pt
Contains:

state_dict

vocab

max_len

model hyperparameters (emb_dim, hid_dim, dropout, etc.)

Streamlit app

Run the demo app:
streamlit run app.py
p >= 0.5 → ✅ Liked it

p < 0.5 → ❌ Didn’t like it

************************************************************
Results (Accuracy)

Transformers accuracy: 0.9088

Improved RNN (second RNN) accuracy: 0.8976

Baseline RNN (first RNN) accuracy: 0.8676
