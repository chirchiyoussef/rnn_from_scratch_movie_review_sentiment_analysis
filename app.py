# app.py
import re
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn

# ---- SAME MODEL CLASSES (needed to load weights) ----
class EmbeddingDropout(nn.Module):
    def __init__(self, p: float):
        super().__init__()
        self.p = p
    def forward(self, x):
        if (not self.training) or self.p == 0.0:
            return x
        mask = x.new_empty((x.size(0), 1, x.size(2))).bernoulli_(1 - self.p)
        return x * mask / (1 - self.p)

class SentimentBiGRU_MeanMax(nn.Module):
    def __init__(self, vocab_size, emb_dim=200, hid_dim=256, n_layers=1,
                 dropout=0.45, emb_dropout=0.15, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.emb_drop = EmbeddingDropout(emb_dropout)
        self.gru = nn.GRU(
            input_size=emb_dim, hidden_size=hid_dim, num_layers=n_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim * 4, 1)
        self.pad_idx = pad_idx

    def forward(self, x):
        emb = self.embedding(x)
        emb = self.emb_drop(emb)
        emb = self.dropout(emb)
        out, _ = self.gru(emb)

        mask = (x != self.pad_idx).unsqueeze(-1)
        out_mean = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        out_max = out.masked_fill(~mask, -1e9).max(dim=1).values
        pooled = torch.cat([out_mean, out_max], dim=1)

        return self.fc(self.dropout(pooled)).squeeze(1)

# ---- TOKENIZER (same as training) ----
TOKEN_RE = re.compile(r"[a-z0-9']+")
def tokenize(t): return TOKEN_RE.findall(t.lower())

def encode(text, vocab, pad_id, unk_id, max_len):
    ids = [vocab.get(tok, unk_id) for tok in tokenize(text)]
    ids = ids[:max_len] + [pad_id] * max(0, max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long)  # (1, T)

# ---- LOAD SAVED BUNDLE (MODEL + WEIGHTS + VOCAB) ----
@st.cache_resource
def load_bundle(bundle_path: str):
    bundle = torch.load(bundle_path, map_location="cpu")
    vocab = bundle["vocab"]
    max_len = int(bundle["max_len"])
    PAD = bundle["pad_token"]
    UNK = bundle["unk_token"]
    hp = bundle["model_hparams"]

    model = SentimentBiGRU_MeanMax(**hp)
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    return model, vocab, vocab[PAD], vocab[UNK], max_len

# ---- STREAMLIT UI ----
st.title("🎬 Movie Review Sentiment (Improved BiGRU)")

bundle_path = Path("model_weights") / "sentiment_bigru_meanmax_bundle.pt"
model, vocab, pad_id, unk_id, max_len = load_bundle(str(bundle_path))

text = st.text_area("Write a movie review:")

if st.button("Predict") and text.strip():
    x = encode(text, vocab, pad_id, unk_id, max_len)
    with torch.no_grad():
        logit = model(x).item()
        prob = torch.sigmoid(torch.tensor(logit)).item()

    st.write("✅ Liked it" if prob >= 0.5 else "❌ Didn’t like it")
    st.write(f"prob={prob:.3f}")
