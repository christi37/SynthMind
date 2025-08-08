# agi_inference.py
# Inference & checkpoint utilities for the custom seq2seq denoiser.
# - Loads encoder+decoder weights saved by the training script (checkpoint contains encoder_state, decoder_state, "vocab")
# - If checkpoint absent, builds model with same architecture and default vocab
# - Provides denoise(text) API and a small interactive REPL
# - Supports saving current model state to disk via .save_checkpoint(path)
#
# Usage:
#   python agi_inference.py         # runs REPL, auto-loads checkpoint if present
#   In REPL type text -> prints denoised output
#   Type ":save" to save current (possibly untrained) model to denoiser_best.pt
#   Type ":exit" to quit

import os
import json
import re
import time
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

# ---------- Config ----------
MODEL_PATH = "denoiser_best.pt"
VOCAB_PATH = "vocab.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Architecture hyperparams (must match training)
EMBED_DIM = 128
HIDDEN_SIZE = 256
ENC_LAYERS = 2
DEC_LAYERS = 2
DROPOUT = 0.2
MAX_SEQ_LEN = 96

# Default vocabulary (same as training script)
DEFAULT_CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789 .,?!':;-()")
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
UNK_TOKEN = "<unk>"
ALL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN] + DEFAULT_CHARS

# ---------- Vocab helpers ----------
def load_vocab(path=VOCAB_PATH):
    if os.path.isfile(path):
        with open(path, "r") as f:
            toks = json.load(f)
    else:
        toks = ALL_TOKENS
    id2tok = {i: t for i, t in enumerate(toks)}
    tok2id = {t: i for i, t in id2tok.items()}
    return id2tok, tok2id

def save_vocab(toks, path=VOCAB_PATH):
    with open(path, "w") as f:
        json.dump(toks, f)

# ---------- Encoder/Decoder (same as training) ----------
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, n_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim, hidden_size, num_layers=n_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (h_n, c_n) = self.rnn(embedded)
        num_layers = h_n.size(0) // 2
        h_n = h_n.view(num_layers, 2, h_n.size(1), h_n.size(2))
        c_n = c_n.view(num_layers, 2, c_n.size(1), c_n.size(2))
        h_cat = torch.cat([h_n[-1, 0], h_n[-1, 1]], dim=1)
        c_cat = torch.cat([c_n[-1, 0], c_n[-1, 1]], dim=1)
        h0 = torch.tanh(self.fc(h_cat)).unsqueeze(0)
        c0 = torch.tanh(self.fc(c_cat)).unsqueeze(0)
        return outputs, (h0, c0)

class BahdanauAttention(nn.Module):
    def __init__(self, enc_hidden, dec_hidden):
        super().__init__()
        self.W1 = nn.Linear(enc_hidden, dec_hidden, bias=False)
        self.W2 = nn.Linear(dec_hidden, dec_hidden, bias=False)
        self.V = nn.Linear(dec_hidden, 1, bias=False)

    def forward(self, enc_outputs, dec_hidden, mask=None):
        dec_h = dec_hidden[-1].unsqueeze(1)
        score = self.V(torch.tanh(self.W1(enc_outputs) + self.W2(dec_h))).squeeze(-1)
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(score, dim=1)
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)
        return context, weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, enc_hidden_size, n_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.LSTM(embed_dim + enc_hidden_size, hidden_size, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.attn = BahdanauAttention(enc_hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size + enc_hidden_size + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_step, last_hidden, enc_outputs, mask=None):
        embedded = self.dropout(self.embedding(input_step))
        context, attn_weights = self.attn(enc_outputs, last_hidden, mask=mask)
        context = context.unsqueeze(1)
        rnn_input = torch.cat([embedded, context], dim=2)
        output, hidden = self.rnn(rnn_input, last_hidden)
        output = output.squeeze(1)
        embedded_s = embedded.squeeze(1)
        context_s = context.squeeze(1)
        combined = torch.cat([output, context_s, embedded_s], dim=1)
        logits = self.out(combined)
        return logits, hidden, attn_weights

# ---------- Tokenization (char-level) ----------
def build_token_helpers(tok2id, id2tok):
    PAD_IDX = tok2id.get(PAD_TOKEN, 0)
    SOS_IDX = tok2id.get(SOS_TOKEN, 1)
    EOS_IDX = tok2id.get(EOS_TOKEN, 2)
    UNK_IDX = tok2id.get(UNK_TOKEN, 3)

    def text_to_ids(s, max_len=MAX_SEQ_LEN):
        ids = [SOS_IDX]
        for ch in s:
            if ch in tok2id:
                ids.append(tok2id[ch])
            elif ch.lower() in tok2id:
                ids.append(tok2id[ch.lower()])
            else:
                ids.append(UNK_IDX)
            if len(ids) >= max_len - 1:
                break
        ids.append(EOS_IDX)
        while len(ids) < max_len:
            ids.append(PAD_IDX)
        return ids

    def ids_to_text(ids):
        out_chars = []
        for i in ids:
            if i == EOS_IDX:
                break
            token = id2tok.get(i, None)
            if token in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN) or token is None:
                continue
            out_chars.append(token)
        return "".join(out_chars)

    return text_to_ids, ids_to_text, PAD_IDX, SOS_IDX, EOS_IDX

# ---------- AGI wrapper ----------
class SynthMindAGI:
    def __init__(self, model_path=MODEL_PATH, vocab_path=VOCAB_PATH, device=DEVICE):
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.device = device
        self.id2tok, self.tok2id = load_vocab(self.vocab_path)
        self.text_to_ids, self.ids_to_text, self.PAD_IDX, self.SOS_IDX, self.EOS_IDX = build_token_helpers(self.tok2id, self.id2tok)
        self.encoder = Encoder(len(self.id2tok), EMBED_DIM, HIDDEN_SIZE, n_layers=ENC_LAYERS, dropout=DROPOUT).to(self.device)
        self.decoder = Decoder(len(self.id2tok), EMBED_DIM, HIDDEN_SIZE, enc_hidden_size=HIDDEN_SIZE*2, n_layers=DEC_LAYERS, dropout=DROPOUT).to(self.device)
        self._loaded = False
        if os.path.isfile(self.model_path):
            self.load_checkpoint(self.model_path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        enc_state = ckpt.get("encoder_state")
        dec_state = ckpt.get("decoder_state")
        vocab = ckpt.get("vocab")
        if vocab:
            save_vocab(vocab, self.vocab_path)
            self.id2tok = {i: t for i, t in enumerate(vocab)}
            self.tok2id = {t: i for i, t in self.id2tok.items()}
            self.text_to_ids, self.ids_to_text, self.PAD_IDX, self.SOS_IDX, self.EOS_IDX = build_token_helpers(self.tok2id, self.id2tok)
        if enc_state:
            self.encoder.load_state_dict(enc_state)
        if dec_state:
            self.decoder.load_state_dict(dec_state)
        self.encoder.to(self.device); self.decoder.to(self.device)
        self.encoder.eval(); self.decoder.eval()
        self._loaded = True
        return True

    def save_checkpoint(self, path=None):
        path = path or self.model_path
        ck = {
            "encoder_state": self.encoder.state_dict(),
            "decoder_state": self.decoder.state_dict(),
            "vocab": [self.id2tok[i] for i in range(len(self.id2tok))]
        }
        torch.save(ck, path)
        return path

    def greedy_decode(self, src_ids, max_len=MAX_SEQ_LEN):
        self.encoder.eval(); self.decoder.eval()
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            enc_outputs, enc_hidden = self.encoder(src_tensor)
            mask = (src_tensor != self.PAD_IDX).to(self.device)
            dec_hidden = enc_hidden
            dec_input = torch.tensor([[self.SOS_IDX]], dtype=torch.long, device=self.device)
            out_ids = []
            for _ in range(max_len):
                logits, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_outputs, mask)
                top1 = logits.argmax(1).item()
                if top1 == self.EOS_IDX:
                    break
                out_ids.append(top1)
                dec_input = torch.tensor([[top1]], dtype=torch.long, device=self.device)
            return out_ids

    def denoise(self, text):
        # small preprocess: normalize spaces, lowercase
        t = re.sub(r"\s+", " ", text.strip())
        src_ids = self.text_to_ids(t, max_len=MAX_SEQ_LEN)
        out_ids = self.greedy_decode(src_ids, max_len=MAX_SEQ_LEN)
        out_text = self.ids_to_text(out_ids)
        return out_text

# ---------- Simple REPL ----------
def repl():
    agi = SynthMindAGI()
    print("SynthMind AGI denoiser REPL. Type text to denoise.")
    print("Commands: ':save' to save model, ':exit' to quit, ':status' to show load status.")
    while True:
        try:
            s = input(">> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not s:
            continue
        if s == ":exit":
            break
        if s == ":status":
            print("Loaded checkpoint:", agi._loaded, "Model path:", agi.model_path)
            continue
        if s.startswith(":save"):
            path = agi.save_checkpoint()
            print("Saved checkpoint to", path)
            continue
        out = agi.denoise(s)
        print("=>", out)

if __name__ == "__main__":
    repl()
