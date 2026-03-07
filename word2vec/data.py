import numpy as np
import re
from collections import Counter

def clean_text(raw_text):
    text = raw_text.lower()
    text = text.replace("<unk>", " ")
    text = re.sub(r"^=.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^a-z]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 1]
    return tokens

def build_vocab(tokens, min_count):
    counts = Counter(tokens)
    vocab = {w: c for w, c in counts.items() if c >= min_count}
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for w, i in word2idx.items()}
    return vocab, word2idx, idx2word

def make_neg_table(vocab, word2idx, table_size=1_000_000, power=0.75):
    freqs = np.array([vocab[w] for w in word2idx])
    freqs = freqs ** power
    freqs_norm = freqs / freqs.sum()
    cumul = np.cumsum(freqs_norm)
    positions = np.arange(table_size) / table_size
    table = np.searchsorted(cumul, positions).astype(np.int32)
    return table

def subsample_tokens(tokens, vocab, word2idx, threshold=1e-3):
    indices = np.array([word2idx[t] for t in tokens if t in word2idx])
    counts = np.array([vocab[t] for t in tokens if t in word2idx])
    total = sum(vocab.values())
    freqs_norm = counts / total
    keep_prob = (np.sqrt(freqs_norm / threshold) + 1) * (threshold/ freqs_norm)
    mask = np.random.random(len(indices)) < keep_prob
    return indices[mask]