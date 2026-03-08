import numpy as np
from word2vec.data import clean_text, build_vocab, make_neg_table, subsample_tokens
from word2vec.model import Word2Vec
from word2vec.train import train
from word2vec.config import (EMBEDDING_DIM, MIN_COUNT, EPOCHS,
                              LR_START, LR_MIN, WINDOW_SIZE, NEG_SAMPLES, SUBSAMPLE_THRESHOLD)

TXT_PATH = "data/wikitext-2-v1.txt"
def main():
    with open(TXT_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = clean_text(text)
    vocab, word2idx, idx2word = build_vocab(tokens, min_count=MIN_COUNT)
    print(f"Vocab size: {len(vocab)}")

    neg_table = make_neg_table(vocab, word2idx)
    token_ids = subsample_tokens(tokens, vocab, word2idx, threshold=SUBSAMPLE_THRESHOLD)
    print(f"Tokens after subsampling: {len(token_ids)}")

    model = Word2Vec(len(vocab), embedding_dim=EMBEDDING_DIM)
    losses = train(model, token_ids, neg_table,
                   epochs=EPOCHS,
                   lr_start=LR_START,
                   lr_min=LR_MIN,
                   window_size=WINDOW_SIZE,
                   neg_samples=NEG_SAMPLES)

    np.save("embeddings.npy", model.W_center)
    np.save("vocab.npy", np.array(list(word2idx.keys())))
    np.save("vocab_counts.npy", np.array([vocab[w] for w in word2idx]))
    np.save("losses.npy", np.array(losses))
    print(f"Done. Saved embeddings.npy, vocab.npy,  vocab_counts.npy")

if __name__ == "__main__":
    main()