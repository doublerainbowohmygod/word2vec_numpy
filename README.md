# Word2Vec — Skip-gram with Negative Sampling (Pure NumPy)

Implementation of the Word2Vec skip-gram model with negative sampling,
using only NumPy. Trained on WikiText-2.

## Project structure

```
word2vec/
├── config.py      # Hyperparameters
├── data.py        # Text cleaning, vocab, subsampling, negative table
├── model.py       # Word2Vec model (forward + SGD update)
└── train.py       # Training loop with linear lr decay
main.py            # Entry point: preprocess → train → save
demo.ipynb         # Dataset EDA + evaluation: neighbours, analogies, t-SNE, PCA
```

An all-in-one notebook (`word2vec_all_in_one.ipynb`) is also provided for convenience — it contains the full pipeline (training + evaluation) in a single file that can be run in Google Colab (without dataset EDA).

Pre-trained embeddings are included in the repository (`embeddings.npy`, `vocab.npy`, `vocab_counts.npy`, `losses.npy`), so `demo.ipynb` can be run without re-training.

## Requirements

python 3.10+

## Quick start

```bash
pip install -r requirements.txt
python main.py
jupyter notebook demo.ipynb
```

## Implementation details

The implementation closely follows the original `word2vec.c` by Mikolov et al.

**Architecture:** Skip-gram with two embedding matrices — `W_center` (syn0) and `W_context` (syn1neg). Using two separate matrices prevents the trivial solution where each word is most similar to itself.

**Negative sampling loss:**

$$L = -\log\sigma(v_c \cdot v_o) - \sum_{i=1}^{k} \log\sigma(-v_c \cdot v_{n_i})$$

**Gradients:** For each pair, `error = σ(score) - label`, then `grad = error × other_vector`. Context vectors are updated individually in a loop (to correctly handle duplicate negative indices), then the center vector is updated last — matching the update order in the original C code.

**Initialization:** `W_center` uses `uniform(-0.5/dim, 0.5/dim)`, `W_context` is initialized to zeros — both matching the original implementation.

**Learning rate decay:** Linear decay per word (not per pair):
`lr = lr_start × (1 - word_count / total_words)`, clamped to `lr_min`. This matches the original C code where `alpha` is updated based on `word_count_actual`, not on the number of training pairs.

**Dynamic window:** For each center word, the actual window size is sampled uniformly from `[1, window_size]`. This implicitly gives more weight to closer context words — words at distance 1 always appear in the context, while words at distance 5 appear only 20% of the time.

**Subsampling:** Mikolov's frequency-based formula with `threshold = 1e-3`:
`P(keep) = (√(f/t) + 1) × (t/f)`. Removes ~37% of tokens, mostly stop words.

**Negative table:** Unigram distribution raised to power 0.75, precomputed via `searchsorted` for O(1) sampling.

**Sigmoid clipping:** Input to sigmoid is clipped to [-6, 6], mimicking the precomputed lookup table (`expTable`) in the original C implementation.

### Hyperparameters

| Parameter | Value |
|---|---|
| Embedding dim | 100 |
| Window size | 5 (dynamic, uniformly sampled 1..5) |
| Negative samples | 5 |
| Epochs | 5 |
| Learning rate | 0.025 → 0.0001 (linear decay) |
| Min count | 5 |
| Subsample threshold | 1e-3 |

## Training results

```
Vocab size: 19381
Tokens after subsampling: 1189211
Epoch 1/5  loss: 2.4739  lr: 0.020000  880s
Epoch 2/5  loss: 2.3119  lr: 0.015000  843s
Epoch 3/5  loss: 2.2946  lr: 0.010000  842s
Epoch 4/5  loss: 2.3036  lr: 0.005000  912s
Epoch 5/5  loss: 2.3406  lr: 0.000100  839s
```

## Observations

### Embedding quality

The embeddings capture meaningful semantic relationships. Nearest neighbours for common words are sensible:

- `king` → lord, emperor, throne
- `computer` → gaming, graphics, multiplayer, online
- `mother` → husband, father, daughter, wife, marriage
- `money` → pay, tickets, paid

Word analogies also produce reasonable results:

- `king - man + woman` → throne, ruler, lord
- `germany - berlin + london` → italy, switzerland, netherlands, austria
- `germany - beer + tea` → austria, turkey, finland, russia
- `hockey - ice + grass` → golf, sports, paralympic
- `poet - novel + music` → jazz, musicians, folk, pop

Some queries (e.g. `berlin`) return less coherent neighbours — likely because city names in WikiText-2 appear in diverse contexts (geography, history, culture), making the embedding more diffuse.

### Loss behaviour

Loss decreases for the first 3 epochs (2.47 → 2.29) and then slightly increases (2.30 → 2.34). This is consistent with overfitting on a small corpus. WikiText-2 has only ~1.2M tokens after subsampling, while the original word2vec was designed for billion-word corpora and typically trained for just 1 epoch.

With more data, the loss would be expected to decrease monotonically. On a small dataset, later epochs revisit the same contexts repeatedly, and the noise from random negative samples begins to outweigh the learning signal — especially as the learning rate approaches its minimum. A practical improvement would be early stopping at epoch 3.

Another option is using a learning rate schedule with slower decay — for example, cosine annealing, which is now standard in LLM training (see HuggingFace's *Smol Training Playbook*). However, since the task asks for a standard word2vec implementation, I kept the linear decay from the original.

### What I learned during development

An earlier version had learning rate decay computed per training pair rather than per word. Since each center word generates ~3 context pairs on average (with dynamic window), the learning rate was decaying roughly 3× faster than intended. This caused it to reach the minimum long before training was complete. The model produced poor embeddings where all cosine similarities were above 0.99 — essentially all word vectors pointed in the same direction. Comparing with the original C code confirmed that `word_count_actual` tracks words, not pairs.

The context embedding matrix (`W_context` / syn1neg) was initially initialized with random values. Changing it to zeros (matching the original code) improved training stability.

### Performance

Pure NumPy, single-threaded, online SGD (one pair at a time). Training takes approximately 70–80 minutes on Google Colab for 5 epochs (~15 min/epoch). The main bottleneck is per-pair Python function call overhead — approximately 18 million calls to `train_pair`.

The original C implementation achieves much higher throughput through compiled code with no interpreter overhead, multi-threading with shared memory, and a precomputed sigmoid lookup table (1000-element array for the [-6, 6] range).

A possible optimization without changing the algorithm: vectorizing updates across all context pairs for a single center word (collecting all contexts and their negatives into matrices and performing one batch update per word instead of one function call per pair). This would reduce Python call overhead while keeping the same online SGD semantics.

### Project variants

I provide two versions: a modular repository with separate files, and an all-in-one notebook for easy review and one-click reproduction. Both contain identical training and model code. Having not received detailed feedback on preferred format from previous applications, I included both for the reviewer's convenience.

## Evaluation (demo.ipynb)

- **Dataset analysis** — frequency distribution, Zipf's law, min_count impact
- **Nearest neighbours** — cosine similarity
- **Word analogies** — vector arithmetic (a − b + c ≈ ?)
- **Training loss curve**
- **t-SNE / PCA** — 2D visualisation of top-200 frequent words

## References

- Mikolov et al., *Efficient Estimation of Word Representations in Vector Space* (2013)
- Mikolov et al., *Distributed Representations of Words and Phrases and their Compositionality* (2013)
- [Original C implementation](https://github.com/tmikolov/word2vec/blob/master/word2vec.c)
- HuggingFace, *The Smol Training Playbook: The Secrets to Building World-Class LLMs* (2025)
