import numpy as np
import time
import sys

def generate_training_data(token_ids, window_size):
    n = len(token_ids)
    for i in range(n):
        w = np.random.randint(1, window_size + 1)
        start = max(0, i - w)
        end = min(n, i + w + 1)
        center = token_ids[i]
        contexts = []
        for j in range(start, end):
            if j != i:
                contexts.append(token_ids[j])
        yield center, contexts


def train(model, token_ids, neg_table, epochs, lr_start, lr_min,
          window_size, neg_samples):
    losses = []
    word_count = 0
    total_words = len(token_ids) * epochs

    for epoch in range(epochs):
        epoch_loss = 0
        pair_in_epoch = 0
        start_time = time.time()

        for center, contexts in generate_training_data(token_ids, window_size):

            lr = max(lr_start * (1 - word_count / total_words), lr_min)
            word_count += 1
            for context in contexts:
                neg_indices = neg_table[np.random.randint(0, len(neg_table), neg_samples)]
                loss = model.train_pair(center, context, neg_indices, lr)
                epoch_loss += loss
                pair_in_epoch += 1

            if word_count % 100000 == 0:
                elapsed = time.time() - start_time
                print(f"\r  Epoch {epoch + 1}: {word_count:,} words  "
                      f"loss: {epoch_loss / max(pair_in_epoch, 1):.4f}  "
                      f"lr: {lr:.6f}  {elapsed:.0f}s", end="")

        avg_loss = epoch_loss / max(pair_in_epoch, 1)
        losses.append(avg_loss)
        elapsed = time.time() - start_time
        print(f"\rEpoch {epoch + 1}/{epochs}  loss: {avg_loss:.4f} "
              f" lr: {lr:.6f}  {elapsed:.0f}s\033[K")
        sys.stdout.flush()
    return losses
