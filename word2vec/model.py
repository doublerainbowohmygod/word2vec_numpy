import numpy as np

class Word2Vec:
    def __init__(self, vocab_size, embedding_dim=100):
        scale = 0.5 / embedding_dim
        self.W_center = np.random.uniform(-scale, scale,(vocab_size, embedding_dim))
        self.W_context = np.zeros((vocab_size, embedding_dim))

    def sigmoid(self, z):
        z = np.clip(z, -6, 6)
        return 1 / (1 + np.exp(-z))

    def train_pair(self, center_idx, context_idx, neg_indices, lr):
        center = self.W_center[center_idx]
        context_pos = self.W_context[context_idx]
        context_neg = self.W_context[neg_indices]

        score_pos = np.dot(center, context_pos)
        error_pos = self.sigmoid(score_pos) - 1

        score_neg = context_neg @ center
        error_neg = self.sigmoid(score_neg)

        grad_center = error_pos * context_pos + (error_neg[:,None] * context_neg).sum(axis=0)

        self.W_context[context_idx] -= lr * (error_pos * center)
        for k in range(len(neg_indices)):
            self.W_context[neg_indices[k]] -= lr * (error_neg[k] * center)
        self.W_center[center_idx] -= lr * grad_center

        loss = -np.log(self.sigmoid(score_pos) + 1e-7) - np.log(self.sigmoid(-score_neg) +1e-7).sum()
        return loss