"""
Metric Learning Loss Functions
SIH Problem Statement 26172
"""

import tensorflow as tf

class ContrastiveLoss(tf.keras.losses.Loss):
    """
    Contrastive Loss for pair-based metric learning.
    L = y * d^2 + (1 - y) * max(0, margin - d)^2
    where y=1 for positive pairs (same word), y=0 for negative pairs (different words).
    """
    def __init__(self, margin=1.0, name="contrastive_loss"):
        super().__init__(name=name)
        self.margin = margin

    def call(self, y_true, y_pred):
        # y_true: binary label (1: same word, 0: different word)
        # y_pred: pairwise Euclidean distance between embeddings
        y_true = tf.cast(y_true, tf.float32)
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(self.margin - y_pred, 0.0))
        return tf.reduce_mean(y_true * square_pred + (1.0 - y_true) * margin_square)

class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    """
    Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)
    Optimizes representation space such that all instances of the same class
    are pulled together while instances of different classes are pushed apart.
    """
    def __init__(self, temperature=0.07, name="supcon_loss"):
        super().__init__(name=name)
        self.temperature = temperature

    def call(self, labels, embeddings):
        # labels: (batch_size, 1) integer class IDs
        # embeddings: (batch_size, embedding_dim) L2 normalized vectors
        labels = tf.reshape(labels, [-1, 1])
        mask = tf.cast(tf.equal(labels, tf.transpose(labels)), tf.float32)

        # Dot product between all pairs (cosine similarity since vectors are L2-normalized)
        similarity = tf.matmul(embeddings, tf.transpose(embeddings)) / self.temperature

        # For numerical stability
        logits_max = tf.reduce_max(similarity, axis=-1, keepdims=True)
        logits = similarity - tf.stop_gradient(logits_max)

        # Mask out self-similarity from diagonal
        logits_mask = tf.ones_like(mask) - tf.eye(tf.shape(mask)[0])
        mask = mask * logits_mask

        # Compute log-likelihood
        exp_logits = tf.exp(logits) * logits_mask
        log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=-1, keepdims=True) + 1e-7)

        # Mean of positive log-probabilities
        num_positives = tf.reduce_sum(mask, axis=-1)
        mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=-1) / tf.maximum(num_positives, 1.0)

        # Loss is negative log-likelihood
        loss = -mean_log_prob_pos
        # Only compute loss for samples that have at least one positive in the batch
        valid_samples = tf.cast(num_positives > 0, tf.float32)
        return tf.reduce_sum(loss * valid_samples) / tf.maximum(tf.reduce_sum(valid_samples), 1.0)
