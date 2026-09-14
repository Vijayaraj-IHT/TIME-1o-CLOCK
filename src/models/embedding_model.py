"""
High-Level Speech Embedding Model Abstraction
SIH Problem Statement 26172
"""

import os
import numpy as np
import tensorflow as tf
from src.models.tiny_cnn import build_tiny_cnn_encoder
from src.models.ds_cnn import build_ds_cnn_encoder

class SpeechEmbeddingModel:
    """
    Modular wrapper managing Tiny CNN and DS-CNN speech embedding encoders.
    """
    def __init__(self, architecture="tiny_cnn", input_shape=(98, 13, 1), embedding_dim=32, l2_normalize=True):
        self.architecture = architecture
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.l2_normalize = l2_normalize

        if architecture == "tiny_cnn":
            self.encoder = build_tiny_cnn_encoder(
                input_shape=input_shape,
                embedding_dim=embedding_dim,
                l2_normalize=l2_normalize
            )
        elif architecture in ["ds_cnn", "dscnn"]:
            self.encoder = build_ds_cnn_encoder(
                input_shape=input_shape,
                embedding_dim=embedding_dim,
                l2_normalize=l2_normalize
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}. Supported: 'tiny_cnn', 'ds_cnn'")

    def summary(self):
        return self.encoder.summary()

    def get_encoder(self):
        return self.encoder

    def encode(self, features):
        """
        Infers L2-normalized embedding vectors for a batch of features.
        """
        if features.ndim == 2:
            features = features[np.newaxis, :, :, np.newaxis]
        elif features.ndim == 3:
            features = features[:, :, :, np.newaxis]
        return self.encoder(features, training=False).numpy()

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.encoder.save(filepath)
