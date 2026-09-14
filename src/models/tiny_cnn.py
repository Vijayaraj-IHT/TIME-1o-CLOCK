"""
Tiny CNN Speech Embedding Encoder
SIH Problem Statement 26172

Ultra-compact 2D CNN architecture designed for microcontroller SRAM constraints (ESP32-S3).
Outputs an L2-normalized 32-D or 64-D embedding on the unit hypersphere.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_tiny_cnn_encoder(input_shape=(98, 13, 1), embedding_dim=32, l2_normalize=True, model_name="tiny_cnn_encoder"):
    """
    Constructs a Tiny CNN speech embedding encoder.
    
    Args:
        input_shape: 3D tuple (time_frames, feature_dim, channels), default (98, 13, 1).
        embedding_dim: Dimensionality of output embedding space (default 32, or 64).
        l2_normalize: Boolean, whether to apply L2 unit hypersphere projection.
        model_name: String name for the Keras model.
        
    Returns:
        tf.keras.Model: The standalone embedding encoder.
    """
    inputs = layers.Input(shape=input_shape, name="audio_features")

    # Block 1
    x = layers.Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=False, name="conv1")(inputs)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)

    # Block 2
    x = layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=False, name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)

    # Block 3
    x = layers.Conv2D(48, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=False, name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)

    # Block 4
    x = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False, name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.ReLU(name="relu4")(x)

    # Global Average Pooling across spatial and temporal dimensions
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)

    # Dense Projection Head to Embedding Space
    embeddings = layers.Dense(embedding_dim, use_bias=False, name=f"dense_embed_{embedding_dim}")(x)

    # L2 Hypersphere Normalization (||e||_2 = 1.0)
    if l2_normalize:
        embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1), name="l2_normalize")(embeddings)

    encoder = models.Model(inputs=inputs, outputs=embeddings, name=model_name)
    return encoder
