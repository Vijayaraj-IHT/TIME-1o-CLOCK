"""
Depthwise Separable CNN (DS-CNN) Speech Embedding Encoder
SIH Problem Statement 26172

Factorizes standard 2D convolutions into spatial depthwise and channel pointwise operations.
Drastically cuts MAC operations and parameter count for edge devices (ESP32-S3).
Outputs an L2-normalized 32-D or 64-D embedding on the unit hypersphere.
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def _ds_conv_block(x, filters, strides=(1, 1), block_id=1):
    """Depthwise Separable Convolution block: Depthwise Conv -> BN -> ReLU -> Pointwise Conv -> BN -> ReLU"""
    x = layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=strides,
        padding="same",
        use_bias=False,
        name=f"ds_depthwise_{block_id}"
    )(x)
    x = layers.BatchNormalization(name=f"ds_bn_dw_{block_id}")(x)
    x = layers.ReLU(name=f"ds_relu_dw_{block_id}")(x)

    x = layers.Conv2D(
        filters,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        name=f"ds_pointwise_{block_id}"
    )(x)
    x = layers.BatchNormalization(name=f"ds_bn_pw_{block_id}")(x)
    x = layers.ReLU(name=f"ds_relu_pw_{block_id}")(x)
    return x

def build_ds_cnn_encoder(input_shape=(98, 13, 1), embedding_dim=32, l2_normalize=True, model_name="ds_cnn_encoder"):
    """
    Constructs a Depthwise Separable CNN (DS-CNN) speech embedding encoder.
    """
    inputs = layers.Input(shape=input_shape, name="audio_features")

    # Initial standard Conv2D layer to expand feature representation
    x = layers.Conv2D(16, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=False, name="initial_conv")(inputs)
    x = layers.BatchNormalization(name="initial_bn")(x)
    x = layers.ReLU(name="initial_relu")(x)

    # Depthwise Separable Blocks
    x = _ds_conv_block(x, filters=32, strides=(1, 1), block_id=1)
    x = _ds_conv_block(x, filters=48, strides=(2, 2), block_id=2)
    x = _ds_conv_block(x, filters=64, strides=(2, 2), block_id=3)
    x = _ds_conv_block(x, filters=64, strides=(1, 1), block_id=4)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)

    # Dense Projection Head
    embeddings = layers.Dense(embedding_dim, use_bias=False, name=f"dense_embed_{embedding_dim}")(x)

    # L2 Hypersphere Normalization (||e||_2 = 1.0)
    if l2_normalize:
        embeddings = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1), name="l2_normalize")(embeddings)

    encoder = models.Model(inputs=inputs, outputs=embeddings, name=model_name)
    return encoder
