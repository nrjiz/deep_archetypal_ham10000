"""
Neural architectures for Deep Archetypal Analysis (DAA).

Notes:
- This project targets TensorFlow 1.x style graphs.
- If you are on TensorFlow 2.x, this file switches to TF1-compat mode.
"""

from __future__ import annotations

import math
from typing import Dict, Callable, List, Optional

import numpy as np

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except Exception:  # pragma: no cover
    import tensorflow as tf  # type: ignore

# Prefer tensorflow_probability if available (works for TF1 and TF2)
try:  # pragma: no cover
    import tensorflow_probability as tfp  # type: ignore
    tfd = tfp.distributions
except Exception:  # pragma: no cover
    tfd = tf.distributions


def share_variables(func):
    """
    Wrapper for tf.make_template as decorator.
    """
    return tf.make_template(func.__name__, func, create_scope_now_=True)


def build_prior(dim_latentspace: int):
    """
    Creates N(0,1) Multivariate Normal prior.
    """
    mu = tf.zeros(dim_latentspace)
    rho = tf.ones(dim_latentspace)
    mvn_diag = tfd.MultivariateNormalDiag(mu, rho)
    return mvn_diag


def build_encoder_basic(dim_latentspace: int, z_fixed: tf.Tensor):
    """
    Basic DAA encoder. (Flatten + FC layers)

    Returns dict:
      - z_predicted: (k, d)
      - mu: (batch, d)
      - sigma: (batch, d)
      - p: posterior distribution q(t|x)
    """

    @share_variables
    def encoder(data: tf.Tensor) -> Dict[str, tf.Tensor]:
        nAT = dim_latentspace + 1

        x = tf.layers.flatten(data)
        net = tf.layers.dense(x, 200, tf.nn.relu)
        net = tf.layers.dense(net, 100)
        mean_branch, var_branch = net[:, :50], net[:, 50:]

        # Weight Matrices
        weights_A = tf.layers.dense(mean_branch, nAT, tf.nn.softmax)
        weights_B_t = tf.layers.dense(mean_branch, nAT)
        weights_B = tf.nn.softmax(tf.transpose(weights_B_t), 1)

        # latent space parametrization
        mu_t = tf.matmul(weights_A, z_fixed)
        sigma_t = tf.layers.dense(var_branch, dim_latentspace, tf.nn.softplus)
        t = tfd.MultivariateNormalDiag(mu_t, sigma_t)

        # predicted archetypes
        z_predicted = tf.matmul(weights_B, mu_t)

        return {"z_predicted": z_predicted, "mu": mu_t, "sigma": sigma_t, "p": t}

    return encoder


def build_encoder_convs(dim_latentspace: int, z_fixed: tf.Tensor, x_shape: List[int]):
    """
    Convolutional DAA encoder (closer to the architecture described in the paper).
    """

    @share_variables
    def encoder(data: tf.Tensor) -> Dict[str, tf.Tensor]:
        nAT = dim_latentspace + 1
        activation = tf.nn.relu

        net = tf.layers.conv2d(tf.reshape(data, [-1] + x_shape), filters=64, kernel_size=4, strides=2,
                               padding='same', activation=activation)
        net = tf.layers.max_pooling2d(net, pool_size=2, strides=2)
        net = tf.layers.conv2d(net, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        net = tf.layers.max_pooling2d(net, pool_size=2, strides=2)
        net = tf.layers.conv2d(net, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        net = tf.layers.max_pooling2d(net, pool_size=2, strides=2)
        net = tf.layers.flatten(net)

        params = tf.layers.dense(net, 100)
        net, sigma = params[:, :50], params[:, 50:]

        # Weight Matrices
        weights_A = tf.layers.dense(net, nAT, tf.nn.softmax)
        weights_B_t = tf.layers.dense(net, nAT)
        weights_B = tf.nn.softmax(tf.transpose(weights_B_t), 1)

        # latent space parametrization as linear combination of archetypes
        mu_t = tf.matmul(weights_A, z_fixed)

        sigma_t = tf.layers.dense(sigma, dim_latentspace, tf.nn.softplus)
        t = tfd.MultivariateNormalDiag(mu_t, sigma_t)

        # predicted archetypes
        z_predicted = tf.matmul(weights_B, mu_t)

        return {"z_predicted": z_predicted, "mu": mu_t, "sigma": sigma_t, "p": t}

    return encoder


def dirichlet_prior(dim_latentspace: int, alpha: float = 1.0):
    """
    Dirichlet over archetype weights (k = d+1).
    """
    nATs = dim_latentspace + 1
    alpha_vec = [alpha] * nATs
    dist = tfd.Dirichlet(alpha_vec)
    return dist


def _decoder_upsample_filters(num_steps: int, base_filters: int = 64, min_filters: int = 8) -> List[int]:
    """
    Helper: for image sizes > 64 we add extra upsampling layers.
    64px: 3 steps -> [64, 32, 16]
    128px: 4 steps -> [64, 32, 16, 8]
    """
    filters = []
    for i in range(num_steps):
        f = max(base_filters // (2 ** i), min_filters)
        filters.append(f)
    return filters


def build_decoder(data_shape: List[int], num_labels: int, trainable_var: bool = False, init_stddev: float = 0.1):
    """
    Decoder with:
      - image branch (conv transpose)
      - side-info branch (FC)

    Supports square image sizes that are powers of two >= 64, e.g. 64, 128.
    """

    H, W, C = data_shape
    if H != W:
        raise ValueError(f"Decoder expects square images; got H={H}, W={W}")
    if H < 64 or (H & (H - 1)) != 0:
        raise ValueError(f"Image size must be power of two >= 64; got {H}")

    # Start from 8x8 grid
    base = 8
    up_steps = int(math.log2(H // base))

    @share_variables
    def decoder(latent_code: tf.Tensor) -> Dict[str, tf.Tensor]:
        activation = tf.nn.relu

        # Trainable (or fixed) decoder stddev for the Normal observation model
        # Parameterize via softplus to keep it positive.
        if trainable_var:
            log_scale = tf.get_variable(
                "decoder_log_scale",
                shape=[],
                dtype=tf.float32,
                initializer=tf.constant_initializer(np.log(init_stddev)),
                trainable=True,
            )
            scale = tf.nn.softplus(log_scale) + 1e-6
        else:
            scale = tf.constant(init_stddev, dtype=tf.float32)

        # start from a spatial grid with enough channels
        x = tf.layers.dense(latent_code, units=base * base * 64, activation=activation)
        x = tf.reshape(x, [-1, base, base, 64])

        # upsample base->...->H
        for f in _decoder_upsample_filters(up_steps):
            x = tf.layers.conv2d_transpose(x, filters=f, kernel_size=4, strides=2, padding='same', activation=activation)

        # final RGB image in [0,1]
        x = tf.layers.conv2d(x, filters=C, kernel_size=3, padding='same', activation=tf.nn.sigmoid)
        x_hat = tf.reshape(x, [-1] + data_shape)

        x_hat = tfd.Normal(loc=x_hat, scale=scale)
        x_hat = tfd.Independent(x_hat, 3)

        side_info = tf.layers.dense(latent_code, 200, tf.nn.relu)
        side_info = tf.layers.dense(side_info, num_labels, tf.nn.sigmoid)

        return {"x_hat": x_hat, "side_info": side_info}

    return decoder


def build_encoder_vae(dim_latentspace: int, x_shape: List[int]):
    """
    Vanilla VAE encoder (conv), used when training with --vae.
    """

    @share_variables
    def encoder(data: tf.Tensor) -> Dict[str, tf.Tensor]:
        activation = tf.nn.relu
        net = tf.layers.conv2d(tf.reshape(data, [-1] + x_shape), filters=64, kernel_size=4, strides=2,
                               padding='same', activation=activation)
        net = tf.layers.max_pooling2d(net, pool_size=2, strides=2)
        net = tf.layers.conv2d(net, filters=64, kernel_size=4, strides=2, padding='same', activation=activation)
        net = tf.layers.max_pooling2d(net, pool_size=2, strides=2)
        net = tf.layers.conv2d(net, filters=64, kernel_size=4, strides=1, padding='same', activation=activation)
        net = tf.layers.max_pooling2d(net, pool_size=2, strides=2)
        net = tf.layers.flatten(net)

        params = tf.layers.dense(net, 100)
        net, sigma = params[:, :50], params[:, 50:]

        sigma_t = tf.layers.dense(sigma, dim_latentspace, tf.nn.softplus)
        mu_t = tf.layers.dense(net, dim_latentspace)
        t = tfd.MultivariateNormalDiag(mu_t, sigma_t)

        return {"mu": mu_t, "sigma": sigma_t, "p": t}

    return encoder
