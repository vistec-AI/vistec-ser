from typing import Callable, List

import tensorflow as tf


def pad_dup(x: tf.Tensor, max_len: tf.Tensor) -> tf.Tensor:
    """Pad an Arguments feature upto specified length.
    The Arguments is repeated until max_len is reached.
    """
    time_dim = tf.cast(tf.shape(x)[0], tf.float32)

    temp = tf.identity(x)
    num_repeat = tf.floor(max_len / time_dim)
    remainder = tf.cast(max_len - (num_repeat * time_dim), tf.int64)
    x_rem = x[:remainder, :]
    for _ in tf.range(num_repeat-1):
        x = tf.concat([x, temp], axis=0)
    x_pad = tf.concat([x, x_rem], axis=0)
    return x_pad


def pad_zero(x: tf.Tensor, max_len: tf.Tensor) -> tf.Tensor:
    """Pad Arguments feature up to specified length.
    The padded values are zero.
    Arguments
    """
    time_dim, freq_dim = tf.shape(x)[0], tf.shape(x)[1]
    time_dim = tf.cast(time_dim, tf.float32)
    freq_dim = tf.cast(freq_dim, tf.float32)

    zeros = tf.zeros((max_len - time_dim, freq_dim))
    x_pad = tf.concat([x, zeros], axis=0)
    return x_pad


def pad_X(X: List[tf.Tensor], pad_fn: Callable, max_len: int = None) -> tf.Tensor:
    """Pad a pack of array to a specified max_len.
    If max_len is not specified, longest preprocessing will
    be use as a max length. This function is used to
    pad the fbank features to its max_len, not chopped.
    Arguments
    """
    if not max_len:
        max_len = tf.cast(tf.maximum([tf.shape(x)[0] for x in X]), tf.float32)
    out = [pad_fn(x, max_len) for x in X]
    return tf.convert_to_tensor(out)