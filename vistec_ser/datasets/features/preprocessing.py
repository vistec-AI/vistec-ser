from typing import Callable, Union, List

import tensorflow as tf

from .padding import pad_dup


def load_waveform(
        audio: str,
        bit_depth: int = 16,
        dither: float = 1.) -> tf.Tensor:
    audio_binary = tf.io.read_file(audio)
    waveform, sr = tf.audio.decode_wav(audio_binary)
    # if sr != sample_rate:
    #     raise ValueError(f"Expected sampling rate of {sample_rate} but got {sr}")
    waveform = tf.squeeze(waveform, axis=-1)
    waveform = waveform + tf.random.normal(tf.shape(waveform), stddev=(dither / 2 ** bit_depth))
    return waveform


def pre_emphasis(waveform: tf.Tensor, coeff: float = 0.97) -> tf.Tensor:
    if isinstance(coeff, float) or coeff <= 0.0:
        return waveform
    return tf.concat([waveform[0], waveform[1:] - coeff * waveform[:-1]])


def normalize_feature(feature: tf.Tensor, scaled: bool = False, epsilon: float = 1e-8) -> tf.Tensor:
    mu = tf.reduce_mean(feature, axis=0)
    sum_x2 = tf.math.reduce_sum(feature ** 2, axis=0) / tf.cast(tf.shape(feature)[0], dtype=tf.float32)
    sigma = sum_x2 - (mu ** 2) + epsilon
    normalized = (feature - mu) / sigma if scaled else feature - mu
    return normalized


def dbscale(spectrogram: tf.Tensor, top_db: int = 80):
    power = tf.math.square(spectrogram)
    log_spec = 10.0 * (tf.math.log(power) / tf.math.log(10.0))
    log_spec = tf.math.maximum(log_spec, tf.math.reduce_max(log_spec) - top_db)
    return log_spec

# ----------------- VTLP ----------------------


def chop_feature(
        x: tf.Tensor,
        n_frames: int,
        thresh: int = 50,
        pad_fn: Callable = pad_dup) -> Union[tf.Tensor, List[tf.Tensor]]:
    """Chop fbank into smaller chunk. Pad function is
    optional. If not specified, this function will return
    a list of preprocessing.
    """
    time_dim = tf.shape(x)[0]
    x_chopped = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for i in tf.range(time_dim):
        if i % n_frames == 0 and i != 0:  # if reach n_frames, cut
            xi = x[i - n_frames:i]
            x_chopped = x_chopped.write(x_chopped.size(), xi)
    if i < time_dim and i < n_frames:  # if file length is less than n_frames, pad
        xi = pad_fn(x, max_len=n_frames)
        x_chopped = x_chopped.write(x_chopped.size(), xi)
    else:  # if file is longer than n_frame, pad remainder
        remainder = x[time_dim - time_dim % n_frames:]
        t_remainder = tf.shape(remainder)[0]
        if t_remainder > thresh:
            xi = pad_fn(remainder, max_len=n_frames)
            x_chopped = x_chopped.write(x_chopped.size(), xi)
    if pad_fn:
        return x_chopped.stack()
    return x_chopped

# ----------------- VTLP (Experimental) ----------------------


def get_scale_factors(freq_dim, sampling_rate, f_hi=4800., alpha=0.9):
    factors = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    freqs = tf.cast(tf.linspace(0, 1, freq_dim), tf.float32)

    scale = f_hi * tf.minimum(alpha, 1.)
    f_boundary = scale / alpha
    half_sr = sampling_rate / 2

    for i in tf.range(len(freqs)):
        f = sampling_rate * freqs[i]
        if tf.math.less_equal(f, f_boundary):
            factors = factors.write(factors.size(), f * alpha)
        else:
            warp_freq = half_sr - (half_sr - scale) / (half_sr - scale / alpha) * (half_sr - f)
            factors = factors.write(factors.size(), warp_freq)
    return factors.stack()


def apply_vtlp(spec, sampling_rate: int = 16000, f_hi: int = 4800, alpha=0.9):
    time_dim, freq_dim = tf.shape(spec)[0], tf.shape(spec)[1]
    factors = get_scale_factors(freq_dim, sampling_rate, f_hi=f_hi, alpha=alpha)
    factors = factors * tf.cast(freq_dim - 1, tf.float32) / tf.reduce_max(factors)
    new_spec = tf.zeros([freq_dim, time_dim], dtype=tf.float32)

    for i in tf.range(freq_dim):
        if tf.equal(i, 0) or tf.greater_equal(i + 1, freq_dim):
            update = tf.expand_dims(spec[:, i], 0)
            new_spec = tf.tensor_scatter_nd_add(new_spec, [[i]], update)
        else:
            warp_up = factors[i] - tf.floor(factors[i])
            warp_down = 1. - warp_up
            pos = tf.cast(tf.floor(factors[i]), tf.int64)

            new_spec = tf.tensor_scatter_nd_add(new_spec, [[pos]], tf.expand_dims(warp_down * spec[:, i], 0))
            new_spec = tf.tensor_scatter_nd_add(new_spec, [[pos + 1]], tf.expand_dims(warp_up * spec[:, i], 0))
    return tf.transpose(new_spec, perm=(1, 0))
