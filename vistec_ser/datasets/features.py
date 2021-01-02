from typing import List, Dict, Any

import tensorflow as tf


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
    sum_x2 = tf.reduce_sum(feature ** 2, axis=0) / tf.cast(tf.shape(feature)[0], dtype=tf.float32)
    sigma = sum_x2 - (mu ** 2) + epsilon
    normalized = (feature - mu) / sigma if scaled else feature - mu
    return normalized


def dbscale(spectrogram: tf.Tensor, top_db: int = 80):
    power = tf.math.square(spectrogram)
    log_spec = 10.0 * (tf.math.log(power) / tf.math.log(10.0))
    log_spec = tf.math.maximum(log_spec, tf.math.reduce_max(log_spec) - top_db)
    return log_spec


# ----------------- VTLP ----------------------


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


# ----------------- FeatureLoader -----------------


class FeatureLoader:
    def __init__(self, config: Dict[str, object]):
        self.sample_rate = config.get('sample_rate', 16000)
        self.max_length = int(self.sample_rate * (config.get('max_length', None) / 1000))  # in ms

        self.frame_length = int(self.sample_rate * (config.get("frame_length", 25) / 1000))
        self.frame_shift = int(self.sample_rate * (config.get("frame_shift", 10) / 1000))

        self.num_mel_bins = config.get('num_mel_bins', 80)
        self.feature_type = config.get('feature_type', 'fbank')
        self.preemphasis = config.get('preemphasis', 0.97)

        self.vtlp = config.get('vtlp', {})
        if len(self.vtlp) > 0:
            self.alpha_range = tf.range(
                self.vtlp.get("min_alpha", 0.9),
                self.vtlp.get("max_alpha", 1.1),
                self.vtlp.get("alpha_step", 0.1),
            )
        else:
            self.alpha_range = tf.ones([1])
        self.scale_normalize = config.get('scale_normalize', True)

    @property
    def shape(self) -> List[Any]:
        if self.feature_type == 'wave':
            return [self.max_length]
        elif self.feature_type == 'fbank':
            frame_shift = self.frame_shift * 1000. / self.sample_rate
            time_dim = int((self.max_length / self.sample_rate) * (1000. / frame_shift)) if self.max_length else None
            return [time_dim, self.num_mel_bins]
        else:
            frame_shift = self.frame_shift * 1000. / self.sample_rate
            time_dim = int((self.max_length / self.sample_rate) * (1000. / frame_shift)) if self.max_length else None
            return [time_dim, None]

    def sample_alpha(self):
        idxs = tf.range(tf.shape(self.alpha_range)[0])
        ridxs = tf.random.shuffle(idxs)[0]
        sampled_alpha = tf.gather(self.alpha_range, ridxs)
        return sampled_alpha

    def stft(self, waveform: tf.Tensor) -> tf.Tensor:
        spectrogram = tf.abs(
            tf.signal.stft(
                waveform,
                frame_length=self.frame_length,
                frame_step=self.frame_shift,
                window_fn=tf.signal.hann_window,
                pad_end=True
            )
        )
        return spectrogram
        # return apply_vtlp(spectrogram, alpha=self.sample_alpha())

    def make_spectrogram(self, waveform: tf.Tensor) -> tf.Tensor:
        spectrogram = self.stft(waveform)
        return dbscale(spectrogram)

    def make_fbank(self, waveform: tf.Tensor, fmin: float = 0., fmax: float = 8000.) -> tf.Tensor:
        spectrogram = self.stft(waveform)
        mel_fbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mel_bins,
            num_spectrogram_bins=spectrogram.shape[-1],
            sample_rate=self.sample_rate,
            lower_edge_hertz=fmin, upper_edge_hertz=fmax
        )
        fbank = tf.tensordot(spectrogram, mel_fbank, 1)
        return dbscale(fbank)

    def make_mfcc(self, waveform: tf.Tensor) -> tf.Tensor:
        log_mel_spectrogram = self.make_fbank(waveform)
        return tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)

    def extract(self, waveform: tf.Tensor) -> tf.Tensor:
        waveform = waveform[:self.max_length] if self.max_length else waveform
        waveform = pre_emphasis(waveform, coeff=self.preemphasis)

        if self.feature_type == "wave":
            return waveform
        elif self.feature_type == "spectrogram":
            features = self.make_spectrogram(waveform)
        elif self.feature_type == "fbank":
            features = self.make_fbank(waveform)
        elif self.feature_type == "mfcc":
            features = self.make_mfcc(waveform)
        else:
            raise KeyError(f"Unsupported feature type `{self.feature_type}`. Only `spectrogram`, `fbank`, and `mfcc` "
                           f"available.")

        return normalize_feature(features, scaled=self.scale_normalize)
