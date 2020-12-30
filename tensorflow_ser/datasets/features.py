from typing import *

import tensorflow as tf


def load_waveform(audio_path: str, sample_rate: int = 16000, bit_depth: int = 16, dither: float = 1.) -> tf.Tensor:
    audio_binary = tf.io.read_file(audio_path)
    waveform, sr = tf.audio.decode_wav(audio_binary)
    waveform = tf.squeeze(waveform, axis=-1)
    return waveform + tf.random.normal(tf.shape(waveform), stddev=(dither / 2 ** bit_depth))


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


class FeatureLoader:
    def __init__(self, config: Dict[str, object]):
        self.sample_rate = config.get('sample_rate', 16000)
        self.frame_length = int(self.sample_rate * (config.get("frame_length", 25) / 1000))
        self.frame_shift = int(self.sample_rate * (config.get("frame_shift", 10) / 1000))

        self.num_mel_bins = config.get('num_mel_bins', 80)
        self.feature_type = config.get('feature_type', 'fbank')
        self.preemphasis = config.get('preemphasis', 0.97)

        self.scale_normalize = config.get('scale_normalize', True)

    @property
    def shape(self) -> List[int]:
        if self.feature_type == 'fbank':
            return [None, self.num_mel_bins]
        else:
            return [None, None]

    def stft(self, waveform: tf.Tensor) -> tf.Tensor:
        return tf.abs(
                tf.signal.stft(
                    waveform,
                    frame_length=self.frame_length,
                    frame_step=self.frame_shift,
                    window_fn=tf.signal.hann_window,
                    pad_end=True
                )
            )

    def dbscale(self, spectrogram: tf.Tensor, top_db: int=80):
        power = tf.math.square(spectrogram)
        log_spec = 10.0 * (tf.math.log(power) / tf.math.log(10.0))
        log_spec = tf.math.maximum(log_spec, tf.math.reduce_max(log_spec) - top_db)
        return log_spec

    def make_spectrogram(self, waveform: tf.Tensor, epsilon: float = 1e-8) -> tf.Tensor:
        spectrogram = self.stft(waveform)
        return self.dbscale(spectrogram)

    def make_fbank(self, waveform: tf.Tensor, fmin: float=0., fmax: float = 8000., epsilon: float = 1e-8) -> tf.Tensor:
        spectrogram = self.stft(waveform)
        mel_fbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.num_mel_bins,
            num_spectrogram_bins=spectrogram.shape[-1],
            sample_rate=self.sample_rate,
            lower_edge_hertz=fmin, upper_edge_hertz=fmax
        )
        fbank = tf.tensordot(spectrogram, mel_fbank, 1)
        return self.dbscale(fbank)

    def make_mfcc(self, waveform: tf.Tensor) -> tf.Tensor:
        log_mel_spectrogram = self.make_fbank(waveform)
        return tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)

    def extract(self, waveform: tf.Tensor) -> tf.Tensor:
        waveform = pre_emphasis(waveform, coeff=self.preemphasis)

        if self.feature_type == "spectrogram":
            features = self.make_spectrogram(waveform)
        elif self.feature_type == "fbank":
            features = self.make_fbank(waveform)
        elif self.feature_type == "mfcc":
            features = self.make_mfcc(waveform)
        else:
            raise KeyError("Unsupported feature type `{}`. Only `spectrogram`, `fbank`, and `mfcc` available.".format(self.feature_type))

        return normalize_feature(features, scaled=self.scale_normalize)
