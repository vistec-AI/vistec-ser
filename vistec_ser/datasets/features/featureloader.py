from typing import List, Dict, Any

import tensorflow as tf

from .preprocessing import dbscale, pre_emphasis, normalize_feature


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
