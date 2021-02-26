import torch
import torchaudio
from torchaudio.compliance import kaldi


class Spectrogram(object):
    """Convert audio into spectrogram with STFT, using torchaudio.compliance.kaldi
    see more at https://pytorch.org/audio/stable/compliance.kaldi.html

    Args:
        frame_length (float, optional) – Frame length in milliseconds (Default: 25.0)
        frame_shift (float, optional) – Frame shift in milliseconds (Default: 10.0)
        preemphasis_coefficient (float, optional) – Coefficient for use in signal preemphasis (Default: 0.97)
        window_type (str, optional) – Type of window (‘hamming’|’hanning’|’povey’|’rectangular’|’blackman’) (Default: 'povey')
        sample_frequency (float, optional) – Waveform data sample frequency (must match the waveform file, if specified there) (Default: 16000.0)
        dither (float, optional) – Dithering constant (0.0 means no dither). If you turn this off, you should set the energy_floor option, e.g. to 1.0 or 0.1 (Default: 0.0)
    """
    def __init__(
            self,
            frame_length: float = 25.,
            frame_shift: float = 10.,
            preemphasis_coefficient: float = 0.97,
            window_type: str = "povey",
            sample_frequency: float = 16000.,
            dither: float = 0.):
        self.spectrogram_params = {
            "frame_length": frame_length,
            "frame_shift": frame_shift,
            "preemphasis_coefficient": preemphasis_coefficient,
            "window_type": window_type,
            "sample_frequency": sample_frequency,
            "dither": dither
        }

    def __call__(self, sample):
        audio, emotion = sample["feature"], sample["emotion"]
        spectrogram = kaldi.spectrogram(audio, **self.spectrogram_params)
        spectrogram = torch.transpose(spectrogram, 0, 1)
        return {"feature": spectrogram, "emotion": emotion}


class FilterBank(object):
    """Convert audio into fbank feature, using torchaudio.compliance.kaldi
        see more at https://pytorch.org/audio/stable/compliance.kaldi.html

        Args:
            frame_length (float, optional) – Frame length in milliseconds (Default: 25.0)
            frame_shift (float, optional) – Frame shift in milliseconds (Default: 10.0)
            num_mel_bins (int, optional) – Number of triangular mel-frequency bins (Default: 23)
            preemphasis_coefficient (float, optional) – Coefficient for use in signal preemphasis (Default: 0.97)
            window_type (str, optional) – Type of window (‘hamming’|’hanning’|’povey’|’rectangular’|’blackman’) (Default: 'povey')
            sample_frequency (float, optional) – Waveform data sample frequency (must match the waveform file, if specified there) (Default: 16000.0)
            dither (float, optional) – Dithering constant (0.0 means no dither). If you turn this off, you should set the energy_floor option, e.g. to 1.0 or 0.1 (Default: 0.0)
            high_freq (float, optional) – High cutoff frequency for mel bins (if <= 0, offset from Nyquist) (Default: 0.0)
            low_freq (float, optional) – Low cutoff frequency for mel bins (Default: 20.0)
        """

    def __init__(
            self,
            frame_length: float = 50.,
            frame_shift: float = 10.,
            num_mel_bins: int = 40,
            preemphasis_coefficient: float = 0.97,
            window_type: str = "hanning",
            sample_frequency: float = 16000.,
            dither: float = 0.,
            low_freq: float = None,
            high_freq: float = None,
            vtln_max: float = 1.0,
            vtln_min: float = 1.0):
        if high_freq is None:
            high_freq = sample_frequency // 2
        if low_freq is None:
            low_freq = 0.
        self.vtln_range = torch.arange(vtln_min, vtln_max, 0.01)
        self.spectrogram_params = {
            "frame_length": frame_length,
            "frame_shift": frame_shift,
            "num_mel_bins": num_mel_bins,
            "preemphasis_coefficient": preemphasis_coefficient,
            "window_type": window_type,
            "sample_frequency": sample_frequency,
            "dither": dither,
            "high_freq": high_freq,
            "low_freq": low_freq
        }

    def _sample_vtln_factor(self):
        if len(self.vtln_range) == 0:
            return 1.0
        else:
            idx = torch.randint(0, len(self.vtln_range), (1,))[0]
            return self.vtln_range[idx]

    def __call__(self, sample):
        audio, emotion = sample["feature"], sample["emotion"]
        alpha = self._sample_vtln_factor()
        fbank = kaldi.fbank(audio, vtln_warp=alpha, **self.spectrogram_params)
        fbank = torch.transpose(fbank, 0, 1)
        return {"feature": fbank, "emotion": emotion}


class NormalizeSample(object):
    def __init__(self,
                 center_feats: bool = True,
                 scale_feats: bool = False):
        self.center_feats = center_feats
        self.scale_feats = scale_feats

    def __call__(self, sample):
        feature, emotion = sample["feature"], sample["emotion"]
        if self.center_feats:
            # feature = feature - feature.mean(dim=0)
            feature = feature - torch.unsqueeze(feature.mean(dim=-1), dim=-1)
        if self.scale_feats:
            # feature = feature / torch.sqrt(feature.var(dim=0) + 1e-8)
            feature = feature / torch.sqrt(torch.unsqueeze(feature.var(dim=-1), dim=-1) + 1e-8)
        return {"feature": feature, "emotion": emotion}


class SpecAugment(object):
    """Time/Freq Masking

    Args:
        freq_mask_param (int): Parameter for masking frequency axis
        time_mask_param (int): Parameter for masking time axis
        n_time_mask (int): number of mask on time axis
        n_freq_mask (int): number of mask on frequency axis
    """
    def __init__(
            self,
            freq_mask_param: int,
            time_mask_param: int,
            n_time_mask: int,
            n_freq_mask: int):
        assert isinstance(freq_mask_param, int)
        assert isinstance(time_mask_param, int)
        assert isinstance(n_freq_mask, int)
        assert isinstance(n_time_mask, int)

        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_time_mask = n_time_mask
        self.n_freq_mask = n_freq_mask

        self.freq_masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
        self.time_masking = torchaudio.transforms.TimeMasking(time_mask_param=time_mask_param)

    def __call__(self, sample):
        spec, emotion = sample["feature"], sample["emotion"]
        for _ in range(self.n_freq_mask):
            spec = self.freq_masking(spec)
        for _ in range(self.n_time_mask):
            spec = self.time_masking(spec)
        return {"feature": spec, "emotion": emotion}
