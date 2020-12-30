import tensorflow_io as tfio
import tensorflow as tf
import numpy as np

from nlpaug.flow import Sequential
from nlpaug.util import Action
from nlpaug.model.spectrogram import Spectrogram
from nlpaug.augmenter.spectrogram import SpectrogramAugmenter


# --------------------- FREQ MASKING

class FreqMaskingModel(Spectrogram):
    def __init__(self, mask_factor: int = 27):
        super().__init__()
        self.mask_factor = mask_factor

    def mask(self, data: np.ndarray) -> np.ndarray:
        data = tf.convert_to_tensor(data)
        return tfio.experimental.audio.freq_mask(data, param=self.mask_factor).numpy()


class FreqMaskingAugmenter(SpectrogramAugmenter):
    def __init__(self,
                 mask_factor: float = 27,
                 name: str = "FreqMaskingAugmenter",
                 verbose=0):
        super().__init__(
            action=Action.SUBSTITUTE, zone=(0.2, 0.8), name=name, device="cpu", verbose=verbose,
            coverage=1., factor=(40, 80), silence=False, stateless=True)
        self.model = FreqMaskingModel(mask_factor)

    def substitute(self, data):
        return self.model.mask(data)


class FreqMasking(SpectrogramAugmenter):
    def __init__(self,
                 num_masks: int = 1,
                 mask_factor: float = 27,
                 name: str = "FreqMasking",
                 verbose=0):
        super().__init__(
            action=Action.SUBSTITUTE, zone=(0.2, 0.8), name=name, device="cpu", verbose=verbose,
            coverage=1., factor=(40, 80), silence=False, stateless=True)
        self.flow = Sequential([FreqMaskingAugmenter(mask_factor) for _ in range(num_masks)])

    def substitute(self, data):
        return self.flow.augment(data)

# --------------------- TIME MASKING


class TimeMaskingModel(Spectrogram):
    def __init__(self, mask_factor: float = 40, p_upperbound: float = 1.0):
        super().__init__()
        self.mask_factor = mask_factor

    def mask(self, data: np.ndarray) -> np.ndarray:
        data = tf.convert_to_tensor(data)
        return tfio.experimental.audio.time_mask(data, param=self.mask_factor).numpy()


class TimeMaskingAugmenter(SpectrogramAugmenter):
    def __init__(self,
                 mask_factor: float = 100,
                 p_upperbound: float = 1,
                 name: str = "TimeMaskingAugmenter",
                 verbose=0):
        super().__init__(
            action=Action.SUBSTITUTE, zone=(0.2, 0.8), name=name, device="cpu", verbose=verbose,
            coverage=1., silence=False, stateless=True)
        self.model = TimeMaskingModel(mask_factor, p_upperbound)

    def substitute(self, data):
        return self.model.mask(data)


class TimeMasking(SpectrogramAugmenter):
    def __init__(self,
                 num_masks: int = 1,
                 mask_factor: float = 100,
                 p_upperbound: float = 1,
                 name: str = "TimeMasking",
                 verbose=0):
        super().__init__(
            action=Action.SUBSTITUTE, zone=(0.2, 0.8), name=name, device="cpu", verbose=verbose,
            coverage=1., silence=False, stateless=True)
        self.flow = Sequential([
            TimeMaskingAugmenter(mask_factor, p_upperbound) for _ in range(num_masks)
        ])

    def substitute(self, data):
        return self.flow.augment(data)
