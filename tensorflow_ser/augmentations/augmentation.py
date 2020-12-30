from typing import *

import nlpaug.flow as naf

from .audio_augments import SignalNoise, SignalMask, SignalCropping, SignalShift, SignalSpeed, SignalVtlp
from .spec_augments import FreqMasking, TimeMasking


AUGMENTATIONS = {
    "freq_masking": FreqMasking,
    "time_masking": TimeMasking,
    "noise": SignalNoise,
    "masking": SignalMask,
    "cropping": SignalCropping,
    "shift": SignalShift,
    "speed": SignalSpeed,
    "vtlp": SignalVtlp
}


class Augmentation:
    def __init__(self, config: Dict[str, object] = None):
        if not config:
            config = dict()
        self.wave_augment = self.parse(config.get("wave_augment", {}))
        self.feat_augment = self.parse(config.get("feature_augment", {}))

    @staticmethod
    def parse(config: dict) -> List[object]:
        augmentations = list()
        for k, v in config.items():
            aug = AUGMENTATIONS.get(k, None)
            if aug is None:
                raise KeyError(f"Augmentation methods `{k}` not available. Only {AUGMENTATIONS.keys()} available.")
            augment = aug(**v) if v is not None else aug()
            augmentations.append(augment)
        return naf.Sometimes(augmentations)
