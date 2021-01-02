from typing import Dict

from .spec_augments import FreqMasking, TimeMasking


AUGMENTATIONS = {
    "freq_masking": FreqMasking,
    "time_masking": TimeMasking,
}


class SpecFlow:
    def __init__(self, flow: list):
        self.flow = flow

    def augment(self, data):
        for f in self.flow:
            data = f.augment(data)
        return data


class Augmentation:
    def __init__(self, config: Dict[str, object] = None):
        if not config:
            config = dict()
        self.vtlp = config.get("wave_augment", {})
        self.feat_augment = self.parse_spec(config.get("feature_augment", {}))

    @staticmethod
    def parse_spec(config: dict):
        augmentations = list()
        for k, v in config.items():
            aug = AUGMENTATIONS.get(k, None)
            if aug is None:
                raise KeyError(f"Augmentation methods `{k}` not available. Only {AUGMENTATIONS.keys()} available.")
            augment = aug(**v) if v is not None else aug()
            augmentations.append(augment)
        return SpecFlow(augmentations)
