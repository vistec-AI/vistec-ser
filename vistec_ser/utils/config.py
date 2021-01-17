import re
from typing import Dict

import yaml


def load_yaml(path: str) -> Dict[str, object]:
    # Fix yaml numbers https://stackoverflow.com/a/30462009/11037553
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    with open(path, "r", encoding="utf-8") as file:
        return yaml.load(file, Loader=loader)


class FeatureConfig:
    def __init__(self, config: Dict[str, any]):
        self.sample_rate = config.get('sample_rate', 16000)
        self.frame_length = config.get('frame_length', 50)
        self.frame_shift = config.get('frame_shift', 10)
        self.num_mel_bins = config.get('num_mel_bins', 40)
        self.feature_type = config.get('feature_type', 'fbank')
        self.preemphasis = config.get('preemphasis', 0.97)
        self.scale_normalize = config.get('scale_normalize', False)


class Config:
    def __init__(self, path: str):
        config = load_yaml(path)
        self.feature_config = config.get('feature_config', {})
        self.model_config = config.get('model_config', {})
