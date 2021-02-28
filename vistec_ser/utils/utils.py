from typing import Tuple
import os

import yaml

from ..data.datasets.aisser import AISSERDataModule


def load_yaml(config_path: str) -> dict:
    assert isinstance(config_path, str), "argument must be string"
    assert os.path.exists(config_path), f"config at `{config_path}` does not exists"
    assert config_path.split(".")[-1] in ["yaml", "yml"], f"Config file `{config_path}`"
    config = yaml.load(open(config_path, "r", encoding="utf-8"), Loader=yaml.SafeLoader)
    return config


def read_config(cfg: dict, test_fold: int = None) -> Tuple[dict, AISSERDataModule]:
    aisser_config = cfg.get("aisser", {})
    if test_fold is not None:
        aisser_config["test_fold"] = test_fold
    aisser_module = AISSERDataModule(**aisser_config)

    emotions = aisser_config.get("emotions", ["neutral", "anger", "happiness", "sadness"])
    in_channel = aisser_config.get("num_mel_bins", 40)
    sequence_length = aisser_config.get("max_len", 3) * aisser_module.sec_to_frame
    model_config = cfg.get("cnn1dlstm", {})
    hparams = {"in_channel": in_channel, "sequence_length": sequence_length, "n_classes": len(emotions), **model_config}
    return hparams, aisser_module
