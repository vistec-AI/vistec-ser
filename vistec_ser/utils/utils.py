from typing import Tuple
import os

import yaml


def load_yaml(config_path: str) -> dict:
    assert isinstance(config_path, str), "argument must be string"
    assert os.path.exists(config_path), f"config at `{config_path}` does not exists"
    assert config_path.split(".")[-1] in ["yaml", "yml"], f"Config file `{config_path}`"
    config = yaml.load(open(config_path, "r", encoding="utf-8"), Loader=yaml.SafeLoader)
    return config


def read_config(cfg: dict, test_fold: int = 0, include_zoom: bool = False) -> Tuple[dict, dict]:
    feature_config = cfg.get("feature", {"frame_shift": 10})
    thaiser_config = cfg.get("thaiser", {})
    module_params = {
        "test_fold": test_fold,
        "include_zoom": include_zoom,
        **thaiser_config, **feature_config}

    emotions = feature_config.get("emotions", ["neutral", "anger", "happiness", "sadness"])
    in_channel = feature_config.get("num_mel_bins", 40)
    sequence_length = feature_config.get("max_len", 3) * feature_config["frame_shift"] * 10
    model_config = cfg.get("cnn1dlstm", {})
    hparams = {"in_channel": in_channel, "sequence_length": sequence_length, "n_classes": len(emotions), **model_config}
    return hparams, module_params
