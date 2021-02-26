import os

import yaml


def load_yaml(config_path: str) -> dict:
    assert isinstance(config_path, str), "argument must be string"
    assert os.path.exists(config_path), f"config at `{config_path}` does not exists"
    assert config_path.split(".")[-1] in ["yaml", "yml"], f"Config file `{config_path}`"
    config = yaml.load(open(config_path, "r", encoding="utf-8"), Loader=yaml.SafeLoader)
    return config
