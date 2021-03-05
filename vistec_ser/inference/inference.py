from typing import Dict, List
import os

import torch.nn.functional as F
import torch

from ..models.base_model import BaseSliceModel
from ..models.network import CNN1DLSTMSlice
from ..utils.utils import read_config, load_yaml
from ..data.datasets.thaiser import ThaiSERDataModule


def setup_server(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file `{config_path}` not found.")

    config = load_yaml(config_path)
    inference_config = config.get("inference", {})

    temp_dir = inference_config.get("temp_dir", "./inference_temp")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    if "checkpoint_path" not in inference_config.keys():
        raise KeyError(f"Error: checkpoint_path not defined")
    checkpoint_path = inference_config["checkpoint_path"]
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint `{checkpoint_path}` not found.")


    hparams, module_params = read_config(config)
    thaiser_module = ThaiSERDataModule(**module_params)
    model = CNN1DLSTMSlice.load_from_checkpoint(checkpoint_path=checkpoint_path, hparams=hparams)
    model.eval()

    return model, thaiser_module, temp_dir


def infer_sample(model: BaseSliceModel, sample: List[Dict[str, torch.Tensor]], emotions=List[str]):
    name = os.path.basename(sample[0]["emotion"][0])
    final_logits = torch.stack([model(chunk["feature"]) for chunk in sample]).mean(dim=0)
    assert len(final_logits) == 1
    emotion_prob = F.softmax(final_logits[0], dim=-1)
    assert len(emotions) == len(emotion_prob), f"Number of emotion is not equal: len(emotions) = {len(emotions)}, " \
                                               f"len(final_logits) = {len(final_logits)} "
    emotion_prob = {emotion: f"{prob*100:.2f}" for emotion, prob in zip(emotions, emotion_prob)}
    return {"name": name, "prob": emotion_prob}
