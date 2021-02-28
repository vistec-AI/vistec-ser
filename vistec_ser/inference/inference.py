from typing import Dict, List
import os

import torch.nn.functional as F
import torch

from vistec_ser.models.base_model import BaseSliceModel
from vistec_ser.models.network import CNN1DLSTMSlice
from vistec_ser.utils.utils import read_config, load_yaml


def setup_server(temp_dir, config_path, checkpoint_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file `{config_path}` not found.")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint `{checkpoint_path}` not found.")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    hparams, aisser_module = read_config(load_yaml(config_path))
    model = CNN1DLSTMSlice.load_from_checkpoint(checkpoint_path=checkpoint_path, hparams=hparams)
    model.eval()
    return model, aisser_module


def infer_sample(model: BaseSliceModel, sample: List[Dict[str, torch.Tensor]], emotions=List[str]):
    name = os.path.basename(sample[0]["emotion"][0])
    final_logits = torch.stack([model(chunk["feature"]) for chunk in sample]).mean(dim=0)
    assert len(final_logits) == 1
    emotion_prob = F.softmax(final_logits[0], dim=-1)
    assert len(emotions) == len(emotion_prob), f"Number of emotion is not equal: len(emotions) = {len(emotions)}, " \
                                               f"len(final_logits) = {len(final_logits)} "
    emotion_prob = {emotion: f"{prob*100:.2f}" for emotion, prob in zip(emotions, emotion_prob)}
    return {"name": name, "prob": emotion_prob}
