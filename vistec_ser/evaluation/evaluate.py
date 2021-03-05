from typing import Tuple

from torch.utils.data import DataLoader
import pytorch_lightning.metrics.functional as FM
import torch

from ..models.base_model import BaseSliceModel


def evaluate_slice_model(
        model: BaseSliceModel,
        test_dataloader: DataLoader,
        n_classes: int = 4) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y_true, y_pred = [], []
    for batch_idx, batch in enumerate(test_dataloader):
        emotion = batch[0]["emotion"]
        final_logits = []
        for chunk in batch:
            logits = model(chunk["feature"])  # dim=(1, 4)
            final_logits.append(logits[0])
        prediction = torch.stack(final_logits).mean(dim=0).argmax(dim=-1, keepdim=True)
        y_true.append(emotion)
        y_pred.append(prediction)
    y_true = torch.stack(y_true).squeeze(-1)
    y_pred = torch.stack(y_pred).squeeze(-1)
    wa = FM.accuracy(y_pred, y_true)
    cm = FM.confusion_matrix(y_pred, y_true, normalize='true', num_classes=n_classes)
    ua = torch.diag(cm).mean()
    return wa, ua, cm
