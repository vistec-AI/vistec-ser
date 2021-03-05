import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning.metrics import functional as FM
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import Adam


class BaseModel(pl.LightningModule):
    def __init__(self, hparams, **kwargs):
        super().__init__(**kwargs)
        self.hparams = hparams
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        x, y = batch["feature"], batch["emotion"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=-1)
        acc = FM.accuracy(preds, y)
        metrics = {"train_loss": loss, "train_acc": acc}
        self.log_dict(metrics, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["feature"], batch["emotion"]
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=-1)
        acc = FM.accuracy(preds, y)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics, prog_bar=True, logger=True)
        return metrics

    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx)
        metrics = {'test_acc': metrics['val_acc'], 'test_loss': metrics['val_loss']}
        self.log_dict(metrics, prog_bar=True, logger=True)

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.hparams.learning_rate)
        return opt


class BaseSliceModel(BaseModel):

    def __init__(self, hparams, **kwargs):
        super().__init__(hparams, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        emotion = batch[0]["emotion"]
        final_logits = torch.stack([self(chunk["feature"])[0] for chunk in batch]).mean(dim=0, keepdim=True)
        loss = F.cross_entropy(final_logits, emotion)
        acc = FM.accuracy(final_logits.argmax(dim=-1), emotion)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics
