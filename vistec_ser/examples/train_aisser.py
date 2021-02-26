import argparse
import warnings
from typing import Tuple

from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
import pytorch_lightning.metrics.functional as FM
import pytorch_lightning as pl
import torch

from vistec_ser.models.base_model import BaseSliceModel
from vistec_ser.models.network import CNN1DLSTMSlice
from vistec_ser.data.datasets.aisser import AISSERDataModule
from vistec_ser.utils.utils import load_yaml
warnings.filterwarnings("ignore")


def run_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(usage="Train AIS-SER Model")
    parser.add_argument("--config-path", "-cp",
                        default="examples/aisser.yaml", type=str, help="Path to training config file")
    parser.add_argument("--n-iter", "-n", default=25, type=int, help="Number of iteration")
    return parser.parse_args()


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


def run_fold(fold: int, config_path: str, n_iter: int = 25):
    # load dataset & model
    config = load_yaml(config_path)

    hparams, aisser_module = read_config(config, test_fold=fold)
    trainer_config = config.get("trainer", {})

    aisser_module.set_fold(fold)  # set fold to evaluate
    if aisser_module.slice_dataset:
        model = CNN1DLSTMSlice(hparams)
    else:
        model = CNN1DLSTMSlice(hparams)

    # trainer
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            save_top_k=5,
            monitor="val_acc",  # change to val_loss later
            mode="min"
        ),
    ]
    logger = TensorBoardLogger(
        save_dir=aisser_module.experiment_dir,
        name="iterations"
    )
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        weights_save_path=aisser_module.experiment_dir,
        **trainer_config)

    # dataloader
    aisser_module.setup()
    train_dataloader = aisser_module.train_dataloader()
    val_dataloader = aisser_module.val_dataloader()
    test_dataloader = aisser_module.test_dataloader()

    open(f"{aisser_module.experiment_dir}/results.txt", "w").write("WeightedAccuracy,UnweightedAccuracy\n")
    open(f"{aisser_module.experiment_dir}/confusion_matrix.txt", "w").write("")
    for i in range(n_iter):
        # train
        print(f"============= Running Experiment {i} =============")
        print("\n>>Training Model...\n")
        trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
        trainer.save_checkpoint(f"{aisser_module.experiment_dir}/final{n_iter}.ckpt")

        # test
        print("\n>>Evaluating Model...\n")
        wa, ua, cm = evaluate_slice_model(model, test_dataloader, n_classes=aisser_module.n_classes)
        template = f"Confusion Matrix:\n{cm.numpy()}\nWeighted Accuracy: {wa*100:.2f}%\nUnweighted Accuracy: {ua*100:.2f}%"
        print(template)
        open(f"{aisser_module.experiment_dir}/results.txt", "a").write(f"{wa*100:.2f},{ua*100:.2f}")
        open(f"{aisser_module.experiment_dir}/confusion_matrix.txt", "a").write(f"Iteration {i}:\n{cm.numpy()}\n\n")


def main(arguments):
    config_path = arguments.config_path
    config = load_yaml(config_path)
    aisser_config = config.get("aisser", {})
    aisser_module = AISSERDataModule(**aisser_config)

    for fold in aisser_module.fold_config.keys():
        print(f"\n+-----------------------------------------+")
        print(f"| Experiment on fold {fold:02d}                   |")
        print(f"+-----------------------------------------+\n")
        run_fold(fold, config_path)


if __name__ == '__main__':
    args = run_parser()
    main(args)
