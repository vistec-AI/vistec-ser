import argparse
import warnings

from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from vistec_ser.models.network import CNN1DLSTMSlice
from vistec_ser.examples.train_aisser import read_config, evaluate_slice_model
from vistec_ser.utils.utils import load_yaml
warnings.filterwarnings("ignore")


def run_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(usage="Train AIS-SER Model")
    parser.add_argument("--config-path", "-cp",
                        default="examples/aisser.yaml", type=str, help="Path to training config file")
    return parser.parse_args()


def main(arguments):
    # load dataset & model
    config_path = arguments.config_path
    config = load_yaml(config_path)

    hparams, aisser_module = read_config(config)
    trainer_config = config.get("trainer", {})

    if aisser_module.slice_dataset:
        model = CNN1DLSTMSlice(hparams)
    else:
        model = CNN1DLSTMSlice(hparams)

    # dataloader
    aisser_module.setup()
    train_dataloader = aisser_module.train_dataloader()
    val_dataloader = aisser_module.val_dataloader()
    test_dataloader = aisser_module.test_dataloader()

    open(f"{aisser_module.experiment_dir}/results.txt", "w").write("WeightedAccuracy,UnweightedAccuracy\n")
    open(f"{aisser_module.experiment_dir}/confusion_matrix.txt", "w").write("")

    # trainer
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            save_top_k=5,
            monitor="val_acc",
            mode="min")
    ]
    logger = TensorBoardLogger(save_dir=aisser_module.experiment_dir)
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        weights_save_path=aisser_module.experiment_dir,
        **trainer_config)

    # train
    print("\n>>Training Model...\n")
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
    trainer.save_checkpoint(f"{aisser_module.experiment_dir}/weights/final.ckpt")

    # test
    print("\n>>Evaluating Model...\n")
    wa, ua, cm = evaluate_slice_model(model, test_dataloader, n_classes=aisser_module.n_classes)
    template = f"Confusion Matrix:\n{cm.numpy()}\nWeighted Accuracy: {wa*100:.2f}%\nUnweighted Accuracy: {ua*100:.2f}%"
    print(template)
    open(f"{aisser_module.experiment_dir}/results.txt", "a").write(f"{wa * 100:.2f},{ua * 100:.2f}\n")
    open(f"{aisser_module.experiment_dir}/confusion_matrix.txt", "a").write(f"{cm.numpy()}")


if __name__ == '__main__':
    args = run_parser()
    main(args)
