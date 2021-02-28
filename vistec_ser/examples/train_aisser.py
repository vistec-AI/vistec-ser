import argparse
import warnings

from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from vistec_ser.data.datasets.aisser import AISSERDataModule
from vistec_ser.evaluation.evaluate import evaluate_slice_model
from vistec_ser.models.network import CNN1DLSTMSlice, CNN1DLSTMAttentionSlice
from vistec_ser.utils.utils import load_yaml, read_config
warnings.filterwarnings("ignore")


def run_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(usage="Train AIS-SER Model")
    parser.add_argument("--config-path", "-cp",
                        default="examples/aisser.yaml", type=str, help="Path to training config file")
    parser.add_argument("--n-iter", "-n", default=25, type=int, help="Number of iteration")
    parser.add_argument("--attention", action="store_true", help="State whether to use attention LSTM or not")
    return parser.parse_args()


def run_fold(fold: int, config_path: str, n_iter: int = 25, use_attn=False):
    # load dataset & model
    config = load_yaml(config_path)
    hparams, aisser_module = read_config(config, test_fold=fold)
    aisser_module.set_fold(fold)  # set fold to evaluate

    # trainer
    trainer_config = config.get("trainer", {})
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            save_top_k=5,
            monitor="val_acc",
            mode="min"
        ),
    ]

    # dataloader
    aisser_module.setup()
    train_dataloader = aisser_module.train_dataloader()
    val_dataloader = aisser_module.val_dataloader()
    test_dataloader = aisser_module.test_dataloader()

    open(f"{aisser_module.experiment_dir}/results.txt", "w").write("WeightedAccuracy,UnweightedAccuracy\n")
    open(f"{aisser_module.experiment_dir}/confusion_matrix.txt", "w").write("")
    for i in range(n_iter):
        # reset model
        model = CNN1DLSTMAttentionSlice(hparams) if use_attn else CNN1DLSTMSlice(hparams)

        # trainer
        logger = TensorBoardLogger(
            save_dir=aisser_module.experiment_dir,
            name="iterations",
            version=f"iteration_{i}")
        trainer = pl.Trainer(
            callbacks=callbacks,
            logger=logger,
            weights_save_path=aisser_module.experiment_dir,
            **trainer_config)

        # train
        print(f"============= Running Experiment {i} =============")
        print("\n>>Training Model...\n")
        trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
        trainer.save_checkpoint(f"{aisser_module.experiment_dir}/weights/final{i}.ckpt")

        # test
        print("\n>>Evaluating Model...\n")
        wa, ua, cm = evaluate_slice_model(model, test_dataloader, n_classes=aisser_module.n_classes)
        template = f"Confusion Matrix:\n{cm.numpy()}\nWeighted Accuracy: {wa*100:.2f}%\nUnweighted Accuracy: {ua*100:.2f}%"
        print(template)
        open(f"{aisser_module.experiment_dir}/results.txt", "a").write(f"{wa*100:.2f},{ua*100:.2f}\n")
        open(f"{aisser_module.experiment_dir}/confusion_matrix.txt", "a").write(f"Iteration {i}:\n{cm.numpy()}\n\n")


def main(arguments):
    config_path = arguments.config_path
    use_attn = arguments.attention

    config = load_yaml(config_path)
    aisser_config = config.get("aisser", {})
    aisser_module = AISSERDataModule(**aisser_config)
    aisser_module.prepare_data()

    for fold in aisser_module.fold_config.keys():
        print(f"\n+-----------------------------------------+")
        print(f"| Experiment on fold {fold:02d}                   |")
        print(f"+-----------------------------------------+\n")
        run_fold(fold, config_path, use_attn=use_attn)


if __name__ == '__main__':
    args = run_parser()
    main(args)
