import argparse
import warnings

from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from vistec_ser.data.datasets.thaiser import ThaiSERDataModule
from vistec_ser.evaluation.evaluate import evaluate_slice_model
from vistec_ser.models.network import CNN1DLSTMSlice, CNN1DLSTMAttentionSlice
from vistec_ser.utils.utils import load_yaml, read_config
warnings.filterwarnings("ignore")


def run_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(usage="Train AIS-SER Model")
    parser.add_argument("--config-path", "-cp",
                        default="examples/thaiser.yaml", type=str, help="Path to training config file")
    parser.add_argument("--include-zoom",
                        action="store_true", help="state whether to include Zoom Recording fold or not")
    parser.add_argument("--n-iter", "-n", default=25, type=int, help="Number of iteration")
    parser.add_argument("--attention", action="store_true", help="State whether to use attention LSTM or not")
    return parser.parse_args()


def run_fold(fold: int, config_path: str, include_zoom: bool, n_iter: int = 25, use_attn: bool = False):
    # load dataset & model
    config = load_yaml(config_path)
    hparams, module_params = read_config(config, test_fold=fold, include_zoom=include_zoom)
    thaiser_module = ThaiSERDataModule(**module_params)
    thaiser_module.set_fold(fold)  # set fold to evaluate

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
    thaiser_module.setup()
    train_dataloader = thaiser_module.train_dataloader()
    val_dataloader = thaiser_module.val_dataloader()
    test_dataloader = thaiser_module.test_dataloader()
    if not include_zoom:
        zoom_dataloader = thaiser_module.zoom_dataloader()
    else:
        zoom_dataloader = None

    open(f"{thaiser_module.experiment_dir}/results.txt", "w").write("WeightedAccuracy,UnweightedAccuracy\n")
    open(f"{thaiser_module.experiment_dir}/confusion_matrix.txt", "w").write("")
    for i in range(n_iter):
        # reset model
        model = CNN1DLSTMAttentionSlice(hparams) if use_attn else CNN1DLSTMSlice(hparams)

        # trainer
        logger = TensorBoardLogger(
            save_dir=thaiser_module.experiment_dir,
            name="iterations",
            version=f"iteration_{i}")
        trainer = pl.Trainer(
            callbacks=callbacks,
            logger=logger,
            weights_save_path=thaiser_module.experiment_dir,
            **trainer_config)

        # train
        print(f"============= Running Experiment {i} =============")
        print("\n>>Training Model...\n")
        trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
        trainer.save_checkpoint(f"{thaiser_module.experiment_dir}/weights/final{i}.ckpt")

        # test
        print("\n>>Evaluating Model...\n")
        wa, ua, cm = evaluate_slice_model(model, test_dataloader, n_classes=thaiser_module.n_classes)
        template = f"Confusion Matrix:\n{cm.numpy()}\nWeighted Accuracy: {wa*100:.2f}%\nUnweighted Accuracy: {ua*100:.2f}%"
        print(template)
        open(f"{thaiser_module.experiment_dir}/results.txt", "a").write(f"{wa*100:.2f},{ua*100:.2f}\n")
        open(f"{thaiser_module.experiment_dir}/confusion_matrix.txt", "a").write(f"Iteration {i}:\n{cm.numpy()}\n\n")

        # test zoom
        if not include_zoom:
            print("\n>>Evaluating Model (Zoom Test Set)...\n")
            wa, ua, cm = evaluate_slice_model(model, zoom_dataloader, n_classes=thaiser_module.n_classes)
            template = f"Confusion Matrix:\n{cm.numpy()}\nWeighted Accuracy: {wa * 100:.2f}%\nUnweighted Accuracy: {ua * 100:.2f}%"
            print(template)
            open(f"{thaiser_module.experiment_dir}/results_zoom.txt", "a").write(f"{wa * 100:.2f},{ua * 100:.2f}\n")
            open(f"{thaiser_module.experiment_dir}/confusion_matrix_zoom.txt", "a").write(f"{cm.numpy()}")


def main(arguments):
    config_path = arguments.config_path
    include_zoom = arguments.include_zoom
    use_attn = arguments.attention

    config = load_yaml(config_path)
    thaiser_config = config.get("thaiser", {})
    thaiser_module = ThaiSERDataModule(test_fold=0, **thaiser_config)
    thaiser_module.prepare_data()

    for fold in thaiser_module.fold_config.keys():
        print(f"\n+-----------------------------------------+")
        print(f"| Experiment on fold {fold:02d}                   |")
        print(f"+-----------------------------------------+\n")
        run_fold(fold, config_path, use_attn=use_attn, include_zoom=include_zoom)


if __name__ == '__main__':
    args = run_parser()
    main(args)
