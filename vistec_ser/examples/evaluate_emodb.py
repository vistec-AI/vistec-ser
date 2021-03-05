import argparse
import os

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from vistec_ser.utils.utils import load_yaml, read_config
from vistec_ser.data.ser_slice_dataset import SERSliceTestDataset
from vistec_ser.data.datasets.emodb import EmoDB
from vistec_ser.data.features.transform import FilterBank
from vistec_ser.evaluation.evaluate import evaluate_slice_model
from vistec_ser.models.network import CNN1DLSTMSlice


def run_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, help="path to config file")
    parser.add_argument("--checkpoint-path", type=str, help="path to model checkpoint")
    return parser.parse_args()


def main(args: argparse.Namespace):
    # configure parser
    config_path = args.config_path
    checkpoint_path = args.checkpoint_path
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config path not found at {config_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    config = load_yaml(config_path)
    hparams, _ = read_config(config)

    # load model
    model = CNN1DLSTMSlice.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        hparams=hparams)

    # load dataset
    feature_config = config.get("feature", {})
    emodb_config = config.get("emodb", {})
    emodb = EmoDB(**emodb_config, **feature_config)
    emodb.prepare_labels()

    # prepare dataloader
    transform = Compose([FilterBank(
        frame_length=emodb.frame_length,
        frame_shift=emodb.frame_shift,
        num_mel_bins=emodb.num_mel_bins)])
    emodb_loader = DataLoader(SERSliceTestDataset(
        csv_file=emodb.label_path,
        sampling_rate=emodb.sampling_rate,
        max_len=emodb.max_len,
        center_feats=emodb.center_feats,
        scale_feats=emodb.scale_feats,
        transform=transform
    ), batch_size=1)

    # evaluate
    print("\n>>Evaluating Model...\n")
    wa, ua, cm = evaluate_slice_model(model=model, test_dataloader=emodb_loader, n_classes=emodb.n_classes)
    template = f"Confusion Matrix:\n{cm.numpy()}\nWeighted Accuracy: {wa * 100:.2f}%\nUnweighted Accuracy: {ua * 100:.2f}%"
    print(template)
    open(f"{emodb.experiment_dir}/results.txt", "a").write(f"{wa * 100:.2f},{ua * 100:.2f}\n")
    open(f"{emodb.experiment_dir}/confusion_matrix.txt", "a").write(f"{cm.numpy()}")


if __name__ == '__main__':
    main(run_parser())
