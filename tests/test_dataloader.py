from vistec_ser.datasets import DataLoader, FeatureLoader
from vistec_ser.utils.config import Config
import sys
import os

CONFIG_PATH = 'config.yml'


def main(args):
    csv_path = args[1]
    config = Config(path=CONFIG_PATH)
    feature_loader = FeatureLoader(config=config.feature_config)
    data_loader = DataLoader(feature_loader=feature_loader, csv_paths=csv_path, augmentations=config.augmentations)
    dataset = data_loader.get_dataset(batch_size=2)
    for x, y in dataset:
        print(x.shape, y)


if __name__ == '__main__':
    main(sys.argv)
