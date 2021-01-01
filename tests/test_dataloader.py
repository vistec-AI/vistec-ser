from vistec_ser.datasets import DataLoader, FeatureLoader
from vistec_ser.utils.config import Config
import vistec_ser

import os

TEST_ROOT = os.path.dirname(vistec_ser.__file__)
CONFIG_PATH = os.path.join(TEST_ROOT, '../tests/config.yml')
CSV_PATH = os.path.join(TEST_ROOT, '../samples/lables.csv')


def main():
    config = Config(path=CONFIG_PATH)
    feature_loader = FeatureLoader(config=config.feature_config)
    data_loader = DataLoader(feature_loader=feature_loader, csv_paths=csv_path, augmentations=config.augmentations)
    dataset = data_loader.get_dataset(batch_size=2)
    for x, y in dataset:
        print(x.shape, y)


if __name__ == '__main__':
    main()
