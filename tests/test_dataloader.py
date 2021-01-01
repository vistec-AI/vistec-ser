from vistec_ser.datasets import DataLoader, FeatureLoader
from vistec_ser.utils.config import Config
import os

CONFIG_PATH = 'tests/config.yml'
CSV_PATH = 'tests/samples/lables.csv'


def main():
    assert os.path.exists(CONFIG_PATH), print(f"Config path does not exists in {os.getcwd()}")
    assert os.path.exists(CSV_PATH), print(f"CSV path does not exists in {os.getcwd()}")

    config = Config(path=CONFIG_PATH)
    feature_loader = FeatureLoader(config=config.feature_config)
    data_loader = DataLoader(feature_loader=feature_loader, csv_paths=CSV_PATH, augmentations=config.augmentations)
    dataset = data_loader.get_dataset(batch_size=2)
    for x, y in dataset:
        print(x.shape, y)


if __name__ == '__main__':
    main()
