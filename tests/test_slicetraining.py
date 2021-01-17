from vistec_ser.datasets import DataLoader, FeatureLoader, SliceDataLoader
from vistec_ser.evaluation.metrics import WeightedAccuracy, UnweightedAccuracy
from vistec_ser.utils.config import Config
from vistec_ser.models import TestModel
import sys
import os


def test_slicedataloader(config: Config, batch_size: int, csv_path: str):
    feature_loader = FeatureLoader(config=config.feature_config)

    train_loader = SliceDataLoader(feature_loader=feature_loader, csv_paths=csv_path)
    X_train, y_train = train_loader.get_dataset()

    val_loader = DataLoader(feature_loader=feature_loader, csv_paths=csv_path)
    val_dataset = val_loader.get_dataset(batch_size=batch_size)
    validation_steps = val_loader.steps_per_epoch

    model = TestModel(config.model_config)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=[WeightedAccuracy(), UnweightedAccuracy()])
    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        epochs=1
    )


def main(argv):
    config_path, csv_path = argv[1], argv[2]
    assert os.path.exists(config_path), f"Config path `{config_path}` does not exists."
    assert os.path.exists(csv_path), f"CSV path `{csv_path}` does not exists."

    batch_size = 2
    config = Config(path=config_path)

    test_slicedataloader(config=config, batch_size=batch_size, csv_path=csv_path)


if __name__ == '__main__':
    main(sys.argv)
