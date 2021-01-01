from typing import *
import multiprocessing as mp

from glob import glob
import tensorflow as tf
import numpy as np
import os

from .features import load_waveform, FeatureLoader
from ..augmentations.augmentation import Augmentation
from ..utils import bytestring_feature, print_one_line


AUTOTUNE = tf.data.experimental.AUTOTUNE
TFRECORD_SHARDS = 16
EMO2IDX = {'neutral': 0, 'anger': 1, 'happiness': 2, 'sadness': 3, 'frustration': 4}
IDX2EMO = {v: k for k, v in EMO2IDX.items()}


def to_tfrecord(path: bytes, audio: bytes, emotion: int) -> tf.train.Example:
    feature = {
        "path": bytestring_feature([path]),
        "audio": bytestring_feature([audio]),
        "emotion": bytestring_feature([emotion])
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecord_file(splitted_csv: List[np.ndarray]):
    shard_path, data = splitted_csv
    with tf.io.TFRecordWriter(shard_path, options='ZLIB') as out:
        for audio_path, emotion in data:
            audio = open(audio_path, 'rb').read()
            example = to_tfrecord(
                path=bytes(audio_path, 'utf-8'),
                audio=audio,
                emotion=emotion
            )
            out.write(example.SerializeToString())
            print_one_line("Processed:", audio_path)
    print(f"\nCreated {shard_path}")


class DataLoader:
    def __init__(self,
                 csv_paths: Union[List[str], str],
                 feature_loader: FeatureLoader,
                 augmentations: Augmentation = Augmentation(None),
                 shuffle: bool = True):
        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]
        self.csv_paths = csv_paths
        self.shuffle = shuffle
        self.feature_loader = feature_loader
        self.augmentations = augmentations

    def read_csv(self) -> np.ndarray:
        lines = list()
        for csv in self.csv_paths:
            print(f"Reading {csv}...")
            with tf.io.gfile.GFile(csv, 'r') as f:
                lines += f.read().splitlines()[1:]  # skip header
        lines = np.array([line.split(',') for line in lines])  # line i: audio_path, emotion
        if self.shuffle:
            np.random.shuffle(lines)
        return lines

    def get_dataset(self, batch_size) -> tf.data.Dataset:
        data = self.read_csv()
        if len(data) == 0:
            return None
        dataset = tf.data.Dataset.from_tensor_slices(data)
        return self.process_dataset(dataset, batch_size=batch_size)

    def preprocess(self, audio_path, emotion) -> Tuple[Any, int]:
        with tf.device('/CPU:0'):
            waveform = load_waveform(audio_path)

            # apply augmentation and feature extraction
            waveform = self.augmentations.wave_augment.augment(waveform.numpy())
            features = self.feature_loader.extract(waveform)
            features = self.augmentations.feat_augment.augment(features)
            features = tf.convert_to_tensor(features)

            return features, EMO2IDX[emotion.decode()]

    @tf.function
    def parse(self, data):
        return tf.numpy_function(
            self.preprocess,
            inp=[data[0], data[1]],
            Tout=[tf.float32, tf.int64]
        )

    def process_dataset(self,
                        dataset: Union[tf.data.Dataset, tf.data.TFRecordDataset],
                        batch_size: int) -> tf.data.Dataset:
        dataset = dataset.map(self.parse, num_parallel_calls=AUTOTUNE)

        if self.shuffle:
            dataset = dataset.shuffle(TFRECORD_SHARDS, reshuffle_each_iteration=True)

        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=(
                tf.TensorShape(self.feature_loader.shape),
                tf.TensorShape([])
            ),
            padding_values=(0., None),
            drop_remainder=True
        )

        dataset = dataset.prefetch(AUTOTUNE)
        return dataset


class TFRecordDataset(DataLoader):
    def __init__(self,
                 csv_paths: Union[List[str], str],
                 tfrecords_dir: str,
                 stage: str,  # name of tfrecord (EX: `train_1.tfrecord` for 1st shard of train set tfrecord)
                 feature_loader: FeatureLoader,
                 augmentations: Augmentation = Augmentation(None),
                 shuffle: bool = True):
        super().__init__(self,
                         csv_paths=csv_paths,
                         feature_loader=feature_loader,
                         augmentations=augmentations,
                         shuffle=shuffle)
        self.stage = stage
        self.tfrecords_dir = tfrecords_dir
        if not os.path.exists(self.tfrecords_dir):
            os.makedirs(self.tfrecords_dir)

    def create_tfrecords(self):
        if glob(os.path.join(self.tfrecords_dir, f'{self.stage}*.tfrecord')):
            print("TFRecord files exists.")
            return True

        print(f"Creating {self.stage}.tfrecord...")

        csv = self.read_csv()
        if len(csv) == 0:
            return False

        def get_shard_path(shard_id):
            return os.path.join(self.tfrecords_dir, f'{self.stage}_{shard_id}.tfrecord')

        shards = [get_shard_path(i+1) for i in range(TFRECORD_SHARDS)]
        splitted_entries = np.array_split(csv, TFRECORD_SHARDS)

        with mp.Pool(TFRECORD_SHARDS) as pool:
            pool.map(write_tfrecord_file, zip(shards, splitted_entries))

        return True

    @tf.function
    def parse(self, record):
        feature_description = {
            "path": tf.io.FixedLenFeature([], tf.string),
            "audio": tf.io.FixedLenFeature([], tf.string),
            "emotion": tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(record, feature_description)

        return tf.numpy_function(
            self.preprocess,
            inp=[example["audio"], example["emotion"]],
            Tout=[tf.float32, tf.int64]
        )

    def get_dataset(self, batch_size: int) -> Union[tf.data.Dataset, None]:
        if not self.create_tfrecords:
            return None

        pattern = os.path.join(self.tfrecords_dir, f"{self.stage}*.tfrecord")
        files_ds = tf.data.Dataset.list_files(pattern)
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        files_ds = files_ds.with_options(ignore_order)
        dataset = tf.data.TFRecordDataset(files_ds, compression_type='ZLIB', num_parallel_reads=AUTOTUNE)

        return self.process_dataset(dataset, batch_size)
