from glob import glob
from typing import Union, List
from multiprocessing import Pool
import os

import numpy as np
import tensorflow as tf

from .dataloader import DataLoader
from .features.featureloader import FeatureLoader
from ..utils import bytestring_feature, print_one_line

AUTOTUNE = tf.data.experimental.AUTOTUNE
TFRECORD_SHARDS = 16


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


class TFRecordDataset(DataLoader):
    def __init__(self,
                 csv_paths: Union[List[str], str],
                 tfrecords_dir: str,
                 stage: str,  # name of tfrecord (EX: `train_1.tfrecord` for 1st shard of train set tfrecord)
                 feature_loader: FeatureLoader,
                 shuffle: bool = True):
        super().__init__(csv_paths=csv_paths,
                         feature_loader=feature_loader,
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

        with Pool(TFRECORD_SHARDS) as pool:
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
