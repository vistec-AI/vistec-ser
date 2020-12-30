from typing import *

import tensorflow as tf
import numpy as np

from .features import load_waveform, FeatureLoader
from augmentations.augmentation import Augmentation


AUTOTUNE = tf.data.experimental.AUTOTUNE
TFRECORD_SHARDS = 16
EMO2IDX = {'neutral': 0, 'anger': 1, 'happiness': 2, 'sadness': 3, 'frustration': 4}
IDX2EMO = {v : k for k, v in EMO2IDX.items()}


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
        self.total_steps = None
        self.feature_loader = feature_loader
        self.augmentations = augmentations

    def read_csv(self):
        lines = list()
        for csv in self.csv_paths:
            print(f"Reading {csv}...")
            with tf.io.gfile.GFile(csv, 'r') as f:
                lines = f.read().splitlines()[1:] # skip header
        lines = np.array([line.split(',') for line in lines]) # line i: audio_path, emotion
        if self.shuffle:
            np.random.shuffle(lines)
        self.total_steps = len(lines)
        return lines

    def get_dataset(self, batch_size):
        data = self.read_csv()
        if len(data) == 0:
            return None
        dataset = tf.data.Dataset.from_tensor_slices(data)
        return self.process_dataset(dataset, batch_size=batch_size)

    def preprocess(self, audio_path, emotion) -> Tuple[tf.Tensor, str]:
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

    def process_dataset(self, dataset: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
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
