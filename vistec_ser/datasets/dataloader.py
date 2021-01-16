from typing import List, Union

import tensorflow as tf
import numpy as np

from .features.featureloader import FeatureLoader
from .features.preprocessing import load_waveform

AUTOTUNE = tf.data.experimental.AUTOTUNE
EMOTIONS = ['neutral', 'anger', 'sadness', 'happiness']


class DataLoader:
    def __init__(self,
                 csv_paths: Union[List[str], str],
                 feature_loader: FeatureLoader,
                 shuffle: bool = True):
        if isinstance(csv_paths, str):
            csv_paths = [csv_paths]
        self.csv_paths = csv_paths
        self.shuffle = shuffle
        self.feature_loader = feature_loader
        self.steps_per_epoch = None

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

    def get_dataset(self, batch_size) -> Union[tf.data.Dataset, None]:
        data = self.read_csv()
        if len(data) == 0:
            return None
        dataset = tf.data.Dataset.from_tensor_slices(data)
        return self.process_dataset(dataset, batch_size=batch_size)

    @tf.function
    def preprocess(self, data):
        audio_path = data[0]
        emotion = data[1]
        with tf.device('/CPU:0'):
            waveform = load_waveform(audio_path)

            # apply augmentation and feature extraction
            features = self.feature_loader.extract(waveform)

            label = tf.argmax(tf.stack([tf.equal(emotion, e) for e in EMOTIONS], axis=-1))
            return features, label

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
        dataset = dataset.map(self.preprocess, num_parallel_calls=AUTOTUNE)

        if self.shuffle:
            dataset = dataset.shuffle(16, reshuffle_each_iteration=True)

        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=(
                tf.TensorShape(self.feature_loader.shape),
                tf.TensorShape([])
            ),
            padding_values=(0., None),
            drop_remainder=False
        )

        dataset = dataset.prefetch(AUTOTUNE)
        self.steps_per_epoch = len(dataset)
        return dataset.repeat(None)
