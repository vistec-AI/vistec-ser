from typing import Union, List, Tuple, Callable, Optional, Any

import tensorflow as tf

from .dataloader import DataLoader, EMOTIONS
from .features.featureloader import FeatureLoader
from .features.padding import pad_dup
from .features.preprocessing import load_waveform, chop_feature


class SliceDataLoader(DataLoader):
    def __init__(self,
                 csv_paths: Union[List[str], str],
                 feature_loader: FeatureLoader,
                 n_frames: int = 300,
                 thresh: int = 50,
                 pad_fn: Callable = pad_dup,
                 shuffle: bool = True):
        super().__init__(feature_loader=feature_loader,
                         csv_paths=csv_paths,
                         shuffle=shuffle)
        self.n_frames = n_frames
        self.thresh = thresh
        self.pad_fn = pad_fn

    def preprocess(self, data):
        audio_path, emotion = data[0], data[1]
        with tf.device('/CPU:0'):
            waveform = load_waveform(audio_path)

            features = self.feature_loader.extract(waveform)

            chunks = chop_feature(features, n_frames=self.n_frames, thresh=self.thresh, pad_fn=self.pad_fn)
            label = tf.argmax(tf.stack([tf.equal(emotion, e) for e in EMOTIONS], axis=-1))
            return chunks, label

    def get_dataset(self, **kwargs) -> Optional[Tuple[Union[Union[tf.Tensor, List[tf.Tensor]], Any], Any]]:
        data = self.read_csv()
        if len(data) == 0:
            return None
        y = tf.TensorArray(tf.int64, size=0, dynamic_size=True)

        data_iterator = iter(data)
        d = next(data_iterator)
        x, label = self.preprocess(d)
        y = y.write(y.size(), label)
        for d in data_iterator:
            chunks, label = self.preprocess(d)
            x = tf.concat([x, chunks], axis=0)
            y = y.write(y.size(), label)
        return x, y.stack()
