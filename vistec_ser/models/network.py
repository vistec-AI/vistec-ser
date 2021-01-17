from typing import Dict, Any

from tensorflow.keras.layers import Conv1D, BatchNormalization, LSTM, Bidirectional, Dense, Masking
import tensorflow as tf

from .base_model import BaseModel
from .layers.cnn_blocks import CNN1DBlock


def get_cnn(filters: int, kernel_size: int, stride: int = 1, activation: str = 'relu'):
    return tf.keras.Sequential([
        Conv1D(filters=filters, kernel_size=kernel_size, strides=stride, activation=activation),
        BatchNormalization()
    ])


class CNN1DLSTM(BaseModel):
    def __init__(
            self,
            config: Dict[str, Any] = None,
            **kwargs):
        super().__init__(config, **kwargs)

        if config is None:
            config = {}
        self.n_classes = config.get('n_classes', 4)
        self.num_channels = config.get('num_channels', [64, 64, 128, 128])
        self.kernel_sizes = config.get('kernel_sizes', [4, 3, 3, 3])
        self.pool_sizes = config.get('pool_sizes', [2, 2, 2, 2])
        self.bidirectional = config.get('bidirectional', True)
        self.rnn_units = config.get('rnn_units', [128])
        self.norm_method = config.get('norm_method', 'layer')
        self.dropout = config.get('dropout', 0.)

        # CNN layers
        assert len(self.num_channels) == len(self.kernel_sizes) == len(
            self.pool_sizes), \
            f'Number of channel / kernel / pool sizes mismatch: {len(self.num_channels)}, ' \
            f'{len(self.kernel_sizes)} and {len(self.pool_sizes)}'

        self.cnn_layers = [CNN1DBlock(
            n_channel=c,
            kernel_size=k,
            pool_size=p,
            norm_method=self.norm_method
        ) for c, k, p in zip(self.num_channels, self.kernel_sizes, self.pool_sizes)]

        # RNN layer
        self.recurrent_layers = [
            LSTM(unit, return_sequences=True) if i != len(self.rnn_units)-1 else LSTM(unit)
            for i, unit in enumerate(self.rnn_units)
        ]

        if self.bidirectional:
            self.recurrent_layers = [Bidirectional(layer) for layer in self.recurrent_layers]
        # logits layer
        self.logits = Dense(self.n_classes)

    def get_config(self):
        return {'num_channels': self.num_channels,
                'kernel_sizes': self.kernel_sizes,
                'pool_sizes': self.pool_sizes,
                'bidirectional': self.bidirectional,
                'rnn_units': self.rnn_units,
                'n_classes': self.n_classes,
                'dropout': self.dropout}

    def call(self, x, training=False, **kwargs) -> tf.Tensor:
        # cnn layers
        for cnn_block in self.cnn_layers:
            x = cnn_block(x, training=training)

        # masking
        x = Masking()(x)

        # lstm
        for rnn in self.recurrent_layers:
            x = rnn(x)

        # dense
        if training:
            x = tf.nn.dropout(x, rate=self.dropout)
        x = self.logits(x)
        return tf.nn.softmax(x, axis=-1)


class TestModel(BaseModel):

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.n_classes = config.get('n_classes', 4)
        self.logits = Dense(self.n_classes)

    def call(self, x, **kwargs):
        x = tf.reduce_mean(x, axis=1)
        x = self.logits(x)
        x = tf.nn.softmax(x, axis=-1)
        return x

    def get_config(self):
        super().get_config()
