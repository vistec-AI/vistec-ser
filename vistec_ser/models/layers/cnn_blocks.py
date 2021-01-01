import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPool1D, Conv2D, MaxPool2D, BatchNormalization, LayerNormalization, Layer


class CNN1DBlock(Layer):
    def __init__(self,
                 n_channel: int,
                 kernel_size: int,
                 pool_size: int,
                 stride: int = 1,
                 padding: str = 'same',
                 norm_method: str = 'layer'):
        super().__init__()
        self.cnn = Conv1D(n_channel, kernel_size, strides=stride, padding=padding)
        self.pool = MaxPool1D(pool_size)
        if norm_method:
            if norm_method == 'layer':
                self.norm = LayerNormalization()
            elif norm_method == 'batch':
                self.norm = BatchNormalization()
            else:
                raise ValueError('Unrecognized normaliziation method: {}'.format(norm_method))
        else:
            self.norm = None

    def call(self, x, training=False):
        x = self.cnn(x)
        if self.norm:
            x = self.norm(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.1)
        x = self.pool(x)
        return x
