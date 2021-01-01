from typing import List

from tensorflow.keras.layers import Bidirectional, LSTM, GRU, SimpleRNN, TimeDistributed, Dense, Layer
import tensorflow as tf


class BiPyramidRNN(Layer):
    def __init__(self,
                 units: int,
                 n_layers: int,
                 rnn_type: str = 'lstm'):
        super().__init__()
        if rnn_type == 'lstm':
            self.rnn_stack = [Bidirectional(LSTM(units, return_sequences=True)) for _ in range(n_layers)]
        elif rnn_type == 'gru':
            self.rnn_stack = [Bidirectional(GRU(units, return_sequences=True)) for _ in range(n_layers)]
        elif rnn_type == 'rnn':
            self.rnn_stack = [Bidirectional(SimpleRNN(units, return_sequences=True)) for _ in range(n_layers)]
        else:
            raise NameError('Unrecognized rnn_type: {}'.format(rnn_type))

    def call(self, x, **kwargs):
        i = 0
        for rnn in self.rnn_stack:
            if i == 0:
                x = rnn(x)
            else:
                if tf.shape(x)[1] % 2 == 1:
                    shape = tf.shape(x)
                    pad = tf.zeros([shape[0], 1, shape[-1]])
                    x = tf.concat([x, pad], axis=1)
                x = tf.concat([x[:, 0::2, :], x[:, 1::2, :]], axis=-1)
                x = rnn(x)
            i += 1
        return x


class AttentiveRNN(Layer):
    def __init__(self,
                 units: List[int] = [128],
                 recurrent_type: str = 'gru',
                 bidirectional: bool = True):
        super().__init__()
        if recurrent_type == 'gru':
            self.RNNs = [GRU(unit, return_sequences=True) for unit in units]
        elif recurrent_type == 'lstm':
            self.RNNs = [LSTM(unit, return_sequences=True) for unit in units]
        elif recurrent_type == 'rnn':
            self.RNNs = [SimpleRNN(unit, return_sequences=True) for unit in units]
        else:
            raise NameError('Unknown recurrent type: {}'.format(recurrent_type))

        if bidirectional:
            self.RNNs = [Bidirectional(layer) for layer in self.RNNs]

        self.f_attn = TimeDistributed(Dense(1, activation='tanh'))

    def call(self, x, training=False, return_attn=True, **kwargs):
        for rnn in self.RNNs:
            x = rnn(x)

        attn_score = tf.nn.softmax(self.f_attn(x), axis=1)
        x = tf.reduce_sum(x * attn_score, axis=1)
        if return_attn:
            return x, attn_score
        return x
