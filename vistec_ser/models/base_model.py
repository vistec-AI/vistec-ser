from abc import ABC
from typing import Dict, Any

import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.python.eager import backprop
from tensorflow.python.keras.engine import data_adapter
from tensorflow.keras.models import Model


def apply_specaugment(
        x: tf.Tensor,
        T: float = 30,
        F: float = 5,
        n_T: int = 2,
        n_F: int = 1,
        mask_value: float = 0.) -> tf.Tensor:
    for _ in range(n_F):
        x = tfio.experimental.audio.freq_mask(x, param=F)
    for _ in range(n_T):
        x = tfio.experimental.audio.time_mask(x, param=T)
    if mask_value:
        x = tf.where(tf.equal(x, 0), tf.ones_like(x) * mask_value, x)
    return x


class BaseModel(Model, ABC):

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        if config is None:
            config = {}
        self.specaugment_params = config.get('spec_augment', {})

    def train_step(self, data):
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        if len(self.specaugment_params) > 0:
            T = self.specaugment_params.get('T', 30)
            nT = self.specaugment_params.get('nT', 2)
            F = self.specaugment_params.get('F', 5)
            nF = self.specaugment_params.get('nF', 1)
            specaugment_fn = lambda sample: apply_specaugment(sample, T=T, n_T=nT, F=F, n_F=nF)
            x = tf.map_fn(specaugment_fn, x)

        with backprop.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}
