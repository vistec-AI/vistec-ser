from typing import Union, List, Tuple

import tensorflow as tf

from .metrics import weighted_accuracy, unweighted_accuracy, compute_confusion_matrix
from ..datasets.features.preprocessing import chop_feature
from ..models.base_model import BaseModel


def evaluate_slice_model(
        model: BaseModel,
        X_test: Union[List[tf.Tensor], tf.Tensor],
        y_test: tf.Tensor,
        mode: str = 'chunk', **kwargs) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    n_samples = len(X_test)
    y_pred = tf.TensorArray(tf.int64, size=0, dynamic_size=True)
    for i in tf.range(n_samples):
        x_test = X_test[i]
        if mode == "chunk":
            y_pred = y_pred.write(y_pred.size(), evaluate_chunk(model, x_test), **kwargs)
        else:
            y_pred = y_pred.write(y_pred.size(), evaluate_full(model, x_test))
    y_pred = y_pred.stack()
    wa = weighted_accuracy(y_test, y_pred)
    ua = unweighted_accuracy(y_test, y_pred)
    cm = compute_confusion_matrix(y_test, y_pred)
    return wa, ua, cm


def evaluate_testloader(
        model: BaseModel,
        test_dataset: tf.data.Dataset,
        n_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    test_iter = iter(test_dataset)
    y_pred, y_true = [], []
    for i in range(n_steps):
        x, y = next(test_iter)
        y_hat = tf.argmax(model(x), axis=-1)
        y_true.append(y)
        y_pred.append(y_hat)
    y_true = tf.concat(y_true, axis=0)
    y_pred = tf.concat(y_pred, axis=0)
    wa = weighted_accuracy(y_true, y_pred)
    ua = unweighted_accuracy(y_true, y_pred)
    cm = compute_confusion_matrix(y_true, y_pred)
    return wa, ua, cm


def evaluate_chunk(model: BaseModel, x_test: tf.Tensor, n_frames: int = 300) -> tf.Tensor:
    chunk_test = chop_feature(x_test, n_frames=n_frames)
    y_pred = tf.argmax(tf.reduce_mean(model(chunk_test, training=False), axis=0), axis=-1)
    return y_pred


def evaluate_full(model: BaseModel, x_test: tf.Tensor) -> tf.Tensor:
    return tf.argmax(model(tf.expand_dims(x_test, 0), training=False))
