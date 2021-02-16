import tensorflow as tf
from tensorflow.keras import backend as K


def weighted_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    if len(tf.shape(y_true)) > 1:
        y_true = tf.argmax(y_true, axis=-1)
    if len(tf.shape(y_pred)) > 1:
        y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    correct_samples = K.cast(K.equal(y_true, y_pred), tf.float32)
    return K.mean(correct_samples)


def unweighted_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor, n_classes: int = 4) -> tf.Tensor:
    cm = compute_confusion_matrix(y_true, y_pred)
    cm = normalize_confusion_matrix(cm)
    return tf.reduce_mean(tf.linalg.diag_part(cm))


def get_emotion_indices(y: tf.Tensor, emotion_index: int) -> tf.Tensor:
    return tf.reshape(tf.where(tf.equal(y, emotion_index)), (-1,))


def compute_confusion_matrix(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    if len(tf.shape(y_true)) > 1:
        y_true = tf.argmax(y_true, axis=-1)
    if len(tf.shape(y_pred)) > 1:
        y_pred = tf.argmax(y_pred, axis=-1)
    return tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32)


def normalize_confusion_matrix(confusion_matrix: tf.Tensor, axis: int = 1) -> tf.Tensor:
    if axis == 0:
        return confusion_matrix / tf.reduce_sum(confusion_matrix, axis=0)
    elif axis == 1:
        cm_transpose = tf.transpose(confusion_matrix, (1, 0))
        norm_cm_transpose = cm_transpose / tf.reduce_sum(confusion_matrix, axis=1)
        return tf.transpose(norm_cm_transpose, (1, 0))
    else:
        raise ValueError("Invalid `axis` argument. Only 0 or 1 available")
