import tensorflow as tf
from tensorflow.keras import backend as K


def weighted_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)

    correct_samples = K.cast(K.equal(y_true, y_pred), tf.float32)
    return K.mean(correct_samples)


def unweighted_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor, n_classes: int = 4) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)
    class_accuracies = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for i in tf.range(0, n_classes, dtype=tf.float32):
        class_idx = tf.reshape(tf.where(tf.equal(y_true, i)), (-1,))
        class_pred = tf.gather(y_pred, class_idx)
        n_correct = tf.cast(tf.equal(class_pred, i), tf.float32)
        class_acc = tf.reduce_mean(n_correct)
        class_accuracies = class_accuracies.write(class_accuracies.size(), class_acc)
    ua = tf.reduce_mean(class_accuracies.stack())
    return ua


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
        cm_transpose = tf.transpose(confusion_matrix, (), (1, 0))
        norm_cm_transpose = cm_transpose / tf.reduce_sum(cm_transpose, axis=1)
        return tf.transpose(norm_cm_transpose, (1, 0))
    else:
        raise ValueError("Invalid `axis` argument. Only 0 or 1 available")
