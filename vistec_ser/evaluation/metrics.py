import tensorflow as tf


def weighted_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.int64)
    y_pred = tf.cast(y_pred, tf.int64)

    n_samples = tf.cast(tf.shape(y_true)[0], tf.int64)
    n_correct = tf.reduce_sum(tf.cast(tf.equal(y_true, y_pred), tf.int64))
    return n_correct / n_samples


def unweighted_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor, return_average: bool = True) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.int64)
    y_pred = tf.cast(y_pred, tf.int64)

    emotions, _ = tf.unique(y_true)
    n_emotions = tf.cast(tf.shape(emotions)[0], tf.float32)
    classes_acc = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    for i in tf.range(tf.shape(emotions)[0]):
        emo = emotions[i]
        emo_idx = get_emotion_indices(y_true, emo)
        emo_pred = tf.gather(y_pred, emo_idx)
        n_correct = tf.reduce_sum(tf.cast(tf.equal(emo_pred, emo), tf.int32))
        n_samples = tf.shape(emo_idx)[0]
        class_accuracy = tf.cast(n_correct / n_samples, tf.float32)
        classes_acc = classes_acc.write(classes_acc.size(), class_accuracy)
    if return_average:
        return tf.reduce_sum(classes_acc.stack()) / n_emotions
    return classes_acc.stack()


def get_emotion_indices(y: tf.Tensor, emotion_index: int) -> tf.Tensor:
    return tf.reshape(tf.where(tf.equal(y, emotion_index)), (-1,))


def compute_confusion_matrix(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return tf.math.confusion_matrix(y_true, y_pred)


def normalize_confusion_matrix(confusion_matrix: tf.Tensor, axis: int = 1) -> tf.Tensor:
    if axis == 0:
        return confusion_matrix / tf.reduce_sum(confusion_matrix, axis=0)
    elif axis == 1:
        cm_transpose = tf.transpose(confusion_matrix, (), (1, 0))
        norm_cm_transpose = cm_transpose / tf.reduce_sum(cm_transpose, axis=1)
        return tf.transpose(norm_cm_transpose, (1, 0))
    else:
        raise ValueError("Invalid `axis` argument. Only 0 or 1 available")
