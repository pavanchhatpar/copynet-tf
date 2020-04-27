import tensorflow as tf


def masked_log_softmax(logits: tf.Tensor,
                       mask: tf.Tensor,
                       axis: int = None,
                       name: str = None):
    if mask is not None:
        # make masked logits highly negative
        logits = logits + tf.math.log(tf.cast(mask, tf.float32) + 1e-45)
    return tf.nn.log_softmax(logits, axis, name)


def masked_softmax(logits: tf.Tensor,
                   mask: tf.Tensor,
                   axis: int = None,
                   name: str = None):
    if mask is None:
        return tf.nn.softmax(logits, axis, name)
    else:
        # nullify contribution of masked logits
        mask = tf.cast(mask, tf.float32)
        result = tf.nn.softmax(logits*mask, axis, name)
        result = result * mask
        # ensures addition to 1
        result /= (tf.reduce_sum(result, axis=axis, keepdims=True) + 1e-45)
        return result
