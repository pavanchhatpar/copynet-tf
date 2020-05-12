import tensorflow as tf
from tensorflow.keras.losses import Loss


class CopyNetLoss(Loss):
    def __init__(self, name=None):
        super(CopyNetLoss, self).__init__(
            reduction=tf.keras.losses.Reduction.AUTO, name=name)

    @tf.function
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        gathered = tf.gather(y_pred, y_true, axis=-1, batch_dims=2)
        # tf.print(
        #     "gathered", gathered[:3],
        #     output_stream="file:///tf/src/data/log1.txt", summarize=-1)
        return -gathered
