from tensorflow.keras.layers import Layer
import tensorflow as tf


class BahdanauAttention(Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def _verify_shapes(self, query, values, values_mask):
        qbatch_size, qhidden_size = query.shape
        vbatch_size, vseq_len, vhidden_size = values.shape
        vmbatch_size, vmseq_len = (
            values_mask.shape if values_mask is not None else values.shape[:2])
        if not qbatch_size == vbatch_size or not vbatch_size == vmbatch_size:
            raise ValueError("Batch sizes should be same everywhere")
        if qhidden_size != vhidden_size:
            raise ValueError("Hidden dim of query and values should match")
        if vseq_len != vmseq_len:
            raise ValueError(
                "Sequence length of values and values_mask should match")

    def call(self, query, values, values_mask=None):
        self._verify_shapes(query, values, values_mask)
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to
        # calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is
        # (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # Make mask scores values highly neg
        if values_mask is not None:
            values_mask = tf.expand_dims(tf.logical_not(values_mask), -1)
            score -= 1.e9 * tf.cast(values_mask, dtype=score.dtype)

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
