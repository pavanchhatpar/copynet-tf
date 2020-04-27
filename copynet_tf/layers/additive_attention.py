from tensorflow.keras.layers import AdditiveAttention as KAttn

from tensorflow.python.keras import backend as K
import tensorflow as tf


class AdditiveAttention(KAttn):
    def __init__(self, use_scale=True, **kwargs):
        super(AdditiveAttention, self).__init__(use_scale, **kwargs)

    def _apply_scores(self, scores, value, scores_mask=None):
        """Applies attention scores to the given value tensor.
        To use this method in your attention layer, follow the steps:
        * Use `query` tensor of shape `[batch_size, Tq]` and `key` tensor of
        shape `[batch_size, Tv]` to calculate the attention `scores`.
        * Pass `scores` and `value` tensors to this method. The method applies
        `scores_mask`, calculates `attention_distribution = softmax(scores)`,
        then returns `matmul(attention_distribution, value).
        * Apply `query_mask` and return the result.
        Args:
        scores: Scores float tensor of shape `[batch_size, Tq, Tv]`.
        value: Value tensor of shape `[batch_size, Tv, dim]`.
        scores_mask: A boolean mask `Tensor` of shape `[batch_size, 1, Tv]` or
            `[batch_size, Tq, Tv]`. If given, scores at positions where
            `scores_mask==False` do not contribute to the result. It must
            contain at least one `True` value in each line along the last
            dimension.
        Returns:
        Tensor of shape `[batch_size, Tq, dim]`,
        Tensor of shape `[batch_size, Tq, Tv]`.
        """
        if scores_mask is not None:
            padding_mask = tf.logical_not(scores_mask)
            # Bias so padding positions do not contribute to attention
            # distribution.
            scores -= 1.e9 * tf.cast(padding_mask, dtype=K.floatx())
        attention_distribution = tf.nn.softmax(scores)
        return (tf.matmul(attention_distribution, value),
                attention_distribution)

    # TODO(b/125916026): Consider exposing a __call__ method with named args.
    def call(self, inputs, mask=None):
        self._validate_call_args(inputs=inputs, mask=mask)
        q = inputs[0]
        v = inputs[1]
        k = inputs[2] if len(inputs) > 2 else v
        q_mask = mask[0] if mask else None
        v_mask = mask[1] if mask else None
        scores = self._calculate_scores(query=q, key=k)
        if v_mask is not None:
            # Mask of shape [batch_size, 1, Tv].
            v_mask = tf.expand_dims(v_mask, axis=-2)
        if self.causal:
            # Creates a lower triangular mask, so position i cannot attend to
            # positions j>i. This prevents the flow of information from the
            # future into the past.
            scores_shape = tf.shape(scores)
            # causal_mask_shape = [1, Tq, Tv].
            causal_mask_shape = tf.concat(
                [tf.ones_like(scores_shape[:-2]), scores_shape[-2:]],
                axis=0)
            causal_mask = _lower_triangular_mask(causal_mask_shape)
        else:
            causal_mask = None
        scores_mask = _merge_masks(v_mask, causal_mask)
        result, attn_weights = self._apply_scores(
            scores=scores, value=v, scores_mask=scores_mask)
        if q_mask is not None:
            # Mask of shape [batch_size, Tq, 1].
            q_mask = tf.expand_dims(q_mask, axis=-1)
            result *= tf.cast(q_mask, dtype=result.dtype)
        return result, attn_weights


def _lower_triangular_mask(shape):
    """Creates a lower-triangular boolean mask over the last 2 dimensions."""
    row_index = tf.cumsum(
        tf.ones(shape=shape, dtype=tf.dtypes.int32), axis=-2)
    col_index = tf.cumsum(
        tf.ones(shape=shape, dtype=tf.dtypes.int32), axis=-1)
    return tf.greater_equal(row_index, col_index)


def _merge_masks(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return tf.logical_and(x, y)
