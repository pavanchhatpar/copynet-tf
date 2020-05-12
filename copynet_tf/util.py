import tensorflow as tf


@tf.function
def masked_log_softmax(logits: tf.Tensor,
                       mask: tf.Tensor,
                       axis: int = None,
                       name: str = None) -> tf.Tensor:
    if mask is not None:
        # make masked logits highly negative
        logits = logits + tf.math.log(tf.cast(mask, tf.float32) + 1e-35)
    return tf.nn.log_softmax(logits, axis, name)


@tf.function
def masked_softmax(logits: tf.Tensor,
                   mask: tf.Tensor,
                   axis: int = None,
                   name: str = None) -> tf.Tensor:
    if mask is None:
        return tf.nn.softmax(logits, axis, name)
    else:
        # nullify contribution of masked logits
        mask = tf.cast(mask, tf.float32)
        inp = logits*mask
        result = tf.nn.softmax(inp, axis, name)
        result = result * mask
        # ensures addition to 1
        result /= (tf.reduce_sum(result, axis=axis, keepdims=True)+1e-35)
        return result


@tf.function
def prep_y_true(
        source_token_ids: tf.Tensor,
        target_token_ids: tf.Tensor,
        target2source_ids: tf.Tensor,
        target_vocab_size: int,
        unk_index: int,
        start_index: int,
        end_index: int) -> tf.Tensor:
    """
    Prepares y_true compatible to use in a `Metric` or `Loss` function
    Takes `target_token_ids` and fills in the `unk_index` values with the
    source position of the corresponding `target2source_ids`.

    If the source has multiple occurences of the corresponding
    `target2source_ids`, last occurence of the token is taken.
    """
    batch_size, source_seq_len = source_token_ids.shape
    _, target_seq_len = target_token_ids.shape

    source_spl_mask = ~(source_token_ids == unk_index)
    source_spl_mask &= ~(source_token_ids == start_index)
    source_spl_mask &= ~(source_token_ids == end_index)
    indices = tf.repeat(tf.reshape(
        tf.range(source_seq_len), (1, source_seq_len)), batch_size, axis=0)
    adjusted_indices = indices + target_vocab_size
    y_true = tf.TensorArray(tf.int32, size=target_seq_len-1)
    for timestep in tf.range(1, target_seq_len):
        next_predictions = target_token_ids[:, timestep]
        copy_candidate = (
            source_token_ids
            == tf.expand_dims(target2source_ids[:, timestep], 1))
        # shape: (batch, source_seq_len)
        copy_candidate = copy_candidate & source_spl_mask
        copied = next_predictions == unk_index
        # shape: (batch, 1)
        copied = tf.expand_dims(copied, 1)
        copy_candidate &= copied
        # (batch, )
        inter = tf.reduce_max(tf.where(
            copy_candidate, adjusted_indices, indices), -1)
        # (batch, )
        next_predictions = tf.where(
            inter >= target_vocab_size, inter, next_predictions)
        y_true = y_true.write(timestep-1, next_predictions)
    # (batch, target_seq_len-1)
    y_true = tf.transpose(y_true.stack())
    # tf.print(
    #         "ytrue", y_true[:3],
    #         output_stream="file:///tf/src/data/log1.txt", summarize=-1)
    return y_true
