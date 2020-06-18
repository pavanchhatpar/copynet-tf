import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense

from ..vocab import Vocab
from ..util import masked_log_softmax
from ..types import State, StrDict


class CopyHead(Layer):
    def __init__(self,
                 vocab: Vocab,
                 encoder_out_dim: int,
                 output_layer: Layer,
                 **kwargs: StrDict) -> None:
        super(CopyHead, self).__init__(**kwargs)
        self.vocab = vocab
        self._unk_index = self.vocab.get_token_id(
            self.vocab._unk_token, "target")
        self._start_index = self.vocab.get_token_id(
            self.vocab._start_token, "target")
        self._end_index = self.vocab.get_token_id(
            self.vocab._end_token, "target")
        self._pad_index = self.vocab.get_token_id(
            self.vocab._pad_token, "target")
        self.encoder_out_dim = encoder_out_dim
        self.decoder_out_dim = self.encoder_out_dim
        self.decoder_inp_dim = self.decoder_out_dim
        self._build_decoder_layers(output_layer)

    def _build_decoder_layers(self,
                              output_layer: Layer) -> None:
        self._target_vocab_size = self.vocab.get_vocab_size("target")
        self.output_generation_layer = output_layer
        self.output_copying_layer = Dense(
            self.decoder_out_dim, activation="tanh")

    @tf.function
    def call(self,
             source_token_ids: tf.Tensor,
             source2target_ids: tf.Tensor,
             source_mask: tf.Tensor,
             encoder_output: tf.Tensor,
             decoder_hidden: tf.Tensor) -> State:
        """
        Parameters
        ----------

        source_token_ids: tf.Tensor
            shape (batch_size, source_seq_len)
        source2target_ids: tf.Tensor
            shape (batch_size, source_seq_len)
        source_mask: tf.Tensor
            shape (batch_size, source_seq_len)
        encoder_output: tf.Tensor
            shape (batch_size, source_seq_len, encoder_out_dim)
        decoder_hidden: tf.Tensor
            shape (batch_size, decoder_out_dim)
        """
        state = {
            "encoder_output": encoder_output,
            "source_mask": source_mask,
            "source_token_ids": source_token_ids,
            "source2target_ids": source2target_ids,
            "decoder_hidden": decoder_hidden
        }

        _, source_seq_len = state["source_mask"].shape

        # Get the un-normalized generation scores for each token in the target
        # vocab.
        # shape: (batch_size, target_vocab_size)
        generation_scores = self._get_generation_scores(state)
        # Get the un-normalized copy scores for each token in the source
        # sentence, excluding the start and end tokens.
        # shape: (batch_size, source_seq_len)
        copy_scores = self._get_copy_score(state)
        # Concat un-normalized generation and copy scores.
        # shape: (batch_size, target_vocab_size + source_seq_len)
        all_scores = tf.concat((generation_scores, copy_scores), -1)
        # shape: (batch_size, source_seq_len)
        copy_mask = tf.cast(state["source_mask"], tf.float32)
        # shape: (batch_size, target_vocab_size + source_seq_len)
        mask = tf.concat(
            [
                tf.fill(
                    generation_scores.shape, tf.constant(1, dtype=tf.float32)),
                copy_mask,
            ],
            axis=-1)
        # Normalize generation and copy scores.
        # shape: (batch_size, target_vocab_size + source_seq_len)
        log_probs = masked_log_softmax(all_scores, mask, axis=1)
        # shape: (batch_size, target_vocab_size), (batch_size, source_seq_len)
        generation_log_probs, copy_log_probs = tf.split(
            log_probs, [self._target_vocab_size, source_seq_len], axis=-1)

        # We now have normalized generation and copy scores, but to produce the
        # final score for each token in the extended vocab, we have to go
        # through and add the copy scores to the generation scores of matching
        # target tokens, and sum the copy scores of duplicate source tokens.
        # shape: (batch_size, target_vocab_size + source_seq_len)
        final_log_probs = self._gather_final_log_probs(
            generation_log_probs, copy_log_probs, state
        )
        return final_log_probs, generation_log_probs, copy_log_probs

    @tf.function
    def _gather_final_log_probs(
        self,
        generation_log_probs: tf.Tensor,
        copy_log_probs: tf.Tensor,
        state: State,
    ) -> tf.Tensor:
        _, source_seq_len = state["source_mask"].shape
        source_token_ids = state["source_token_ids"]
        # tf.print(
        #         "source tokens", source_token_ids[:3],
        #         output_stream="file:///tf/src/data/log1.txt", summarize=-1)
        # tf.print(
        #         "source target tokens", state["source2target_ids"][:3],
        #         output_stream="file:///tf/src/data/log1.txt", summarize=-1)
        # tf.print(
        #         "gen prob", generation_log_probs[:3],
        #         output_stream="file:///tf/src/data/log1.txt", summarize=-1)
        # tf.print(
        #         "copy prob", copy_log_probs[:3],
        #         output_stream="file:///tf/src/data/log1.txt", summarize=-1)
        # shape: [(batch_size, *)]
        modified_log_probs_list = []
        for i in range(source_seq_len):
            # shape: (batch_size,)
            copy_log_probs_slice = copy_log_probs[:, i]
            # `source2target_ids` is a matrix of shape (batch_size,
            # source_seq_len) where element (i, j) is the vocab index of the
            # target token that matches the jth source token in the ith group,
            # if there is one, or the index of the OOV symbol otherwise. We'll
            # use this to add copy scores to corresponding generation scores.
            # shape: (batch_size,)
            source2target_ids_slice = state["source2target_ids"][:, i]
            # The OOV index in the source2target_ids_slice indicates that the
            # source token is not in the target vocab, so we don't want to add
            # that copy score to the OOV token.
            copy_log_probs_to_add_mask = (
                source2target_ids_slice != self._unk_index)
            copy_log_probs_to_add_mask &= (
                source2target_ids_slice != self._start_index)
            copy_log_probs_to_add_mask &= (
                source2target_ids_slice != self._end_index)
            copy_log_probs_to_add_mask &= (
                source2target_ids_slice != self._pad_index)
            copy_log_probs_to_add_mask = tf.cast(
                copy_log_probs_to_add_mask, tf.float32)
            copy_log_probs_to_add = (
                copy_log_probs_slice
                + tf.math.log(copy_log_probs_to_add_mask + 1e-35)
            )
            # shape: (batch_size, 1)
            copy_log_probs_to_add = tf.expand_dims(copy_log_probs_to_add, -1)
            # shape: (batch_size, 1)
            selected_generation_log_probs = tf.gather(
                generation_log_probs,
                tf.expand_dims(source2target_ids_slice, -1),
                axis=1, batch_dims=1)

            # shape: (batch_size,)
            combined_scores = tf.math.reduce_logsumexp(
                tf.concat([
                    selected_generation_log_probs, copy_log_probs_to_add],
                    axis=1),
                axis=-1
            )
            indices = tf.expand_dims(tf.range(
                source2target_ids_slice.shape[0],
                dtype=source2target_ids_slice.dtype), -1)
            indices = tf.concat(
                [indices, tf.expand_dims(source2target_ids_slice, -1)], -1)
            generation_log_probs = tf.tensor_scatter_nd_update(
                generation_log_probs, indices, combined_scores)
            # We have to combine copy scores for duplicate source tokens so
            # that we can find the overall most likely source token. So, if
            # this is the last occurence of this particular source token, we
            # add the log_probs from all other occurences, otherwise we zero it
            # out since it will be accounted for later.
            if i > 0:
                # Sum copy scores from past occurences of source token.
                # shape: (batch_size, i)
                source_past_occurences = tf.cast(
                    source_token_ids[:, :i]
                    == tf.expand_dims(source_token_ids[:, i], -1), tf.float32)
                # shape: (batch_size, i)
                past_copy_log_probs = (
                    copy_log_probs[:, :i]
                    + tf.math.log(source_past_occurences + 1e-35)
                )
                # shape: (batch_size, 1 + i)
                combined = tf.concat(
                    [tf.expand_dims(copy_log_probs_slice, -1),
                     past_copy_log_probs], axis=-1
                )
                # shape: (batch_size,)
                copy_log_probs_slice = tf.math.reduce_logsumexp(
                    combined, axis=-1)
            if i < (source_seq_len - 1):
                # Remove copy log_probs that will be accounted for later.
                # shape: (batch_size, source_seq_len - i)
                source_future_occurences = tf.cast(source_token_ids[
                    :, (i+1):
                ] == tf.expand_dims(source_token_ids[:, i], -1), tf.int32)
                # shape: (batch_size,)
                duplicate_mask = tf.cast(tf.reduce_sum(
                    source_future_occurences, axis=-1) == 0, tf.float32)
                copy_log_probs_slice = (
                    copy_log_probs_slice + tf.math.log(duplicate_mask + 1e-35)
                )
            # Finally, we zero-out copy scores that we added to the generation
            # scores above so that we don't double-count them.
            # shape: (batch_size,)
            copy_log_probs_not_add_mask = ~tf.cast(
                copy_log_probs_to_add_mask, bool)
            copy_log_probs_not_add_mask = tf.cast(
                copy_log_probs_not_add_mask, tf.float32)
            left_over_copy_log_probs = (
                copy_log_probs_slice
                + tf.math.log(copy_log_probs_not_add_mask + 1e-35)
            )
            # tf.print(
            #     "leftover copy slice", left_over_copy_log_probs[:3],
            #     output_stream="file:///tf/src/data/log1.txt", summarize=-1)
            modified_log_probs_list.append(
                tf.expand_dims(left_over_copy_log_probs, -1))
        modified_log_probs_list.insert(0, generation_log_probs)

        # shape: (batch_size, target_vocab_size + source_seq_len)
        modified_log_probs = tf.concat(modified_log_probs_list, axis=-1)
        # tf.py_function(
        #     self.debug, ['modified_log_probas', tf.reduce_logsumexp(
        #         modified_log_probs[:3], axis=-1)], [])

        return modified_log_probs

    @tf.function
    def _get_generation_scores(
            self, state: State) -> tf.Tensor:
        return self.output_generation_layer(state["decoder_hidden"])

    @tf.function
    def _get_copy_score(self, state: State) -> tf.Tensor:
        # shape: (batch, source_seq_len, encoder_out_dim)
        encoder_output = state["encoder_output"]
        # shape: (batch, source_seq_len, decoder_out_dim)
        copy_projection = self.output_copying_layer(encoder_output)

        # shape: (batch, decoder_out_dim, 1)
        decoder_hidden = tf.expand_dims(state["decoder_hidden"], -1)

        # shape: (batch, source_seq_len, 1)
        copy_scores = tf.matmul(copy_projection, decoder_hidden)

        # shape: (batch, source_seq_len)
        copy_scores = tf.squeeze(copy_scores, axis=-1)

        return copy_scores
