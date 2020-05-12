import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, GRU
from typing import Tuple
import logging

from .layers import FixedEmbedding, BahdanauAttention
from .vocab import Vocab
from .search import Searcher
from .util import masked_log_softmax, masked_softmax
from .types import State, StrDict


class Decoder(Layer):
    def __init__(self,
                 vocab: Vocab,
                 encoder_out_dim: int,
                 searcher: Searcher,
                 output_layer: Layer,
                 copy_token: str = "@COPY@",
                 **kwargs: StrDict) -> None:
        super(Decoder, self).__init__(**kwargs)
        self.vocab = vocab
        self.logger = logging.getLogger(__name__)
        self._copy_index = self.vocab.get_token_id(
            copy_token, "target")
        self._unk_index = self.vocab.get_token_id(
            self.vocab._unk_token, "target")
        self._start_index = self.vocab.get_token_id(
            self.vocab._start_token, "target")
        self._end_index = self.vocab.get_token_id(
            self.vocab._end_token, "target")
        self._pad_index = self.vocab.get_token_id(
            self.vocab._pad_token, "target")
        self._srcunk_index = self.vocab.get_token_id(
            self.vocab._unk_token, "source")
        self._srcstart_index = self.vocab.get_token_id(
            self.vocab._start_token, "source")
        self._srcend_index = self.vocab.get_token_id(
            self.vocab._end_token, "source")
        self._srcpad_index = self.vocab.get_token_id(
            self.vocab._pad_token, "source")
        self._searcher = searcher
        self.encoder_out_dim = encoder_out_dim
        self.decoder_out_dim = self.encoder_out_dim
        self.decoder_inp_dim = self.decoder_out_dim
        self._build_decoder_layers(output_layer)

    def _build_decoder_layers(self,
                              output_layer: Layer) -> None:
        embedding_matrix = self.vocab.get_embedding_matrix("target")
        self.target_embedder = FixedEmbedding(
            embedding_matrix, self.vocab.get_sequence_len("target"))
        self.decode_attn = BahdanauAttention(self.decoder_out_dim)
        self.decoder_input_projector = Dense(self.decoder_inp_dim)
        self.gru = GRU(self.decoder_out_dim)
        self._target_vocab_size = self.vocab.get_vocab_size("target")
        self.output_generation_layer = output_layer
        self.output_copying_layer = Dense(
            self.decoder_out_dim, activation="tanh")

    @tf.function
    def call(self,
             source_token_ids: tf.Tensor,
             source2target_ids: tf.Tensor,
             source_mask: tf.Tensor,
             state_og: State,
             target_token_ids: tf.Tensor = None,
             target2source_ids: tf.Tensor = None,
             training: bool = False) -> State:
        state = state_og.copy()
        state["source_mask"] = source_mask
        state["source_token_ids"] = source_token_ids
        state["source2target_ids"] = source2target_ids
        state["decoder_hidden"] = state["encoder_final_output"]

        output_dict = {}
        if training:
            output_dict = self._teacher_force(
                state, target_token_ids, target2source_ids)
        else:
            output_dict = self._decode_output(state)
        return output_dict

    @tf.function
    def _decode_output(self, state_og: State) -> State:
        state = state_og.copy()
        batch_size, source_length = state["source_mask"].shape
        # shape: (batch_size, source_length)
        state["copy_log_probs"] = tf.math.log(
            (tf.zeros((batch_size, source_length), dtype=tf.float32) + 1e-35))
        start_predictions = tf.fill(
            (batch_size,), tf.constant(self._start_index, dtype=tf.int32))

        # shape: (batch_size, beam_width, max_decoding_steps)
        # shape: (batch_size, beam_width)
        all_top_k_predictions, log_probabilities = self._searcher.search(
            start_predictions, state, self._search_step)

        return {
            "predicted_probas": log_probabilities,
            "predictions": all_top_k_predictions,
            "ypred": 1.0,
            "attentive_weights": 1.0,
            "selective_weights": 1.0,
        }

    @tf.function
    def _teacher_force(self,
                       state_og: State,
                       target_token_ids: tf.Tensor,
                       target2source_ids: tf.Tensor) -> State:
        state = state_og.copy()
        batch_size, source_seq_len = state["source_mask"].shape
        _, target_length = target_token_ids.shape
        target_vocab_size = self.vocab.get_vocab_size("target")
        # shape: (batch_size, source_seq_len)
        state["copy_log_probs"] = tf.math.log(
            (tf.zeros(
                (batch_size, source_seq_len), dtype=tf.float32) + 1e-35))
        log_probas = tf.TensorArray(tf.float32, size=target_length-1)
        attn_weights = tf.TensorArray(tf.float32, size=target_length-1)
        selec_weights = tf.TensorArray(tf.float32, size=target_length-1)
        source_spl_mask = ~(state["source_token_ids"] == self._unk_index)
        source_spl_mask &= ~(state["source_token_ids"] == self._start_index)
        source_spl_mask &= ~(state["source_token_ids"] == self._end_index)
        indices = tf.repeat(tf.reshape(
            tf.range(source_seq_len), (1, source_seq_len)), batch_size, axis=0)
        adjusted_indices = indices + target_vocab_size
        for timestep in tf.range(target_length - 1):
            # shape: (batch_size, )
            last_predictions = target_token_ids[:, timestep]
            copy_candidate = (
                state["source_token_ids"]
                == tf.expand_dims(target2source_ids[:, timestep], 1))
            # shape: (batch, source_seq_len)
            copy_candidate = copy_candidate & source_spl_mask
            copied = last_predictions == self._unk_index
            # shape: (batch, 1)
            copied = tf.expand_dims(copied, 1)
            copy_candidate &= copied
            # (batch, )
            inter = tf.reduce_max(tf.where(
                copy_candidate, adjusted_indices, indices), -1)
            # (batch, )
            last_predictions = tf.where(
                inter >= target_vocab_size, inter, last_predictions)
            # tf.print(
            #     "last preds", last_predictions[:3],
            #     output_stream="file:///tf/src/data/log1.txt", summarize=-1)
            final_log_probs, state = self._take_step(last_predictions, state)
            log_probas = log_probas.write(timestep, final_log_probs)
            selec_weights = selec_weights.write(
                timestep, state["selective_weights"])
            attn_weights = attn_weights.write(
                timestep, state["attentive_weights"])
            del state["attentive_weights"]
            del state["selective_weights"]
        # shape (batch_size, target_length-1, target_vocab_size+source_seq_len)
        all_log_probas = tf.reshape(
            tf.transpose(log_probas.stack(), perm=[1, 0, 2]),
            (batch_size, target_length-1, -1))
        # shape (batch_size, target_length-1, source_seq_len)
        all_attentive_weights = tf.reshape(
            tf.transpose(attn_weights.stack(), perm=[1, 0, 2]),
            (batch_size, target_length-1, -1))
        # shape (batch_size, target_length-1, source_seq_len)
        all_selective_weights = tf.reshape(
            tf.transpose(selec_weights.stack(), perm=[1, 0, 2]),
            (batch_size, target_length-1, -1))
        # tf.print(
        #     "Source token id", state["source_token_ids"][:3],
        #     output_stream="file:///tf/src/data/log1.txt", summarize=-1)
        # tf.print(
        #         "Full proba", all_log_probas[:3],
        #         output_stream="file:///tf/src/data/log1.txt", summarize=-1)
        return {
            "ypred": all_log_probas,
            "attentive_weights": all_attentive_weights,
            "selective_weights": all_selective_weights,
            "predicted_probas": 1.0,
            "predictions": 1,
        }

    @tf.function
    def _search_step(self,
                     last_predictions: tf.Tensor,
                     state: State) -> Tuple[
                              tf.Tensor, State]:
        final_log_probs, state = self._take_step(last_predictions, state)
        del state["attentive_weights"]
        del state["selective_weights"]
        return final_log_probs, state

    @tf.function
    def _take_step(self,
                   last_predictions: tf.Tensor,
                   state: State):
        _, source_seq_len = state["source_mask"].shape

        # Get input to the decoder RNN and the selective weights.
        # `input_choices` is the result of replacing target OOV tokens in
        # `last_predictions` with the copy symbol. `selective_weights` consist
        # of the normalized copy probabilities assigned to the source tokens
        # that were copied. If no tokens were copied, there will be all zeros.
        # shape: (batch_size,), (batch_size, source_seq_len)
        input_choices, selective_weights = (
            self._get_input_and_selective_weights(last_predictions, state))
        # tf.print(
        #         "input choices", input_choices[:3],
        #         output_stream="file:///tf/src/data/log1.txt", summarize=-1)
        # tf.print(
        #         "selective weights", selective_weights[:3],
        #         output_stream="file:///tf/src/data/log1.txt", summarize=-1)

        # Update the decoder state by taking a step through the RNN.
        state = self._decoder_step(input_choices, selective_weights, state)

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
        # Update copy_probs needed for getting the `selective_weights` at the
        # next timestep.
        state["copy_log_probs"] = copy_log_probs

        # We now have normalized generation and copy scores, but to produce the
        # final score for each token in the extended vocab, we have to go
        # through and add the copy scores to the generation scores of matching
        # target tokens, and sum the copy scores of duplicate source tokens.
        # shape: (batch_size, target_vocab_size + source_seq_len)
        final_log_probs = self._gather_final_log_probs(
            generation_log_probs, copy_log_probs, state
        )
        return final_log_probs, state

    @tf.function
    def _get_input_and_selective_weights(
        self, last_predictions: tf.Tensor, state: State
    ) -> Tuple[tf.Tensor, tf.Tensor]:

        batch_size, source_seq_len = state["source_mask"].shape

        # This is a mask indicating which last predictions were copied from the
        # the source AND not in the target vocabulary (OOV).
        # (batch_size,)
        only_copied_mask = tf.cast(
            last_predictions >= self._target_vocab_size, tf.int32)

        # If the last prediction was in the target vocab or OOV but not copied,
        # we use that as input, otherwise we use the COPY token.
        # shape: (batch_size,)
        copy_input_choices = tf.fill(
            (batch_size,), value=tf.constant(self._copy_index, dtype=tf.int32))

        input_choices = (
            last_predictions * (1 - only_copied_mask)
            + copy_input_choices * only_copied_mask
        )

        # In order to get the `selective_weights`, we need to find out which
        # predictions were copied or copied AND generated, which is the case
        # when a prediction appears in both the source sentence and the target
        # vocab. But whenever a prediction is in the target vocab (even if it
        # also appeared in the source sentence), its index will be the
        # corresponding target vocab index, not its index in the source
        # sentence offset by the target vocab size. So we first use
        # `state["source2target_ids"]` to get an indicator of every source
        # token that matches the predicted target token.
        # shape: (batch_size, source_seq_len)
        expanded_last_predictions = tf.repeat(
            tf.expand_dims(last_predictions, -1), source_seq_len, axis=-1)
        # shape: (batch_size, source_seq_len)
        source_copied_and_generated = (
            state["source2target_ids"] == expanded_last_predictions
        )

        # In order to get indicators for copied source tokens that are OOV with
        # respect to the target vocab, we'll make use of
        # `state["source_token_ids"]`. First we adjust predictions relative to
        # the start of the source tokens. This makes sense because predictions
        # for copied tokens are given by the index of the copied token in the
        # source sentence, offset by the size of the target vocabulary.
        # shape: (batch_size,)
        adjusted_predictions = last_predictions - self._target_vocab_size
        # The adjusted indices for items that were not copied will be negative
        # numbers, and therefore invalid. So we zero them out.
        adjusted_predictions = adjusted_predictions * only_copied_mask
        # shape: (batch_size, source_seq_len)
        source_token_ids = state["source_token_ids"]
        # shape: (batch, source_seq_len)
        source_spl_mask = ~(state["source_token_ids"] == self._unk_index)
        source_spl_mask &= ~(state["source_token_ids"] == self._start_index)
        source_spl_mask &= ~(state["source_token_ids"] == self._end_index)
        # shape: (batch_size, 1)
        adjusted_prediction_ids = tf.gather(
            source_token_ids,
            tf.expand_dims(adjusted_predictions, -1), axis=-1, batch_dims=1)
        # This mask will contain indicators for source tokens that were copied
        # during the last timestep.
        # shape: (batch_size, source_seq_len)
        source_only_copied = source_token_ids == adjusted_prediction_ids
        # Since we zero'd-out indices for predictions that were not copied,
        # we need to zero out all entries of this mask corresponding to those
        # predictions.
        source_only_copied = (
            source_only_copied
            & tf.cast(tf.expand_dims(only_copied_mask, -1), bool))

        # source_copied_and_generated &= source_spl_mask

        # shape: (batch_size, source_seq_len)
        mask = source_only_copied | source_copied_and_generated

        mask &= source_spl_mask
        # shape: (batch_size, source_seq_len)
        selective_weights = masked_softmax(
            state["copy_log_probs"], mask, axis=1)

        return input_choices, selective_weights

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
    def _decoder_step(self,
                      last_predictions: tf.Tensor,
                      selective_weights: tf.Tensor,
                      state_og: State) -> State:
        """
        # Parameters

        last_predictions : `tf.Tensor`
            Shape : (batch,)
        selective_weights : `tf.Tensor`
            Shape : (batch, source_seq_len)
        state : `State`
        """
        state = state_og.copy()
        # (batch, source_seq_len)
        source_mask = state["source_mask"]

        # shape: (batch, 1)
        last_predictions = tf.expand_dims(last_predictions, 1)
        # shape: (batch, 1, target_emb_dim)
        target_embedded = self.target_embedder(last_predictions)
        # shape: (batch, decoder_out_dim)
        query = state["decoder_hidden"]
        # shape: (batch, decoder_out_dim), (batch, source_seq_len, 1)
        attentive_read, attentive_weights = self.decode_attn(
            query, state["encoder_output"], source_mask)

        # shape: (batch, 1, decoder_out_dim)
        attentive_read = tf.expand_dims(attentive_read, 1)

        # shape: (batch, 1, source_seq_len)
        selective_weights = tf.expand_dims(selective_weights, -2)
        # shape: (batch, 1, encoder_out_dim)
        selective_read = tf.matmul(selective_weights, state["encoder_output"])

        # shape: (batch, 1, target_emb_dim+encoder_out_dim+decoder_out_dim)
        decoder_input = tf.concat([
            target_embedded, attentive_read, selective_read], -1)
        projected_decoder_input = self.decoder_input_projector(decoder_input)

        # shape: (batch, decoder_out_dim)
        state["decoder_hidden"] = self.gru(
            projected_decoder_input, initial_state=state["decoder_hidden"])

        state["attentive_weights"] = tf.squeeze(attentive_weights, axis=-1)
        state["selective_weights"] = tf.squeeze(selective_weights, axis=-2)
        return state

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
