import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, GRU
from typing import Dict, Any, Tuple
import logging

from .layers import FixedEmbedding, BahdanauAttention
from .vocab import Vocab
from .search import Searcher
from .util import masked_log_softmax, masked_softmax


class Decoder(Model):
    def __init__(self,
                 vocab: Vocab,
                 encoder_out_dim: int,
                 searcher: Searcher,
                 output_layer: Layer,
                 copy_token: str = "@COPY@",
                 **kwargs: Dict[str, Any]) -> None:
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

    def _build_decoder_layers(self, output_layer):
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

    def call(self,
             source_token_ids: tf.Tensor,
             source2target_ids: tf.Tensor,
             source_embeddings: tf.Tensor,
             source_mask: tf.Tensor,
             state: Dict[str, tf.Tensor],
             target_token_ids: tf.Tensor = None,
             target2source_ids: tf.Tensor = None,
             training: bool = None) -> Dict[str, tf.Tensor]:
        """
        # Parameters

        source_token_ids : `tf.Tensor`
            Shape: (batch_size, source_seq_len)
        source2target_ids : `tf.Tensor`
            Shape: (batch_size, source_seq_len)
        target_token_ids : `tf.Tensor`
            Shape: (batch_size, target_seq_len)
        target2source_ids : `tf.Tensor`
            Shape: (batch_size, target_seq_len)
        training : `bool`
        """
        state["source_mask"] = source_mask
        state["source_token_ids"] = source_token_ids
        state["source2target_ids"] = source2target_ids
        state["source_embeddings"] = source_embeddings

        if target_token_ids is not None:
            state = self._init_decoder_state(state)
            output_dict = self._calculate_loss(
                target_token_ids, target2source_ids, state)
            # fill dummy values to make `call` and `predict` graph give
            # consistent outputs
            output_dict["predictions"] = 1
            output_dict["predicted_probas"] = 1.0
        else:
            output_dict = {}

        return output_dict

    def predict(self,
                source_token_ids: tf.Tensor,
                source2target_ids: tf.Tensor,
                source_embeddings: tf.Tensor,
                source_mask: tf.Tensor,
                state: Dict[str, tf.Tensor],
                target_token_ids: tf.Tensor = None,
                target2source_ids: tf.Tensor = None,
                training: bool = None) -> Dict[str, tf.Tensor]:
        output_dict = {}
        state = self._init_decoder_state(state)
        output_dict.update(self._decode_output(state))
        # fill dummy values to make `call` and `predict` graph give
        # consistent outputs
        output_dict["loss"] = 1.0
        output_dict["attentive_weights"] = 1.0
        output_dict["selective_weights"] = 1.0
        return output_dict

    def _init_decoder_state(
            self, state: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        # shape: (batch, decoder_out_dim)
        state["decoder_hidden"] = state["encoder_final_output"]
        return state

    def _decode_output(
            self, state: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        batch_size, source_length = state["source_mask"].shape
        # shape: (batch_size, source_length)
        state["copy_log_probs"] = tf.math.log(
            (tf.zeros((batch_size, source_length), dtype=tf.float32) + 1e-35))
        start_predictions = tf.fill(
            (batch_size,), tf.constant(self._start_index, dtype=tf.int32))

        # shape: (batch_size, beam_width, max_decoding_steps)
        # shape: (batch_size, beam_width)
        all_top_k_predictions, log_probabilities = self._searcher.search(
            start_predictions, state, self._take_search_step)

        return {
            "predicted_probas": log_probabilities,
            "predictions": all_top_k_predictions,
        }

    def _take_search_step(self,
                          last_predictions: tf.Tensor,
                          state: Dict[str, tf.Tensor]) -> Tuple[
                              tf.Tensor, Dict[str, tf.Tensor]]:
        _, source_seq_len = state["source_mask"].shape

        # Get input to the decoder RNN and the selective weights.
        # `input_choices` is the result of replacing target OOV tokens in
        # `last_predictions` with the copy symbol. `selective_weights` consist
        # of the normalized copy probabilities assigned to the source tokens
        # that were copied. If no tokens were copied, there will be all zeros.
        # shape: (batch_size,), (batch_size, source_seq_len)
        input_choices, selective_weights = (
            self._get_input_and_selective_weights(last_predictions, state))

        # Update the decoder state by taking a step through the RNN.
        state = self._decoder_step(input_choices, selective_weights, state)
        del state["selective_weights"]
        del state["attentive_weights"]
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

    def _get_input_and_selective_weights(
        self, last_predictions: tf.Tensor, state: Dict[str, tf.Tensor]
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

    def _gather_final_log_probs(
        self,
        generation_log_probs: tf.Tensor,
        copy_log_probs: tf.Tensor,
        state: Dict[str, tf.Tensor],
    ) -> tf.Tensor:
        _, source_seq_len = state["source_mask"].shape
        source_token_ids = state["source_token_ids"]

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
            copy_log_probs_to_add_mask = tf.cast(
                source2target_ids_slice != self._unk_index, tf.float32)
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
            # this is the first occurence of this particular source token, we
            # add the log_probs from all other occurences, otherwise we zero it
            # out since it was already accounted for.
            if i < (source_seq_len - 1):
                # Sum copy scores from future occurences of source token.
                # shape: (batch_size, source_seq_len - i)
                source_future_occurences = tf.cast(
                    source_token_ids[:, (i + 1):]
                    == tf.expand_dims(source_token_ids[:, i], -1), tf.float32)
                # shape: (batch_size, source_seq_len - i)
                future_copy_log_probs = (
                    copy_log_probs[:, (i + 1):]
                    + tf.math.log(source_future_occurences + 1e-35)
                )
                # shape: (batch_size, 1 + source_seq_len - i)
                combined = tf.concat(
                    [tf.expand_dims(copy_log_probs_slice, -1),
                     future_copy_log_probs], axis=-1
                )
                # shape: (batch_size,)
                copy_log_probs_slice = tf.math.reduce_logsumexp(
                    combined, axis=-1)
            if i > 0:
                # Remove copy log_probs that we have already accounted for.
                # shape: (batch_size, i)
                source_previous_occurences = tf.cast(source_token_ids[
                    :, 0:i
                ] == tf.expand_dims(source_token_ids[:, i], -1), tf.int32)
                # shape: (batch_size,)
                duplicate_mask = tf.cast(tf.reduce_sum(
                    source_previous_occurences, axis=-1) == 0, tf.float32)
                copy_log_probs_slice = (
                    copy_log_probs_slice + tf.math.log(duplicate_mask + 1e-35)
                )

            # Finally, we zero-out copy scores that we added to the generation
            # scores above so that we don't double-count them.
            # shape: (batch_size,)
            left_over_copy_log_probs = (
                copy_log_probs_slice
                + tf.math.log(1 - copy_log_probs_to_add_mask + 1e-35)
            )
            modified_log_probs_list.append(
                tf.expand_dims(left_over_copy_log_probs, -1))
        modified_log_probs_list.insert(0, generation_log_probs)

        # shape: (batch_size, target_vocab_size + source_seq_len)
        modified_log_probs = tf.concat(modified_log_probs_list, axis=-1)
        # tf.py_function(
        #     self.debug, ['modified_log_probas', tf.reduce_logsumexp(
        #         modified_log_probs[:3], axis=-1)], [])

        return modified_log_probs

    def _calculate_loss(self,
                        target_token_ids: tf.Tensor,
                        target2source_ids: tf.Tensor,
                        state: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        # Parameters

        target_token_ids : `tf.Tensor`
            Shape: (batch_size, target_seq_len)
        target2source_ids : `tf.Tensor`
            Shape: (batch_size, target_seq_len)
        state : `Dict[str, tf.Tensor]`
        """
        batch_size, target_seq_len = target_token_ids.shape
        _, source_seq_len = state["source_token_ids"].shape
        num_decoding_steps = target_seq_len - 1
        # shape: (batch, source_seq_len)
        source2target_slice = tf.zeros_like(
            state["source_token_ids"], dtype=tf.float32)

        # mask 1 if source token is not unknown
        # shape: (batch, source_seq_len)
        source_spl_mask = ~(state["source_token_ids"] == self._unk_index)
        source_spl_mask &= ~(state["source_token_ids"] == self._start_index)
        source_spl_mask &= ~(state["source_token_ids"] == self._end_index)

        # shape: (batch,)
        copy_slice = tf.fill((batch_size, ), tf.constant(
            self._copy_index, dtype=tf.int32))

        # shape: (batch, source_seq_len)
        selective_weights = tf.zeros_like(
            state["source_mask"], dtype=tf.float32)

        # shape: (batch, source_seq_len)
        copy_mask = tf.cast(state["source_mask"], tf.float32)

        # shape: (batch, target_vocab_size)
        generation_mask = tf.ones(
            (batch_size, self._target_vocab_size), dtype=tf.float32)

        # shape: (batch, num_decoding_steps)
        step_log_likelihoods = tf.zeros(
            (batch_size, num_decoding_steps), dtype=tf.float32)

        all_attentive_weights = tf.zeros(
            (batch_size, num_decoding_steps, source_seq_len), dtype=tf.float32)
        all_selective_weights = tf.zeros(
            (batch_size, num_decoding_steps, source_seq_len), dtype=tf.float32)
        for timestep in tf.range(num_decoding_steps):
            # shape: (batch,)
            target_input_slice = target_token_ids[:, timestep]

            if timestep < num_decoding_steps - 1:
                # shape: (batch,)
                copied = tf.cast((
                    (target_input_slice == self._unk_index)
                    & (tf.reduce_sum(source2target_slice, -1) > 0)
                ), tf.int32)

                target_input_slice = (
                    target_input_slice * (1 - copied) + copy_slice * copied
                )

                # shape: (batch, source_seq_len)
                source2target_slice = (
                    state["source_token_ids"]
                    == tf.expand_dims(target2source_ids[:, timestep + 1], 1))

                # shape: (batch, source_seq_len)
                source2target_slice = source2target_slice & source_spl_mask

                # shape: (batch, source_seq_len)
                source2target_slice = tf.cast(source2target_slice, tf.float32)

            # Update decoder hidden state by stepping through the decoder
            state = self._decoder_step(
                target_input_slice, selective_weights, state)

            # shape: (batch_size, source_seq_len)
            attn_weights = tf.squeeze(state["attentive_weights"], -1)

            indices = tf.expand_dims(
                tf.range(batch_size, dtype=tf.int32), 1)
            # shape: (batch_size, 2)
            indices = tf.concat([
                indices,
                tf.fill(
                    (batch_size, 1),
                    timestep)], 1)

            all_attentive_weights = tf.tensor_scatter_nd_update(
                all_attentive_weights, indices, attn_weights)

            # shape: (batch_size, source_seq_len)
            sel_weights = tf.squeeze(state["selective_weights"], -2)

            all_selective_weights = tf.tensor_scatter_nd_update(
                all_selective_weights, indices, sel_weights)

            del state["selective_weights"]
            del state["attentive_weights"]

            # shape: (batch, target_vocab_size)
            generation_scores = self._get_generation_scores(state)

            # shape: (batch, source_seq_len)
            copy_scores = self._get_copy_score(state)

            # shape: (batch,)
            target_output_slice = target_token_ids[:, timestep+1]

            # shape: (batch,), (batch, source_seq_len)
            step_log_likelihood, selective_weights = self._get_ll_contrib(
                generation_scores, copy_scores, generation_mask,
                copy_mask, target_output_slice, source2target_slice)

            # shape: (batch, 1)
            step_log_likelihood = tf.expand_dims(step_log_likelihood, 1)

            if timestep > 0:
                step_log_likelihoods = tf.concat(
                    [step_log_likelihoods[:, :timestep],
                     step_log_likelihood,
                     step_log_likelihoods[:, timestep+1:]], 1)
                step_log_likelihoods.set_shape(
                    (batch_size, num_decoding_steps))
            else:
                step_log_likelihoods = tf.concat(
                    [step_log_likelihood,
                     step_log_likelihoods[:, 1:]], 1)
            # step_log_likelihoods = tf.concat(
            #     [step_log_likelihoods, step_log_likelihood], 1)

        # shape: (batch, num_decoding_steps)
        log_likelihoods = step_log_likelihoods

        # shape: (batch, target_seq_len)
        target_mask = tf.cast(target_token_ids != self._pad_index, tf.float32)

        # shape: (batch, num_decoding_steps)
        target_mask = target_mask[:, 1:]

        # shape: (batch,)
        log_likelihood = tf.reduce_sum(log_likelihoods*target_mask, axis=-1)

        # The loss is the negative log-likelihood, averaged over the batch.
        # shape: ()
        loss = -tf.reduce_sum(log_likelihood) / batch_size

        return {
            "loss": loss,
            "attentive_weights": all_attentive_weights,
            "selective_weights": all_selective_weights,
        }

    def _decoder_step(self,
                      last_predictions: tf.Tensor,
                      selective_weights: tf.Tensor,
                      state: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """
        # Parameters

        last_predictions : `tf.Tensor`
            Shape : (batch,)
        selective_weights : `tf.Tensor`
            Shape : (batch, source_seq_len)
        state : `Dict[str, tf.Tensor]`
        """
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

        state["attentive_weights"] = attentive_weights
        state["selective_weights"] = selective_weights
        return state

    def _get_generation_scores(
            self, state: Dict[str, tf.Tensor]) -> tf.Tensor:
        return self.output_generation_layer(state["decoder_hidden"])

    def _get_copy_score(self, state: Dict[str, tf.Tensor]) -> tf.Tensor:
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

    def debug(self, name, value):
        self.logger.debug(f"Debug {name} {value}")

    def _get_ll_contrib(self,
                        generation_scores: tf.Tensor,
                        copy_scores: tf.Tensor,
                        generation_mask: tf.Tensor,
                        copy_mask: tf.Tensor,
                        target_token_id: tf.Tensor,
                        source2target: tf.Tensor) -> Tuple[
                            tf.Tensor, tf.Tensor]:
        _, target_size = generation_scores.shape
        # tf.py_function(
        #     self.debug, ['target_token_id', target_token_id[:3]], [])
        # tf.py_function(
        #     self.debug, ['source2target', source2target[:3]], [])

        # shape: (batch, target_vocab_size + source_seq_len)
        mask = tf.concat([generation_mask, copy_mask], -1)

        # shape: (batch, target_vocab_size + source_seq_len)
        all_scores = tf.concat([generation_scores, copy_scores], -1)

        # Normalize generation and copy scores.
        # shape: (batch_size, target_vocab_size + source_seq_len)
        log_probs = masked_log_softmax(all_scores, mask, axis=1)

        # Calculate the log probability (`copy_log_probs`) for each token in
        # the source sentence that matches the current target token. We use the
        # sum of these copy probabilities for matching tokens in the source
        # sentence to get the total probability for the target token. We also
        # need to normalize the individual copy probabilities to create
        # `selective_weights`, which are used in the next timestep to create
        # a selective read state.
        # shape: (batch, source_seq_len)
        copy_log_probs = (
            log_probs[:, target_size:] + tf.math.log(
                tf.cast(source2target, tf.float32) + 1e-35)
        )
        # Since `log_probs[:, target_size]` gives us the raw copy log
        # probabilities, we use a non-log softmax to get the normalized non-log
        # copy probabilities.
        selective_weights = masked_softmax(
            log_probs[:, target_size:], source2target, axis=1)
        # tf.py_function(
        #     self.debug, ['selective_weights', selective_weights[:3]], [])
        # This mask ensures that item in the batch has a non-zero generation
        # probabilities for this timestep only when the real target token is
        # not OOV or there are no matching tokens in the source sentence.
        # shape: (batch_size,)
        gen_mask = tf.cast(
            (target_token_id != self._unk_index)
            | (tf.reduce_sum(source2target, -1) == 0), tf.float32)
        # shape: (batch_size, 1)
        log_gen_mask = tf.expand_dims(tf.math.log(gen_mask + 1e-35), 1)

        # shape: (batch_size, 1)
        generation_log_probs = (
            tf.gather(
                log_probs, tf.expand_dims(target_token_id, 1), axis=1,
                batch_dims=1)
            + log_gen_mask
        )
        # shape: (batch, 1+source_seq_len)
        combined_gen_copy = tf.concat(
            [generation_log_probs, copy_log_probs], -1)
        # tf.py_function(
        #     self.debug, ['combined_gen_copy', combined_gen_copy[:3]], [])

        # shape: (batch,)
        step_log_likelihood = tf.math.reduce_logsumexp(
            combined_gen_copy, axis=-1)

        # tf.py_function(
        #     self.debug, ['step_log_likelihood', step_log_likelihood[:3]], [])

        return step_log_likelihood, selective_weights
