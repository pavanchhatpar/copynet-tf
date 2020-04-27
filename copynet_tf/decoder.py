import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, GRU
from typing import Dict, Any, Tuple

from .layers import FixedEmbedding, AdditiveAttention
from .vocab import Vocab
from .search import Searcher
from .util import masked_log_softmax, masked_softmax


class Decoder(Model):
    def __init__(self,
                 encoder: Model,  # TODO consider interface adding constraints
                 vocab: Vocab,
                 searcher: Searcher,
                 embedding_matrix: tf.Tensor,
                 copy_token: str = "@COPY@"):
        self.vocab = vocab
        self._copy_index = self.vocab.add_token(copy_token, "target")
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
        self._encoder = encoder
        self._searcher = searcher
        self._embedding_matrix = embedding_matrix
        self.encoder_out_dim = self._encoder.get_output_dim()
        self.decoder_out_dim = self.encoder_out_dim
        self.decoder_inp_dim = self.decoder_out_dim
        self._build_decoder_layers()

    def _build_decoder_layers(self):
        self.target_embedder = FixedEmbedding(
            self._embedding_matrix, self.vocab, "target")
        self.decode_attn = AdditiveAttention()
        self.decoder_input_projector = Dense(self.decoder_inp_dim)
        self.gru = GRU(self.decoder_out_dim)
        self._target_vocab_size = self.vocab.get_vocab_size("target")
        self.output_generation_layer = Dense(self._target_vocab_size)
        self.output_copying_layer = Dense(
            self.decoder_out_dim, activation="tanh")

    def call(self,
             source_token_ids: tf.Tensor,
             source2target_ids: tf.Tensor,
             target_token_ids: tf.Tensor = None,
             target2source_ids: tf.Tensor = None,
             encoder_args: Dict[str, Any] = {},
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
        encoder_args : `Dict[str, Any]`
        training : `bool`
        """
        state = self._encoder(
            source_token_ids, training=training, **encoder_args)
        state["source_token_ids"] = source_token_ids
        state["source2target_ids"] = source2target_ids

        if target_token_ids is not None:
            state = self._init_decoder_state(state)
            output_dict = self._calculate_loss(
                target_token_ids, target2source_ids, state)
        else:
            output_dict = {}

        if not training:
            state = self._init_decoder_state(state)
            predictions = self._decode_output(state)
            output_dict["predictions"] = predictions

        return output_dict

    def _init_decoder_state(self,
                            state: Dict[str, tf.Tensor]):
        # shape: (batch, decoder_out_dim)
        state["decoder_hidden"] = state["encoder_final_output"]
        return state

    def _decode_output(self, state: Dict[str, tf.Tensor]) -> tf.Tensor:
        # TODO write this function
        pass

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
        num_decoding_steps = target_seq_len - 1

        # shape: (batch, )
        source2target_slice = tf.zeros_like(
            state["source2target_ids"], dtype=tf.float32)

        # shape: (batch,)
        copy_slice = tf.fill((batch_size, ), tf.constant(
            self._copy_index, dtype=tf.int32))

        # shape: (batch, source_seq_len)
        selective_weights = tf.zeros_like(
            state["source_mask"], dtype=tf.float32)

        # shape: (batch, source_seq_len)
        copy_mask = state["source_mask"]

        # shape: (batch, target_vocab_size)
        generation_mask = tf.ones(
            (batch_size, self._target_vocab_size), dtype=tf.float32)

        step_log_likelihoods = []
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

                # shape: (batch, )
                source2target_slice = (
                    state["source2target_ids"]
                    == target_token_ids[:, timestep + 1])

            # Update decoder hidden state by stepping through the decoder
            state = self._decoder_step(
                target_input_slice, selective_weights, state)

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
            step_log_likelihood = tf.expand_dims(step_log_likelihood)
            step_log_likelihoods.append(step_log_likelihood)

        # shape: (batch, num_decoding_steps)
        log_likelihoods = tf.concat(step_log_likelihoods, 1)

        # shape: (batch, target_seq_len)
        target_mask = tf.cast(target_token_ids != self._pad_index, tf.float32)

        # shape: (batch, num_decoding_steps)
        target_mask = target_mask[:, 1:]

        # shape: (batch,)
        log_likelihood = tf.reduce_sum(log_likelihoods*target_mask, axis=-1)

        # The loss is the negative log-likelihood, averaged over the batch.
        # shape: ()
        loss = -tf.reduce_sum(log_likelihood) / batch_size

        return {"loss": loss}

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
        # shape: (batch, 1, decoder_out_dim)
        query = tf.expand_dims(state["decoder_hidden"], -2)
        # shape: (batch, 1, decoder_out_dim), (batch, 1, source_seq_len)
        attentive_read, attentive_weights = self.decode_attn(
            inputs=[query, state["encoder_output"]],
            mask=[None, source_mask])

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
        copy_scores = tf.squeeze(copy_scores)

        return copy_scores

    def _get_ll_contrib(self,
                        generation_scores: tf.Tensor,
                        copy_scores: tf.Tensor,
                        generation_mask: tf.Tensor,
                        copy_mask: tf.Tensor,
                        target_token_id: tf.Tensor,
                        source2target: tf.Tensor) -> Tuple(
                            tf.Tensor, tf.Tensor):
        _, target_size = generation_scores.shape

        # shape: (batch, target_vocab_size + source_seq_len)
        mask = tf.concat([generation_mask, copy_mask], -1)

        # shape: (batch, target_vocab_size + source_seq_len)
        all_scores = tf.concat([generation_scores, copy_scores], -1)

        # Normalize generation and copy scores.
        # shape: (batch_size, target_vocab_size + source_seq_len)
        log_probs = masked_log_softmax(all_scores, mask)

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
                tf.cast(source2target, tf.float32) + 1e-45)
        )
        # Since `log_probs[:, target_size]` gives us the raw copy log
        # probabilities, we use a non-log softmax to get the normalized non-log
        # copy probabilities.
        selective_weights = masked_softmax(
            log_probs[:, target_size:], source2target
        )
        # This mask ensures that item in the batch has a non-zero generation
        # probabilities for this timestep only when the real target token is
        # not OOV or there are no matching tokens in the source sentence.
        # shape: (batch_size,)
        gen_mask = tf.cast(
            (target_token_id != self._unk_index)
            | (tf.reduce_sum(source2target, -1) == 0), tf.float32)
        # shape: (batch_size, 1)
        log_gen_mask = tf.expand_dims(tf.math.log(gen_mask + 1e-45), 1)

        # shape: (batch_size, 1)
        generation_log_probs = (
            tf.gather(log_probs, tf.expand_dims(target_token_id, 1), axis=1)
            + log_gen_mask
        )
        # shape: (batch, 1+source_seq_len)
        combined_gen_copy = tf.concat(
            [generation_log_probs, copy_log_probs], -1)

        # shape: (batch,)
        step_log_likelihood = tf.math.reduce_logsumexp(
            combined_gen_copy, axis=-1)

        return step_log_likelihood, selective_weights
