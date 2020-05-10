import tensorflow as tf
from typing import Dict, Callable, Tuple
import logging

from .searcher import Searcher


class BeamSearch(Searcher):
    def __init__(self,
                 beam_width: int,
                 end_index: int,
                 max_decoding_steps: int) -> None:
        self.logger = logging.getLogger(__name__)
        self.beam_width = beam_width
        self._end_index = end_index
        self._max_decoding_steps = max_decoding_steps

    def debug(self, name, value):
        self.logger.debug(f"Debug {name} {value}")

    def _first_token(self,
                     start_predictions: tf.Tensor,
                     state: Dict[str, tf.Tensor],
                     step: Callable[
                         [tf.Tensor, Dict[str, tf.Tensor]],
                         Tuple[tf.Tensor, Dict[str, tf.Tensor]]]
                     ) -> Tuple[tf.Tensor, tf.Tensor]:
        batch_size = start_predictions.shape[0]
        # shape: (batch_size*beam_width, max_decoding_steps)
        predictions = tf.fill(
            (batch_size*self.beam_width, self._max_decoding_steps),
            tf.constant(self._end_index, tf.int32), name="predictions")
        # shape: (batch_size*beam_width, )
        log_probs = tf.fill(
            (batch_size*self.beam_width, ),
            tf.constant(1, tf.float32), name="log_probs")
        # shape: (batch_size, target_vocab_size+source_seq_len)
        predicted_proba, state = step(start_predictions, state)
        # shape: (batch_size, beam_width) both
        topkpreds, topkidx = tf.math.top_k(
            predicted_proba, self.beam_width)
        # shape: (batch_size*beam_width, )
        topkidx = tf.reshape(
            tf.cast(topkidx, tf.int32), (batch_size*self.beam_width, ))
        # shape: (batch_size*beam_width, )
        topkpreds = tf.reshape(
            topkpreds, (batch_size*self.beam_width, ))
        indices = tf.expand_dims(
            tf.range(batch_size*self.beam_width, dtype=tf.int32), 1)
        # shape: (batch_size*beam_width, 2)
        indices = tf.concat([
            indices,
            tf.fill(
                (batch_size*self.beam_width, 1),
                tf.constant(0, dtype=tf.int32))], 1)
        predictions = tf.tensor_scatter_nd_update(
            predictions, indices, topkidx)
        log_probs = log_probs*topkpreds
        # tf.py_function(self.debug, ['first', predictions[:9]], [])
        for key, state_tensor in state.items():
            state_tensor = tf.repeat(state_tensor, self.beam_width, axis=0)
            state[key] = state_tensor

        return state, predictions, log_probs

    def search(self,
               start_predictions: tf.Tensor,
               start_state: Dict[str, tf.Tensor],
               step: Callable[
                   [tf.Tensor, Dict[str, tf.Tensor]],
                   Tuple[tf.Tensor, Dict[str, tf.Tensor]]]
               ) -> Tuple[tf.Tensor, tf.Tensor]:

        batch_size = start_predictions.shape[0]
        # shape: (batch_size*beam_width, max_decoding_steps),
        #        (batch_size*beam_width, )
        state, predictions, log_probabilities = self._first_token(
            start_predictions, start_state, step)

        for timestep in tf.range(1, self._max_decoding_steps):
            # shape: (batch_size*beam_width, )
            last_predictions = predictions[:, timestep-1]
            # shape: (batch_size*beam_width, target_vocab_size+source_seq_len)
            predicted_proba, state = step(last_predictions, state)
            # shape: (batch_size*beam_width, beam_width) both
            topkpreds, topkidx = tf.math.top_k(
                predicted_proba, self.beam_width)
            # shape: (batch_size*beam_width, 1)
            was_end = tf.expand_dims(last_predictions == self._end_index, 1)
            # shape: (batch_size, beam_width*beam_width)
            has_no_end = tf.reshape(
                tf.repeat(~was_end, self.beam_width, axis=-1),
                (batch_size, self.beam_width*self.beam_width))
            # shape: (batch_size, beam_width*beam_width)
            has_no_end = tf.cast(has_no_end, tf.float32)

            # shape: (batch_size*beam_width, beam_width)
            beam_mask = tf.repeat(~was_end, self.beam_width, axis=-1)
            beam_mask = tf.concat(
                [was_end | ~was_end, beam_mask[:, 1:]], axis=-1)

            # shape: (batch_size, beam_width*beam_width)
            beam_mask = tf.cast(tf.reshape(
                beam_mask, (batch_size, self.beam_width*self.beam_width)),
                tf.float32)

            # shape: (batch_size, beam_width*beam_width)
            topkidx = tf.reshape(
                tf.cast(topkidx, tf.int32),
                (batch_size, self.beam_width*self.beam_width))
            # shape: (batch_size, beam_width*beam_width)
            topkpreds = tf.reshape(
                topkpreds, (batch_size, self.beam_width*self.beam_width))
            # shape: (batch_size, beam_width*beam_width)
            step_log_probs = tf.reshape(
                tf.repeat(log_probabilities, self.beam_width, axis=0),
                (batch_size, self.beam_width*self.beam_width))
            step_log_probs = step_log_probs + topkpreds*has_no_end

            # we don't want to repeatedly select beams from end token which
            # have all same probability. So we make all but one of such beams'
            # log probability highly negative to never get it in top k beams
            step_log_probs += tf.math.log(beam_mask + 1e-35)

            # shape: (batch_size, beam_width) both
            topsteplog, topsteplogidx = tf.math.top_k(
                step_log_probs, self.beam_width)

            # shape: (batch_size*beam_width, )
            topsteplog = tf.reshape(topsteplog, (batch_size*self.beam_width,))
            log_probabilities = topsteplog

            # shape: (batch_size, beam_width)
            topkidx = tf.gather(topkidx, topsteplogidx, axis=-1, batch_dims=1)
            # shape: (batch_size*beam_width, )
            topkidx = tf.reshape(topkidx, (batch_size*self.beam_width,))

            # shape: (batch_size, beam_width)
            has_no_end = tf.gather(
                has_no_end, topsteplogidx, axis=-1, batch_dims=1)
            # shape: (batch_size*beam_width, )
            has_no_end = tf.reshape(has_no_end, (batch_size*self.beam_width,))
            topkidx = tf.where(
                tf.cast(has_no_end, bool),
                topkidx,
                self._end_index)

            indices = tf.expand_dims(
                tf.range(batch_size*self.beam_width, dtype=tf.int32), 1)
            # shape: (batch_size*beam_width, 2)
            indices = tf.concat([
                indices,
                tf.fill(
                    (batch_size*self.beam_width, 1),
                    tf.cast(timestep, tf.int32))], 1)
            predictions = tf.tensor_scatter_nd_update(
                predictions, indices, topkidx)

            # tf.py_function(self.debug, ['current', predictions[:9]], [])
            for key, state_tensor in state.items():
                _, *last_dims = state_tensor.shape
                state_tensor = tf.reshape(tf.repeat(
                    state_tensor, self.beam_width, axis=0),
                    (batch_size, self.beam_width*self.beam_width, *last_dims))
                state_tensor = tf.gather(
                    state_tensor, topsteplogidx, axis=1, batch_dims=1)
                state_tensor = tf.reshape(
                    state_tensor, (batch_size*self.beam_width, *last_dims))
                state[key] = state_tensor

            for t in tf.range(timestep):
                prev_preds = predictions[:, t]
                # shape: (batch_size, beam_width*beam_width)
                prev_preds = tf.reshape(tf.repeat(
                    prev_preds, self.beam_width, axis=0),
                    (batch_size, self.beam_width*self.beam_width))
                # shape: (batch_size, beam_width)
                prev_preds = tf.gather(
                    prev_preds, topsteplogidx, axis=-1, batch_dims=1)
                # shape: (batch_size*beam_width, )
                prev_preds = tf.reshape(
                    prev_preds, (batch_size*self.beam_width,))
                indices = tf.expand_dims(
                    tf.range(batch_size*self.beam_width, dtype=tf.int32), 1)
                # shape: (batch_size*beam_width, 2)
                indices = tf.concat([
                    indices,
                    tf.fill(
                        (batch_size*self.beam_width, 1),
                        tf.cast(t, tf.int32))], 1)
                predictions = tf.tensor_scatter_nd_update(
                    predictions, indices, prev_preds)
                # tf.py_function(self.debug, ['backfill', predictions[:9]], [])

        # shape: (batch_size, beam_width, max_decoding_steps)
        predictions = tf.reshape(
            predictions,
            (batch_size, self.beam_width, self._max_decoding_steps))
        # shape: (batch_size, beam_width)
        log_probabilities = tf.reshape(
            log_probabilities, (batch_size, self.beam_width))

        return predictions, log_probabilities
