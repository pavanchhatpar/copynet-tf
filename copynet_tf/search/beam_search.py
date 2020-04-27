import tensorflow as tf
from .searcher import Searcher


class BeamSearch(Searcher):
    def __init__(self, beam_width, end_index, max_decoding_steps):
        self._beam_width = beam_width
        self._end_index = end_index
        self._max_decoding_steps = max_decoding_steps

    def search(self, start_predictions, decode_step, state):
        sequences = start_predictions
        log_probs = tf.ones(start_predictions.shape[0])
        i = 0
        prev_predictions = start_predictions
        while i < self._max_decoding_steps:
            state = decode_step(prev_predictions, state)
            # TODO use tf.math.top_k with k = beam_width
