from .beam_search import BeamSearch


class GreedySearch(BeamSearch):
    def __init__(self, end_index, max_decoding_steps):
        super(GreedySearch, self).__init__(1, end_index, max_decoding_steps)
