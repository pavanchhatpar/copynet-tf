from tensorflow.keras.layers import Embedding


class FixedEmbedding(Embedding):
    def __init__(self,
                 embedding_matrix,
                 vocab,
                 namespace,
                 **kwargs):
        super().__init__(
            embedding_matrix.shape[0],
            embedding_matrix.shape[1],
            weights=[embedding_matrix],
            trainable=False,
            input_length=vocab.get_sequence_len(namespace),
            **kwargs
        )
