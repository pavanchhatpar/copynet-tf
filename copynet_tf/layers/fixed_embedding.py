from tensorflow.keras.layers import Embedding


class FixedEmbedding(Embedding):
    def __init__(self, embedding_matrix, max_seq_len=None, **kwargs):
        super().__init__(
            embedding_matrix.shape[0],
            embedding_matrix.shape[1],
            trainable=False,
            input_length=max_seq_len,
            **kwargs
        )
        self.embeddings = embedding_matrix

    def build(self, input_shape):
        # with tf.device('cpu:0'):
        #     self.embeddings = self.add_weight(
        #         shape=(self.input_dim, self.output_dim),
        #         initializer=self.embeddings_initializer,
        #         name='embeddings',
        #         regularizer=self.embeddings_regularizer,
        #         constraint=self.embeddings_constraint,
        #         caching_device='cpu:0')
        self.built = True
