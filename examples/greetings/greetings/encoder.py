import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Bidirectional, GRU
from typing import Dict, Any, List
from copynet_tf import Vocab

from .config import cfg


class Encoder(Model):
    def __init__(self,
                 vocab: Vocab,
                 **kwargs: Dict[str, Any]) -> None:
        super(Encoder, self).__init__(**kwargs)
        self.vocab = vocab
        self._build()

    def _build(self):
        self.bigru = Bidirectional(GRU(
            cfg.HIDDEN_DIM//2, return_sequences=True, return_state=True))

    def get_output_dim(self) -> int:
        return cfg.HIDDEN_DIM

    def initialize_hidden_size(self, batch_sz: int) -> List[tf.Tensor]:
        return [
            tf.zeros((batch_sz, cfg.HIDDEN_DIM//2)),
            tf.zeros((batch_sz, cfg.HIDDEN_DIM//2))
        ]

    def call(
            self,
            source_token_ids: tf.Tensor,
            source_embeddings: tf.Tensor,
            source_mask: tf.Tensor,
            encoder_hidden: tf.Tensor,
            training: bool = None):
        encoder_output, enc_final_f, enc_final_b = self.bigru(
            source_embeddings, mask=source_mask, initial_state=encoder_hidden)

        encoder_final_output = tf.concat([enc_final_f, enc_final_b], -1)

        return {
            "encoder_output": encoder_output,
            "encoder_final_output": encoder_final_output,
        }
