import tensorflow as tf
from tensorflow.keras import Model
from copynet_tf import Vocab, CopyNetDecoder
from copynet_tf.search import BeamSearch
from copynet_tf.layers import FixedEmbedding, FixedDense
from copynet_tf.types import StrDict
from copynet_tf.util import prep_y_true
# from copynet_tf.metrics import compute_bleu
# import os
# from tqdm import tqdm

from .config import cfg
from .encoder import Encoder
from .glove_reader import GloVeReader


class GreetingModel(Model):
    def __init__(self,
                 **kwargs: StrDict) -> None:
        super(GreetingModel, self).__init__(**kwargs)
        reader = GloVeReader()
        self.vocab = Vocab.load(
            reader.START,
            reader.END,
            reader.PAD,
            reader.UNK,
            cfg.SSEQ_LEN,
            cfg.TSEQ_LEN,
            cfg.VOCAB_SAVE
        )
        copy_token = "@COPY@"
        self.vocab.add_token(copy_token, "target")
        self.searcher = BeamSearch(
            10, self.vocab.get_token_id(self.vocab._end_token, "target"),
            cfg.TSEQ_LEN - 1)
        self.encoder = Encoder(self.vocab)
        target_vocab_size = self.vocab.get_vocab_size("target")
        target_emb_mat = tf.transpose(tf.convert_to_tensor(
            self.vocab.get_embedding_matrix("target")))
        self.decoder_output_layer = FixedDense(
            target_vocab_size,
            [target_emb_mat, tf.zeros(target_vocab_size)])
        self.decoder = CopyNetDecoder(
            self.vocab, self.encoder.get_output_dim(),
            self.searcher, self.decoder_output_layer, copy_token=copy_token)
        emb_mat = tf.convert_to_tensor(
            self.vocab.get_embedding_matrix("source"))
        self.source_embedder = FixedEmbedding(
            emb_mat, cfg.SSEQ_LEN, mask_zero=True)

    @tf.function
    def call(self, X, y=(None, None), training=False):
        source_token_ids, source2target_ids = X
        target_token_ids, target2source_ids = y
        batch_size, _ = source_token_ids.shape
        enc_hidden = self.encoder.initialize_hidden_size(batch_size)
        source_embeddings = self.source_embedder(source_token_ids)
        source_mask = self.source_embedder.compute_mask(source_token_ids)
        state = self.encoder(
            source_token_ids, source_embeddings,
            source_mask, enc_hidden, training=training)
        output_dict = self.decoder(
            source_token_ids, source2target_ids, source_mask, state,
            target_token_ids, target2source_ids, training=training)
        return output_dict

    @tf.function
    def train_step(self, data):
        X, y = data
        training = tf.constant(True)
        source_token_ids, source2target_ids = X
        target_token_ids, target2source_ids = y

        target_vocab_size = self.vocab.get_vocab_size("target")
        unk_index = self.vocab.get_token_id(self.vocab._unk_token, "source")
        start_index = self.vocab.get_token_id(
            self.vocab._start_token, "source")
        end_index = self.vocab.get_token_id(self.vocab._end_token, "source")
        y_true = prep_y_true(
            source_token_ids, target_token_ids, target2source_ids,
            target_vocab_size, unk_index, start_index, end_index)
        with tf.GradientTape() as tape:
            output_dict = self(X, y, training)
            # shape: ()
            loss = self.compiled_loss(
                y_true,
                output_dict['ypred'])

            gradients = tape.gradient(loss, self.trainable_variables)

            self.optimizer.apply_gradients(zip(
                gradients, self.trainable_variables))
        self.compiled_metrics.update_state(
                y_true,
                output_dict['ypred'])
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        X, y = data
        training = tf.constant(True)
        source_token_ids, source2target_ids = X
        target_token_ids, target2source_ids = y

        target_vocab_size = self.vocab.get_vocab_size("target")
        unk_index = self.vocab.get_token_id(self.vocab._unk_token, "source")
        start_index = self.vocab.get_token_id(
            self.vocab._start_token, "source")
        end_index = self.vocab.get_token_id(self.vocab._end_token, "source")
        y_true = prep_y_true(
            source_token_ids, target_token_ids, target2source_ids,
            target_vocab_size, unk_index, start_index, end_index)

        output_dict = self(X, y, training)
        self.compiled_loss(
                y_true,
                output_dict['ypred'])
        self.compiled_metrics.update_state(
                y_true,
                output_dict['ypred'])
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def predict_step(self, data):
        X, _ = data
        return self(X)
