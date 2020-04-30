import tensorflow as tf
from tensorflow.keras import Model
from typing import Dict, Any
from copynet_tf import Vocab, CopyNetDecoder
from copynet_tf.search import BeamSearch
from copynet_tf.layers import FixedEmbedding, FixedDense
import os
from tqdm import tqdm

from .config import cfg
from .encoder import Encoder
from .glove_reader import GloVeReader


class GreetingModel(Model):
    def __init__(self,
                 **kwargs: Dict[str, Any]) -> None:
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
        self.searcher = BeamSearch(
            3, self.vocab.get_token_id(self.vocab._end_token, "target"),
            cfg.TSEQ_LEN)
        self.encoder = Encoder(self.vocab)
        target_vocab_size = self.vocab.get_vocab_size("target")
        target_emb_mat = tf.transpose(tf.convert_to_tensor(
            self.vocab.get_embedding_matrix("target")))
        self.decoder_output_layer = FixedDense(
            target_vocab_size,
            [target_emb_mat, tf.zeros(target_vocab_size)])
        self.decoder = CopyNetDecoder(
            self.vocab, self.encoder.get_output_dim(),
            self.searcher, self.decoder_output_layer)
        emb_mat = tf.convert_to_tensor(
            self.vocab.get_embedding_matrix("source"))
        self.source_embedder = FixedEmbedding(
            emb_mat, cfg.SSEQ_LEN, mask_zero=True)

    @tf.function
    def forward_loss(
            self, X, y, enc_hidden, epoch_no, batch_no, training=False):
        source_token_ids, source2target_ids = X
        target_token_ids, target2source_ids = y
        source_embeddings = self.source_embedder(source_token_ids)
        source_mask = self.source_embedder.compute_mask(source_token_ids)
        state = self.encoder(
            source_token_ids, source_embeddings,
            source_mask, enc_hidden, training=training)
        output_dict = {}
        if training:
            output_dict = self.decoder(
                source_token_ids, source2target_ids, source_embeddings,
                source_mask, state, target_token_ids, target2source_ids,
                training=training)
        else:
            output_dict = self.decoder.predict(
                source_token_ids, source2target_ids, source_embeddings,
                source_mask, state, target_token_ids, target2source_ids,
                training=training)
        return output_dict

    @tf.function
    def train_step(self, X, y, enc_hidden, epoch_no, batch_no):
        with tf.GradientTape() as tape:
            output_dict = self.forward_loss(
                X, y, enc_hidden, epoch_no, batch_no, True)

            loss = output_dict['loss']

            vars = (self.encoder.trainable_variables
                    + self.decoder.trainable_variables)

            gradients = tape.gradient(loss, vars)

            self.optimizer.apply_gradients(zip(gradients, vars))

        return loss

    def fit(self, dataset, epochs, save_loc, eval_set=None, warm_start=False):
        if not warm_start:
            self.optimizer = tf.keras.optimizers.Adam(
                cfg.LR, clipnorm=cfg.CLIP_NORM)
            ckpt_saver = tf.train.Checkpoint(
                optimizer=self.optimizer,
                encoder=self.encoder,
                decoder=self.decoder)
        else:
            self.optimizer, ckpt_saver = self._load(save_loc)
        save_prefix = os.path.join(save_loc, "ckpt")
        for epoch in tf.range(epochs):
            eloss = tf.constant(0, dtype=tf.float32)
            i = tf.constant(0, dtype=tf.float32)
            with tqdm(
                    dataset, desc=f"Epoch {epoch.numpy()+1}/{epochs}") as pbar:
                for X, y in pbar.iterable:
                    enc_hidden = self.encoder.initialize_hidden_size(
                        y[0].shape[0])
                    bloss = self.train_step(X, y, enc_hidden, epoch+1, i+1)
                    pbar.update(1)
                    i += 1
                    eloss = (eloss*(i-1) + bloss)/i
                    metrics = {"train-loss": f"{eloss:.4f}"}
                    pbar.set_postfix(metrics)
                if eval_set is not None:
                    vloss = tf.constant(0, dtype=tf.float32)
                    n = tf.constant(0, dtype=tf.float32)
                    training = tf.constant(True)
                    for X, y in eval_set:
                        enc_hidden = self.encoder.initialize_hidden_size(
                            y[0].shape[0])
                        loss = self.forward_loss(
                            X, y, enc_hidden, epoch+1, n+1, training)
                        vloss += loss['loss']
                        n += 1
                    metrics['val-loss'] = f"{vloss/n:.4f}"
                    pbar.set_postfix(metrics)
                ckpt_saver.save(file_prefix=save_prefix)

    def predict(self, dataset):
        ret_val = None
        ret_proba = None
        i = tf.constant(1)
        e = tf.constant(1)
        nottraining = tf.constant(False)
        for X, y in dataset:
            enc_hidden = self.encoder.initialize_hidden_size(
                        y[0].shape[0])
            op = self.forward_loss(
                X, y, enc_hidden, e, i, nottraining)
            i += 1
            if ret_val is None:
                ret_val = op["predictions"]
                ret_proba = op["predicted_probas"]
                continue
            ret_val = tf.concat([ret_val, op["predictions"]], 0)
            ret_proba = tf.concat([ret_proba, op["predicted_probas"]], 0)
        return ret_val, ret_proba

    def _load(self, save_loc):
        optimizer = tf.keras.optimizers.Adam(cfg.LR, clipnorm=cfg.CLIP_NORM)
        ckpt = tf.train.Checkpoint(
            optimizer=optimizer,
            encoder=self.encoder,
            decoder=self.decoder
        )
        ckpt.restore(
            tf.train.latest_checkpoint(save_loc)).expect_partial()
        return optimizer, ckpt

    def load(self, save_loc):
        self._load(save_loc)
