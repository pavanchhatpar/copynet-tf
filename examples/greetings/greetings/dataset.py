import logging
import en_core_web_sm
import os
import tensorflow as tf
from copynet_tf import Vocab
from typing import Dict, Tuple, Any

from .glove_reader import GloVeReader
from .config import cfg


class Dataset:
    def __init__(self,
                 prepare: bool = False) -> None:
        self.logger = logging.getLogger(__name__)
        if prepare:
            self._prepare()
        if not os.path.exists(cfg.SAVE_LOC) or not os.path.isdir(cfg.SAVE_LOC):
            raise ValueError(f"{cfg.SAVE_LOC} should be a directory!")
        TRAIN = os.path.join(cfg.SAVE_LOC, "train.tfrecord")
        TEST = os.path.join(cfg.SAVE_LOC, "test.tfrecord")
        if not os.path.exists(TRAIN) or not os.path.exists(TEST):
            raise ValueError(f"Dataset not present inside {cfg.SAVE_LOC}!")
        train = tf.data.TFRecordDataset([TRAIN], compression_type='ZLIB')
        test = tf.data.TFRecordDataset([TEST], compression_type='ZLIB')
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.train = train.map(self.parse_ex, num_parallel_calls=AUTOTUNE)
        self.test = test.map(self.parse_ex, num_parallel_calls=AUTOTUNE)

    def __getstate__(self) -> Dict[str, Any]:
        dic = self.__dict__.copy()
        del dic['logger']
        return dic

    def __setstate__(self, dic: Dict[str, Any]) -> None:
        self.__dict__.update(dic)
        self.logger = logging.getLogger(__name__)

    def parse_ex(self, example_proto) -> Tuple[tf.Tensor, tf.Tensor]:
        feature_description = {
            'X': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'y': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'Xt': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'yt': tf.io.FixedLenFeature([], tf.string, default_value=''),
        }
        example = tf.io.parse_single_example(
            example_proto, feature_description)
        X = tf.io.parse_tensor(example['X'], out_type=tf.int32)
        X.set_shape([cfg.SSEQ_LEN, ])
        y = tf.io.parse_tensor(example['y'], out_type=tf.int32)
        y.set_shape([cfg.TSEQ_LEN, ])
        Xt = tf.io.parse_tensor(example['Xt'], out_type=tf.int32)
        Xt.set_shape([cfg.SSEQ_LEN, ])
        yt = tf.io.parse_tensor(example['yt'], out_type=tf.int32)
        yt.set_shape([cfg.TSEQ_LEN, ])
        return ((X, Xt), (y, yt))

    def _prepare(self) -> None:
        self.logger.info("****Preparing dataset****")
        reader = GloVeReader()
        pretrained_vectors = reader.read(cfg.EMBS_FILE)
        vocab = Vocab(
            reader.START,
            reader.END,
            reader.PAD,
            reader.UNK,
            cfg.SSEQ_LEN,
            cfg.TSEQ_LEN
        )
        pardir = os.path.dirname(cfg.VOCAB_SAVE)
        if not os.path.exists(pardir):
            os.makedirs(pardir)
        nlp = en_core_web_sm.load()
        train_file = os.path.join(cfg.RAW_DATA, 'greetings/train.tsv')
        test_file = os.path.join(cfg.RAW_DATA, 'greetings/validation.tsv')
        train_X = []
        train_y = []
        with open(train_file, 'r') as f:
            for line in f:
                X, y = line.split("\t")
                train_X.append(X.strip())
                train_y.append(y.strip())
        train_X_docs = list(nlp.pipe(train_X))
        train_y_docs = list(nlp.pipe(train_y))

        test_X = []
        test_y = []
        with open(test_file, 'r') as f:
            for line in f:
                X, y = line.split("\t")
                test_X.append(X.strip())
                test_y.append(y.strip())
        test_X_docs = list(nlp.pipe(test_X))
        test_y_docs = list(nlp.pipe(test_y))

        vocab.fit(
            train_X_docs, train_y_docs, pretrained_vectors, 0, 3)

        vocab.save(cfg.VOCAB_SAVE)

        train_X_ids = vocab.transform(train_X_docs, "source")
        train_y_ids = vocab.transform(train_y_docs, "target")
        train_Xt_ids = vocab.transform(train_X_docs, "target")
        train_yt_ids = vocab.transform(train_y_docs, "source")

        test_X_ids = vocab.transform(test_X_docs, "source")
        test_y_ids = vocab.transform(test_y_docs, "target")
        test_Xt_ids = vocab.transform(test_X_docs, "target")
        test_yt_ids = vocab.transform(test_y_docs, "source")

        sseq = cfg.SSEQ_LEN
        tseq = cfg.TSEQ_LEN

        def gen():
            for X, Xt, y, yt in zip(
                    train_X_ids, train_Xt_ids, train_y_ids, train_yt_ids):
                yield (X, Xt, y, yt)

        train_dataset = tf.data.Dataset.from_generator(
            gen,
            (tf.int32, tf.int32, tf.int32, tf.int32),
            (tf.TensorShape([sseq]), tf.TensorShape([sseq]),
                tf.TensorShape([tseq]), tf.TensorShape([tseq]))
        )

        def gen():
            for X, Xt, y, yt in zip(
                    test_X_ids, test_Xt_ids, test_y_ids, test_yt_ids):
                yield (X, Xt, y, yt)

        test_dataset = tf.data.Dataset.from_generator(
            gen,
            (tf.int32, tf.int32, tf.int32, tf.int32),
            (tf.TensorShape([sseq]), tf.TensorShape([sseq]),
                tf.TensorShape([tseq]), tf.TensorShape([tseq]))
        )

        train_dataset = train_dataset.map(
            self.make_example, num_parallel_calls=-1)
        test_dataset = test_dataset.map(
            self.make_example, num_parallel_calls=-1)

        self.save(train_dataset, test_dataset)

    def save(self, train, test):
        if not os.path.exists(cfg.SAVE_LOC):
            os.makedirs(cfg.SAVE_LOC)
        if not os.path.isdir(cfg.SAVE_LOC):
            raise ValueError(f"{cfg.SAVE_LOC} should be a directory!")
        self.logger.info("******** Saving Test set ********")
        fname = os.path.join(cfg.SAVE_LOC, "test.tfrecord")
        writer = tf.data.experimental.TFRecordWriter(fname, "ZLIB")
        writer.write(test)

        self.logger.info("******** Saving Training set ********")
        fname = os.path.join(cfg.SAVE_LOC, "train.tfrecord")
        writer = tf.data.experimental.TFRecordWriter(fname, "ZLIB")
        writer.write(train)
        self.logger.info("******** Finished saving dataset ********")

    def make_example(self, X, Xt, y, yt):
        serialized = tf.py_function(
            self.serialize,
            [X, Xt, y, yt],
            tf.string
        )
        return tf.reshape(serialized, ())

    def _bytes_feature(self, value):
        """Returns a bytes_list from a string / byte."""

        # BytesList won't unpack a string from an EagerTensor.
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def serialize(self, X, Xt, y, yt):
        X = tf.io.serialize_tensor(X)
        y = tf.io.serialize_tensor(y)
        Xt = tf.io.serialize_tensor(Xt)
        yt = tf.io.serialize_tensor(yt)
        feature = {
            "X": self._bytes_feature(X),
            "y": self._bytes_feature(y),
            "Xt": self._bytes_feature(Xt),
            "yt": self._bytes_feature(yt),
        }
        example_proto = tf.train.Example(
            features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
