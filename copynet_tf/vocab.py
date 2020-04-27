from collections import defaultdict
import numpy as np
import pickle


# TODO check how to get embedding matrix for source and target
class Vocab:
    def __init__(self,
                 start_token,
                 end_token,
                 pad_token,
                 unk_token,
                 source_seq_len,
                 target_seq_len):
        self._start_token = start_token
        self._end_token = end_token
        self._pad_token = pad_token
        self._unk_token = unk_token
        self._source_seq_len = source_seq_len
        self._target_seq_len = target_seq_len

    def fit(self,
            tokenized_source,
            tokenized_target,
            min_source_freq=1,
            min_target_freq=5):
        source_vocab = defaultdict()
        for tokens in tokenized_source:
            for token in tokens:
                source_vocab[token] += 1

        target_vocab = defaultdict()
        for tokens in tokenized_target:
            for token in tokens:
                target_vocab[token] += 1

        source_vocab = sorted(source_vocab.items(), key=lambda x: -x[1])
        if min_source_freq is not None:
            source_vocab = filter(
                lambda x: x[1] > min_source_freq, source_vocab)

        target_vocab = sorted(target_vocab.items(), key=lambda x: -x[1])
        if min_target_freq is not None:
            target_vocab = filter(
                lambda x: x[1] > min_target_freq, target_vocab)

        self._source = {}
        self._source[self._pad_token] = 0
        self._source[self._unk_token] = 1
        self._source[self._start_token] = 2
        self._source[self._end_token] = 3
        for i, k in enumerate(source_vocab, start=4):
            self._source[k[0]] = i

        self._source_inverse = {}
        for token, idx in self._source:
            self._source_inverse[idx] = token

        self._target = {}
        self._target[self._pad_token] = 0
        self._target[self._unk_token] = 1
        self._target[self._start_token] = 2
        self._target[self._end_token] = 3
        for i, k in enumerate(target_vocab, start=4):
            self._target[k[0]] = i

        self._target_inverse = {}
        for token, idx in self._target:
            self._target_inverse[idx] = token

    def _transform(self, tokenized, token2index, seq_len):
        res = np.zeros((len(tokenized), seq_len), dtype=np.int64)
        for i, tokens in enumerate(tokenized):
            res[i, 0] = token2index[self._start_token]
            for j, token in enumerate(tokens, start=1):
                if j > seq_len - 2:
                    j -= 1
                    break
                res[i, j] = token2index.get(
                    token, token2index[self._unk_token])
            res[i, j+1] = token2index[self._end_token]
        return res

    def transform(self, tokenized, namespace):
        if namespace == 'source':
            return self._transform(
                tokenized, self._source, self._source_seq_len)
        elif namespace == 'target':
            return self._transform(
                tokenized, self._target, self._target_seq_len)
        else:
            raise ValueError(f"Unknown namespace: {namespace}")

    def _get_token_id(self, token, token2index):
        return token2index.get(token, token2index[self._unk_token])

    def get_token_id(self, token, namespace):
        if namespace == 'source':
            return self._get_token_id(token, self._source)
        elif namespace == 'target':
            return self._get_token_id(token, self._target)
        else:
            raise ValueError(f"Unknown namespace: {namespace}")

    def _add_token(self, token, token2index):
        if token in token2index:
            return token2index[token]
        i = len(token2index)
        token2index[token] = i
        return i

    def add_token(self, token, namespace):
        if namespace == 'source':
            return self._get_token_id(token, self._source)
        elif namespace == 'target':
            return self._get_token_id(token, self._target)
        else:
            raise ValueError(f"Unknown namespace: {namespace}")

    def get_sequence_len(self, namespace):
        if namespace == 'source':
            return self._source_seq_len
        elif namespace == 'target':
            return self._target_seq_len
        else:
            raise ValueError(f"Unknown namespace: {namespace}")

    def _get_token_text(self, idx, idx2token):
        return idx2token.get(idx)

    def get_token_text(self, idx, namespace):
        if namespace == 'source':
            return self._get_token_text(idx, self._source_inverse)
        elif namespace == 'target':
            return self._get_token_text(idx, self._target_inverse)
        else:
            raise ValueError(f"Unknown namespace: {namespace}")

    def _inverse_transform(self, idxed, idx2token):
        res = np.full_like(idxed, dtype=object)
        for i, idxs in enumerate(idxed):
            for j, idx in enumerate(idxs):
                res[i, j] = self._get_token_text(idx, idx2token)
        return res

    def inverse_transform(self, idxed, namespace):
        if namespace == 'source':
            return self._inverse_transform(idxed, self._source_inverse)
        elif namespace == 'target':
            return self._inverse_transform(idxed, self._target_inverse)
        else:
            raise ValueError(f"Unknown namespace: {namespace}")

    def save(self, filename_prefix):
        source_filename = filename_prefix+"_source.pkl"
        target_filename = filename_prefix+"_target.pkl"
        with open(source_filename, "wb") as f:
            pickle.dump(self._source, f, protocol=4)
        with open(target_filename, "wb") as f:
            pickle.dump(self._target, f, protocol=4)

    @classmethod
    def load(cls,
             start_token,
             end_token,
             pad_token,
             unk_token,
             source_seq_len,
             target_seq_len,
             filename_prefix):
        source_filename = filename_prefix+"_source.pkl"
        target_filename = filename_prefix+"_target.pkl"
        inst = cls(start_token, end_token, pad_token, unk_token,
                   source_seq_len, target_seq_len)
        with open(source_filename, "wb") as f:
            inst._source = pickle.load(f)
        with open(target_filename, "wb") as f:
            inst._target = pickle.load(f)
        inst._source_inverse = {}
        for token, idx in inst._source:
            inst._source_inverse[idx] = token
        inst._target_inverse = {}
        for token, idx in inst._target:
            inst._target_inverse[idx] = token
        return inst

    def get_vocab_size(self, namespace):
        if namespace == 'source':
            return len(self._source)
        elif namespace == 'target':
            return len(self._target)
        else:
            raise ValueError(f"Unknown namespace: {namespace}")
