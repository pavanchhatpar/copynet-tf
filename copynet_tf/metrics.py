# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""Python implementation of BLEU and smooth-BLEU.
This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

import collections
# import math
from tensorflow.keras.metrics import Metric
import tensorflow as tf


class BLEU(Metric):
    def __init__(
            self, name='bleu', max_order=4, ignore_tokens=[],
            ignore_all_tokens_after=-1, smooth=False, **kwargs):
        super(BLEU, self).__init__(name=name, **kwargs)
        self.max_order = max_order
        self.ignore_tokens = ignore_tokens
        self.ignore_all_tokens_after = ignore_all_tokens_after
        self.smooth = smooth

        self.matches_by_order = self.add_weight(
            name='matches_by_order', initializer='zeros',
            shape=(self.max_order,))
        self.possible_matches_by_order = self.add_weight(
            name='possible_matches_by_order', initializer='zeros',
            shape=(self.max_order,))
        self.reference_length = self.add_weight(
            name='reference_length', initializer='zeros')
        self.translation_length = self.add_weight(
            name='translation_length', initializer='zeros')

    def _get_ngrams(self, segment):
        """Extracts all n-grams upto a given maximum order from an input
        segment.
        Args:
            segment: text segment from which n-grams will be extracted.
        Returns:
            The Counter containing all n-grams upto max_order in segment
            with a count of how many times each n-gram occurred.
        """
        ngram_counts = collections.Counter()
        for order in range(1, self.max_order + 1):
            for i in range(0, len(segment) - order + 1):
                ngram = tuple(segment[i:i+order])
                ngram_counts[ngram] += 1
        return ngram_counts

    def _preprocess(self, reference_corpus, translation_corpus):
        if self.ignore_all_tokens_after != -1:
            reference_corpus_n = []
            for references in reference_corpus:
                references_n = []
                for reference in references:
                    i = 0
                    while (i < len(reference)
                           and reference[i] != self.ignore_all_tokens_after):
                        i += 1
                    i += 1
                    references_n.append(reference[:i])
                reference_corpus_n.append(references_n)
            reference_corpus = reference_corpus_n

            translation_corpus_n = []
            for translation in translation_corpus:
                i = 0
                while (i < len(translation)
                       and translation[i] != self.ignore_all_tokens_after):
                    i += 1
                i += 1
                translation_corpus_n.append(translation[:i])
            translation_corpus = translation_corpus_n

        if len(self.ignore_tokens) > 0:
            reference_corpus_n = []
            for references in reference_corpus:
                references_n = []
                for reference in references:
                    ref = []
                    for token in reference:
                        if token not in self.ignore_tokens:
                            ref.append(token)
                    references_n.append(ref)
                reference_corpus_n.append(references_n)
            reference_corpus = reference_corpus_n

            translation_corpus_n = []
            for translation in translation_corpus:
                trans = []
                for token in translation:
                    if token not in self.ignore_tokens:
                        trans.append(token)
                translation_corpus_n.append(trans)
            translation_corpus = translation_corpus_n
        return reference_corpus, translation_corpus

    def _compute_bleu(self, reference_corpus, translation_corpus):
        """Computes BLEU score of translated segments against one or more
        references.
        Args:
            reference_corpus: list of lists of references for each translation.
                Each reference should be tokenized into a list of tokens.
            translation_corpus: list of translations to score. Each translation
                should be tokenized into a list of tokens.
        """
        reference_corpus, translation_corpus = self._preprocess(
            reference_corpus.numpy(), translation_corpus.numpy())

        matches_by_order = [0] * self.max_order
        possible_matches_by_order = [0] * self.max_order
        reference_length = 0
        translation_length = 0
        for (references, translation) in zip(
                reference_corpus, translation_corpus):
            reference_length += min(len(r) for r in references)
            translation_length += len(translation)

            merged_ref_ngram_counts = collections.Counter()
            for reference in references:
                merged_ref_ngram_counts |= self._get_ngrams(reference)
                translation_ngram_counts = self._get_ngrams(translation)
                overlap = translation_ngram_counts & merged_ref_ngram_counts
            for ngram in overlap:
                matches_by_order[len(ngram)-1] += overlap[ngram]
            for order in range(1, self.max_order+1):
                possible_matches = len(translation) - order + 1
                if possible_matches > 0:
                    possible_matches_by_order[order-1] += possible_matches

        return (
            matches_by_order,
            possible_matches_by_order,
            reference_length,
            translation_length
        )

    @tf.function
    def _prepare_inputs(self, y_true, y_pred):
        reference_corpus = tf.expand_dims(tf.cast(y_true, tf.int32), axis=-2)
        translation_corpus = tf.argmax(y_pred, axis=-1)
        return reference_corpus, translation_corpus

    @tf.function
    def update_state(self, y_true, y_pred, sample_weight=None):
        reference_corpus, translation_corpus = self._prepare_inputs(
            y_true, y_pred)
        state = tf.py_function(self._compute_bleu, [
            reference_corpus, translation_corpus
        ], [
            tf.float32, tf.float32, tf.float32, tf.float32
        ], name="update-bleu-state")
        self.matches_by_order.assign_add(state[0])
        self.possible_matches_by_order.assign_add(state[1])
        self.reference_length.assign_add(state[2])
        self.translation_length.assign_add(state[3])

    @tf.function
    def result(self):
        precisions = tf.zeros((self.max_order,), dtype=tf.float32)
        precisions = tf.TensorArray(tf.float32, size=self.max_order)
        for i in tf.range(self.max_order):
            if self.smooth:
                precisions = precisions.write(
                    i,
                    ((self.matches_by_order[i]+1)
                     / (self.possible_matches_by_order[i]+1)))
            else:
                if self.possible_matches_by_order[i] > 0:
                    precisions = precisions.write(
                        i,
                        (self.matches_by_order[i]
                         / self.possible_matches_by_order[i]))
                else:
                    precisions = precisions.write(
                        i, tf.constant(0, tf.float32))
        precisions = precisions.stack()
        if tf.reduce_min(precisions) > 0:
            p_log_sum = tf.reduce_mean(tf.math.log(precisions))
            geo_mean = tf.exp(p_log_sum)
        else:
            geo_mean = tf.constant(0, tf.float32)

        ratio = self.translation_length / self.reference_length
        if ratio > 1:
            bp = tf.constant(1, tf.float32)
        else:
            bp = tf.exp(1 - 1/ratio)

        bleu = geo_mean * bp

        return bleu

    def reset_states(self):
        self.translation_length.assign(tf.constant(0, tf.float32))
        self.reference_length.assign(tf.constant(0, tf.float32))
        zeros = tf.zeros((self.max_order,), tf.float32)
        self.matches_by_order.assign(zeros)
        self.possible_matches_by_order.assign(zeros)


# def _get_ngrams(segment, max_order):
#     """Extracts all n-grams upto a given maximum order from an input segment.
#     Args:
#         segment: text segment from which n-grams will be extracted.
#         max_order: maximum length in tokens of the n-grams returned by this
#             methods.
#     Returns:
#         The Counter containing all n-grams upto max_order in segment
#         with a count of how many times each n-gram occurred.
#     """
#     ngram_counts = collections.Counter()
#     for order in range(1, max_order + 1):
#         for i in range(0, len(segment) - order + 1):
#             ngram = tuple(segment[i:i+order])
#             ngram_counts[ngram] += 1
#     return ngram_counts


# def _preprocess(
#         reference_corpus, translation_corpus,
#         ignore_tokens, ignore_all_tokens_after):
#     if ignore_all_tokens_after is not None:
#         reference_corpus_n = []
#         for references in reference_corpus:
#             references_n = []
#             for reference in references:
#                 i = 0
#                 while (i < len(reference)
#                        and reference[i] != ignore_all_tokens_after):
#                     i += 1
#                 i += 1
#                 references_n.append(reference[:i])
#             reference_corpus_n.append(references_n)
#         reference_corpus = reference_corpus_n

#         translation_corpus_n = []
#         for translation in translation_corpus:
#             i = 0
#             while (i < len(translation)
#                    and translation[i] != ignore_all_tokens_after):
#                 i += 1
#             i += 1
#             translation_corpus_n.append(translation[:i])
#         translation_corpus = translation_corpus_n

#     if ignore_tokens is not None:
#         reference_corpus_n = []
#         for references in reference_corpus:
#             references_n = []
#             for reference in references:
#                 ref = []
#                 for token in reference:
#                     if token not in ignore_tokens:
#                         ref.append(token)
#                 references_n.append(ref)
#             reference_corpus_n.append(references_n)
#         reference_corpus = reference_corpus_n

#         translation_corpus_n = []
#         for translation in translation_corpus:
#             trans = []
#             for token in translation:
#                 if token not in ignore_tokens:
#                     trans.append(token)
#             translation_corpus_n.append(trans)
#         translation_corpus = translation_corpus_n
#     return reference_corpus, translation_corpus


# def compute_bleu(reference_corpus, translation_corpus, max_order=4,
#                  smooth=False, ignore_tokens=None,
#                  ignore_all_tokens_after=None):
#     """Computes BLEU score of translated segments against one or more
#     references.
#     Args:
#         reference_corpus: list of lists of references for each translation.
#             Each reference should be tokenized into a list of tokens.
#         translation_corpus: list of translations to score. Each translation
#             should be tokenized into a list of tokens.
#         max_order: Maximum n-gram order to use when computing BLEU score.
#         smooth: Whether or not to apply Lin et al. 2004 smoothing.
#     Returns:
#         3-Tuple with the BLEU score, n-gram precisions, geometric mean of
#         n-gram precisions and brevity penalty.
#     """
#     reference_corpus, translation_corpus = _preprocess(
#         reference_corpus, translation_corpus,
#         ignore_tokens, ignore_all_tokens_after)
#     matches_by_order = [0] * max_order
#     possible_matches_by_order = [0] * max_order
#     reference_length = 0
#     translation_length = 0
#     for (references, translation) in zip(reference_corpus,
#                                          translation_corpus):
#         reference_length += min(len(r) for r in references)
#         translation_length += len(translation)

#         merged_ref_ngram_counts = collections.Counter()
#         for reference in references:
#             merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
#             translation_ngram_counts = _get_ngrams(translation, max_order)
#             overlap = translation_ngram_counts & merged_ref_ngram_counts
#         for ngram in overlap:
#             matches_by_order[len(ngram)-1] += overlap[ngram]
#         for order in range(1, max_order+1):
#             possible_matches = len(translation) - order + 1
#             if possible_matches > 0:
#                 possible_matches_by_order[order-1] += possible_matches

#     precisions = [0] * max_order
#     for i in range(0, max_order):
#         if smooth:
#             precisions[i] = ((matches_by_order[i] + 1.) /
#                              (possible_matches_by_order[i] + 1.))
#         else:
#             if possible_matches_by_order[i] > 0:
#                 precisions[i] = (float(matches_by_order[i]) /
#                                  possible_matches_by_order[i])
#             else:
#                 precisions[i] = 0.0

#     if min(precisions) > 0:
#         p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
#         geo_mean = math.exp(p_log_sum)
#     else:
#         geo_mean = 0

#     ratio = float(translation_length) / reference_length

#     if ratio > 1.0:
#         bp = 1.
#     else:
#         bp = math.exp(1 - 1. / ratio)

#     bleu = geo_mean * bp

#     return (
#           bleu, precisions, bp, ratio, translation_length, reference_length)
