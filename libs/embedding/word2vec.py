import os
from typing import Iterable

import gensim.models
import numpy as np
from scipy.sparse.csr import csr_matrix

from .tf_idf import vectorize_with_tf_idf


def vectorize_with_word2vec(corpus: Iterable[str]):
    word_vectors = load_word2vec_word_vectors()
    X, vectorizer = vectorize_with_tf_idf(
        corpus, return_vectorizer=True, min_df=1)
    vocabulary = vectorizer.get_feature_names_out()
    return product_sentence_vectors_with_word_vectors(
        X,
        vocabulary,
        word_vectors)


def load_word2vec_word_vectors():
    os.environ['GENSIM_DATA_DIR'] = './.venv/lib/gensim-data'
    import gensim.downloader  # pylint: disable=redefined-outer-name

    return gensim.downloader.load('word2vec-google-news-300')


def product_sentence_vectors_with_word_vectors(
        sentence_vectors: csr_matrix,
        vocabulary: np.ndarray,
        word_vectors: gensim.models.KeyedVectors):
    ret = []
    for tf_idf_vector in sentence_vectors:
        producted = np.zeros(300)
        reconized_count = 0
        for data_idx, idx in enumerate(tf_idf_vector.indices):
            try:
                wv = word_vectors[vocabulary[idx]]
                producted += (wv * tf_idf_vector.data[data_idx])
                reconized_count += 1
            except KeyError:
                pass
        if reconized_count > 0:
            producted /= reconized_count
        ret.append(producted)

    return np.array(ret)
