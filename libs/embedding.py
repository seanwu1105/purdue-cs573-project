import os
from typing import Iterable

import pandas as pd
from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def vectorize_with_bag_of_words(corpus: Iterable[str]):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer


def vectorize_with_tf_idf(corpus: Iterable[str]):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer


def pprint_vectors(vectors: csr_matrix,
                   vectorizer: CountVectorizer):
    print(pd.DataFrame(vectors.toarray(),
                       columns=vectorizer.get_feature_names_out()))


def vectorize_with_word2vec(corpus: Iterable[str]):
    os.environ['GENSIM_DATA_DIR'] = './.venv/lib/gensim-data'
    import gensim.downloader
    word_vectors = gensim.downloader.load('word2vec-google-news-300')

    X, vectorizer = vectorize_with_tf_idf(corpus)
    vocabulary = vectorizer.get_feature_names_out()

    # TODO: create a matrix of sentence vectors


# vectorize_with_word2vec(['hello world', 'fuck this world'])
x, y = vectorize_with_tf_idf(['hello world', 'fuck this world'])
print(y.get_feature_names_out())
pprint_vectors(x, y)
