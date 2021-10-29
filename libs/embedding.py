import pandas as pd
from typing import Iterable
from scipy.sparse.csr import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def bag_of_words(corpus: Iterable[str]):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer


def tf_idf(corpus: Iterable[str]):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer


def pprint_vectors(vectors: csr_matrix,
                   vectorizer: CountVectorizer):
    print(pd.DataFrame(vectors.toarray(),
                       columns=vectorizer.get_feature_names_out()))
