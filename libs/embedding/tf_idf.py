from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_with_tf_idf(corpus: Iterable[str]):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer
