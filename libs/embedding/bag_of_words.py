from typing import Iterable

from sklearn.feature_extraction.text import CountVectorizer


def vectorize_with_bag_of_words(corpus: Iterable[str]):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer
