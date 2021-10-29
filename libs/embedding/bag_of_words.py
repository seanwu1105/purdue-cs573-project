from typing import Iterable

from sklearn.feature_extraction.text import CountVectorizer


def vectorize_with_bag_of_words(corpus: Iterable[str], min_df: int = 5):
    vectorizer = CountVectorizer(min_df=min_df)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer
