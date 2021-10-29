from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_with_tf_idf(corpus: Iterable[str], return_vectorizer: bool = False, min_df: int = 5):
    vectorizer = TfidfVectorizer(min_df=min_df, sublinear_tf=True)
    sentence_vectors = vectorizer.fit_transform(corpus)

    if return_vectorizer:
        return sentence_vectors, vectorizer
    return sentence_vectors
