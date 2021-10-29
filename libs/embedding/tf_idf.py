from typing import Iterable

from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_with_tf_idf(corpus: Iterable[str], min_df: int = 5):
    vectorizer = TfidfVectorizer(min_df=min_df, sublinear_tf=True)
    return vectorizer.fit_transform(corpus)
