import numpy as np
import pandas as pd
import scipy.sparse

from libs.embedding import vectorize_with_tf_idf


def generate_embedded_tf_idf():
    df = pd.read_csv('./assets/preprocessed.csv')
    df['text'] = df['text'].astype(str)
    sentence_vectors = vectorize_with_tf_idf(df['text'])
    scipy.sparse.save_npz('./assets/embedded_tf_idf', sentence_vectors)


def load_embedded_tf_idf() -> np.ndarray:
    ret = scipy.sparse.load_npz('./assets/embedded_tf_idf.npz').toarray()
    assert isinstance(ret, np.ndarray)
    return ret


if __name__ == '__main__':
    generate_embedded_tf_idf()
    print(load_embedded_tf_idf().shape)
