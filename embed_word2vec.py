import numpy as np
import pandas as pd

from libs.embedding import vectorize_with_word2vec


def generate_embedded_word2vec():
    df = pd.read_csv('./assets/preprocessed.csv')
    df['text'] = df['text'].astype(str)
    sentence_vectors = vectorize_with_word2vec(df['text'])

    np.savez_compressed('./assets/embedded_word2vec', sentence_vectors)


def load_embedded_word2vec() -> np.ndarray:
    ret = np.load('./assets/embedded_word2vec.npz')['arr_0']
    assert isinstance(ret, np.ndarray)
    return ret


if __name__ == '__main__':
    generate_embedded_word2vec()
    print(load_embedded_word2vec().shape)
