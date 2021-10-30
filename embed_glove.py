import numpy as np
import pandas as pd
import scipy.sparse

from libs.embedding import vectorize_with_glove


def generate_embedded_glove():
    df = pd.read_csv('./assets/preprocessed.csv', encoding='ISO-8859-1')
    df['text'] = df['text'].astype(str)
    sentence_vectors = vectorize_with_glove(df['text'])
    np.savez_compressed('./assets/embedded_glove', sentence_vectors)


def load_embedded_glove() -> np.ndarray:
    ret = np.load('./assets/embedded_glove.npz')['arr_0']
    assert isinstance(ret, np.ndarray)
    return ret


if __name__ == '__main__':
    generate_embedded_glove()
    print(load_embedded_glove().shape)
