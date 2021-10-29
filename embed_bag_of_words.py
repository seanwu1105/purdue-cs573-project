import numpy as np
import pandas as pd
import scipy.sparse

from libs.embedding import vectorize_with_bag_of_words


def generate_embedded_bag_of_words():
    df = pd.read_csv('./assets/preprocessed.csv')
    df['text'] = df['text'].astype(str)
    sentence_vectors = vectorize_with_bag_of_words(df['text'])
    scipy.sparse.save_npz('./assets/embedded_bag_of_words', sentence_vectors)


def load_embedded_bag_of_words() -> np.ndarray:
    ret = scipy.sparse.load_npz('./assets/embedded_bag_of_words.npz').toarray()
    assert isinstance(ret, np.ndarray)
    return ret


if __name__ == '__main__':
    generate_embedded_bag_of_words()
    print(load_embedded_bag_of_words().shape)
