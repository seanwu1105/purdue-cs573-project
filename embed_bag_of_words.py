import pandas as pd
import scipy.sparse

from libs.embedding import vectorize_with_bag_of_words


def main():
    df = pd.read_csv('./assets/preprocessed.csv')
    df['text'] = df['text'].astype(str)
    sentence_vectors = vectorize_with_bag_of_words(df['text'])
    scipy.sparse.save_npz('./assets/embedded_bag_of_words', sentence_vectors)


if __name__ == '__main__':
    main()
