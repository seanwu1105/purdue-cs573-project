import pandas as pd
import scipy.sparse

from libs.embedding import vectorize_with_tf_idf


def main():
    df = pd.read_csv('./assets/preprocessed.csv')
    df['text'] = df['text'].astype(str)
    sentence_vectors = vectorize_with_tf_idf(df['text'])
    scipy.sparse.save_npz('./assets/embedded_tf_idf', sentence_vectors)


if __name__ == '__main__':
    main()
