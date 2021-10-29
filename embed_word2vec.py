import numpy as np
import pandas as pd
from libs.embedding import vectorize_with_word2vec


def main():
    train_data = pd.read_csv('./assets/Corona_NLP_train_clean.csv')
    sentence_vectors = vectorize_with_word2vec(train_data['text'])

    np.save('./assets/word2vec_train_vectors', sentence_vectors)


if __name__ == '__main__':
    main()
