import numpy as np
import pandas as pd

from libs.embedding import vectorize_with_bag_of_words


def main():
    df = pd.read_csv(f'./assets/clean.csv')
    df['text'] = df['text'].astype(str)
    sentence_vectors, _ = vectorize_with_bag_of_words(df['text'])
    print(sentence_vectors.shape)
    # np.save(f'./assets/embedded_bag_of_words', sentence_vectors)


if __name__ == '__main__':
    main()
