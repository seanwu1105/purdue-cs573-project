import pickle

import numpy as np


def vectorize_with_glove(data, dim=100):
    with open('glove.pickle', 'rb') as handle:
        glove = pickle.load(handle)
    X = np.zeros((len(data), dim))
    invalid = 0
    for n in range(len(data)):
        tweet = data[n]
        tokens = tweet.split()
        vecs = []
        for word in tokens:
            try:
                # throws KeyError if word not found
                vecs.append(glove[word])
            except KeyError:
                pass
        if len(vecs) > 0:
            vecs = np.array(vecs)
            X[n] = vecs.mean(axis=0)
        else:
            invalid += 1
    return X
