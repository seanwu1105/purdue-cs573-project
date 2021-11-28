import pandas as pd
df = pd.read_csv('./assets/preprocessed.csv', encoding='ISO-8859-1')
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
#from xgboost import XGBClassifier
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

classifiers = (
    # LogisticRegression(max_iter=100000),
    # KNeighborsClassifier(n_neighbors=10),
    # GaussianNB(),
    # RandomForestClassifier(max_depth=20, n_estimators=1000),
    LinearSVC(max_iter=100000),
    # XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
)

from embed_bag_of_words import load_embedded_bag_of_words
from embed_tf_idf import load_embedded_tf_idf
from embed_word2vec import load_embedded_word2vec
from embed_glove import load_embedded_glove

embeddings = {#'bag of words': load_embedded_bag_of_words,
              #'tf idf': load_embedded_tf_idf,
              'word2vec': load_embedded_word2vec,
              'glove': load_embedded_glove}

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

labels = df['label'].to_numpy()
for name, load_func in embeddings.items():
    print(name)
    data = load_func()
    X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                                        test_size=0.2,
                                                        random_state=42)
    max_features = 20000
    # Input for variable-length sequences of integers
    inputs = keras.Input(shape=(None,), dtype="int32")
    # Embed each integer in a 128-dimensional vector
    x = layers.Embedding(max_features, 128)(inputs)
    # Add 2 bidirectional LSTMs

    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.GlobalMaxPool1D()(x)
    #x = layers.Bidirectional(layers.LSTM(16))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    # Add a classifier
    outputs = layers.Dense(3, activation="relu")(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    model.compile("adam", "mse", metrics=["accuracy"])
    model.fit(X_train, y_train, batch_size=32, epochs=2, validation_data=(X_test, y_test))

    '''
    for classifier in classifiers:
        classifier.n_jobs = -1
        start_time = time.time()
        classifier.fit(X_train, y_train)
        time_elapsed = time.time() - start_time
        y_pred = classifier.predict(X_test)
        acc_test = accuracy_score(y_test, y_pred)
        print(f'{classifier.__class__.__name__} test accuracy: {acc_test:.3f}, training time: {time_elapsed}')
    print()
    '''