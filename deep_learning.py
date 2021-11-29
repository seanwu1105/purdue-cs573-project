import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from embed_word2vec import load_embedded_word2vec

def feed_forward_neural_network():
    model = keras.Sequential()
    model.add(layers.Input(shape=(300,)))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(3, activation='softmax'))
    model.summary()
    model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    return model

def LSTM():
    max_features = 300
    model = keras.Sequential()
    inputs = keras.Input(shape=(None,), dtype="int32")
    model.add(inputs)
    model.add(layers.Embedding(max_features, 128))
    model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(3, activation='softmax'))
    model.summary()
    model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    return model


df = pd.read_csv('./assets/preprocessed.csv', encoding='ISO-8859-1')

embeddings = {'word2vec': load_embedded_word2vec}

from sklearn.model_selection import train_test_split

labels = df['label'].to_numpy()
for name, load_func in embeddings.items():
    print(name)
    data = load_func()
    X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                                        test_size=0.2,
                                                        random_state=42)
    # model = feed_forward_neural_network()
    model = LSTM()

    history = model.fit(X_train, y_train, batch_size=16, epochs=10, validation_split=0.2)
    loss, acc = model.evaluate(x=X_test,
                               y=y_test)
    print("test loss: ", loss, ", test acc: ", 100 * acc, "%")
    model.save('lstm_model')
