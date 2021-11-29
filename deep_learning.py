import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

from embed_word2vec import load_embedded_word2vec

df = pd.read_csv('./assets/preprocessed.csv', encoding='ISO-8859-1')

embeddings = {'word2vec': load_embedded_word2vec}


labels = df['label'].to_numpy()
for name, load_func in embeddings.items():
    print(name)
    data = load_func()
    X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                                        test_size=0.2,
                                                        random_state=42)
    # max length of word2vec we have now
    max_features = 300
    # Input for variable-length sequences of integers

    inputs = keras.Input(shape=(None,), dtype="int32")
    # Embed each integer in a 128-dimensional vector
    x = keras.Sequential()
    x = layers.Embedding(max_features, 128)(inputs)
    # Add 2 bidirectional LSTMs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(16, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    # Add a classifier
    outputs = layers.Dense(3, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    model.compile(optimizer="adam",
                  loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    history = model.fit(X_train, y_train, batch_size=16,
                        epochs=1, validation_split=0.2)
    loss, acc = model.evaluate(x=X_test,
                               y=y_test)
    print("test loss: ", loss, ", test acc: ", 100 * acc, "%")
    model.save('lstm_model')
