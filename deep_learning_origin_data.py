import pandas as pd
import tensorflow as tf
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers, preprocessing

EPOCHS = 16
BATCH_SIZE = 16
MAX_SEQUENCE_LENGTH = 266

PREPROCESSED_DATA_PATH = 'assets/preprocessed.csv'
df = pd.read_csv(PREPROCESSED_DATA_PATH, encoding='ISO-8859-1')
data = df['text']
data.fillna('', inplace=True)
MAX_NB_WORDS = 50000
tokenizer = preprocessing.text.Tokenizer(
    num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~', lower=True)
data = tokenizer.texts_to_sequences(data.values)
data = preprocessing.sequence.pad_sequences(data, maxlen=MAX_SEQUENCE_LENGTH)
labels = df['label'].to_numpy()

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    data, labels, test_size=0.2, random_state=42)

y_train = tf.keras.utils.to_categorical(LabelEncoder().fit_transform(y_train))
y_test = tf.keras.utils.to_categorical(LabelEncoder().fit_transform(y_test))

# build LSTM model
# Text Input
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
embedding_vector_feature = 64
voc_size = 20000
model = keras.Sequential()
model.add(layers.Embedding(voc_size, embedding_vector_feature,
          input_length=MAX_SEQUENCE_LENGTH))
model.add(layers.SpatialDropout1D(0.2))
# model.add(layers.Embedding(voc_size, embedding_vector_feature)(text_input))
model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.4))
model.add(layers.Dense(3, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train
history = model.fit(x=X_train,
                    y=y_train,
                    validation_split=0.2,
                    epochs=EPOCHS,
                    verbose=1,
                    batch_size=BATCH_SIZE)
# Evaluation
loss, acc = model.evaluate(x=X_test,
                           y=y_test)
print("test loss: ", loss, ", test acc: ", 100 * acc, "%")
model.save('lstm_original_data')
