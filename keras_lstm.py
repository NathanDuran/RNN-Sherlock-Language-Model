import keras
from keras.preprocessing import text
from keras.callbacks import LambdaCallback
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Masking, Embedding, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.utils.data_utils import get_file
import numpy as np
import random
from utilities import *
import h5py

# Parameters
vocabulary_size = 8000
hidden_layer = 100
learning_rate = 0.005
num_epoch = 10
file_path = "resources\Short Stories.txt"
data_path = "./data/sherlock-training-data.pkl"
model_path = "./data/keras-sherlock-language-model"
data = load_training_data(data_path)

x_train = np.array(data["x_train"])
y_train = np.array(data["y_train"])
word_to_index = data["word_to_index"]
index_to_word = data["index_to_word"]
vocabulary = data["vocabulary"]

for i in range(len(x_train)):
    np.array(x_train[i])
    np.array(y_train[i])

x_train = pad_sequences(x_train, maxlen=None, padding='post', value=0)
y_train = pad_sequences(y_train, maxlen=None, padding='post', value=0)

num_sentences = x_train.shape[0]
max_input_len = x_train.shape[1]
print(x_train[0])
# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(Masking(mask_value=0, input_shape=(124,)))
model.add(Embedding(vocabulary_size, hidden_layer, input_length=max_input_len))
model.add(LSTM(hidden_layer))
model.add(Dense(max_input_len))
# model.add(TimeDistributed(Dense(max_input_len)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

model.fit(x_train, y_train, batch_size=128, epochs=num_epoch)

model.save(model_path + '.hdf5', overwrite=True)

model = load_model(model_path + '.hdf5')
print(model.summary())

unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# We start the sentence with the start token
new_sentence = np.zeros(max_input_len)
new_sentence = np.insert(new_sentence, 0, word_to_index[sentence_start_token])
print(new_sentence)
model.p
preds = model.predict(new_sentence)
print("PREDICTION" , preds)
