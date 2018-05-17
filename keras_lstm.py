import keras
from keras.preprocessing import text
from keras.callbacks import LambdaCallback
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Masking, Embedding, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot, text_to_word_sequence
from keras.utils import to_categorical
from keras.utils.data_utils import get_file
import numpy as np
import random
import matplotlib.pyplot as plt
from keras_batch_generator import KerasBatchGenerator
from utilities import *
import h5py

file_path = "resources\Short Stories.txt"
data_path = "./data/sherlock-training-data.pkl"
model_path = "./data/keras-sherlock-language-model.hdf5"
data = load_training_data(data_path)

x_train = data["x_train_pad"]
y_train = data["y_train_pad"]
word_to_index = data["word_to_index"]
index_to_word = data["index_to_word"]
vocabulary = data["vocabulary"]
num_sentences = data["num_sentences"]
max_input_len = data["max_input_len"]

print("Number of Sentences: ", num_sentences)
print("Max Sentence length: ", max_input_len)

# Parameters
vocabulary_size = 8000
hidden_layer = 100
learning_rate = 0.001
num_epoch = 5
batch_size = 20

training_data_generator = KerasBatchGenerator(x_train, y_train, max_input_len, num_sentences, batch_size, vocabulary_size, skip_step=max_input_len)
evaluate_data_generator = KerasBatchGenerator(x_train, y_train, max_input_len, 10, batch_size, vocabulary_size, skip_step=max_input_len)

# Build the model
print('Build model...')
model = Sequential()
model.add(Masking(mask_value=0, input_shape=(max_input_len, )))
model.add(Embedding(vocabulary_size, hidden_layer, input_length=max_input_len))
model.add(LSTM(hidden_layer, return_sequences=True))
model.add(TimeDistributed(Dense(vocabulary_size)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
print(model.summary())

# Train the model
print("Training model...")
history = model.fit_generator(training_data_generator.generate(), steps_per_epoch=num_sentences/batch_size, epochs=num_epoch)
model.save(model_path, overwrite=True)

# List all data in history
print(history.history.keys())

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

# Final evaluation of the model
scores = model.evaluate_generator(evaluate_data_generator.generate(), steps=10)
print(scores)
print(model.metrics_names)
# print("Accuracy: %.2f%%" % (scores[1]*100))



# model = load_model(model_path + '.hdf5')
#
# unknown_token = "UNKNOWN_TOKEN"
# sentence_start_token = "SENTENCE_START"
# sentence_end_token = "SENTENCE_END"
#
# # We start the sentence with the start token
# new_sentence = np.zeros((128, max_input_len - 1))
# new_sentence = np.insert(new_sentence, 0, word_to_index[sentence_start_token], axis=1)
# print(new_sentence.shape)
# print(new_sentence[0])
# word = ""
# sentence = ""
#
# # while not word == sentence_end_token:
# preds = model.predict_proba(new_sentence, batch_size=128, verbose=1)
#
# for x in range(0, len(preds[0])):
#     word = index_to_word[x]
#     sentence += word
#     if word == sentence_end_token:
#         break
# print(sentence)