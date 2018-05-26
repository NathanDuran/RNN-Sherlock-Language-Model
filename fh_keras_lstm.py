import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress tensorflow warnings

import math
from keras.layers import Dense, Activation, Embedding, TimeDistributed
from keras.layers import LSTM
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras_batch_generator import KerasBatchGenerator
from utilities import *

data_path = "/data/sherlock-training-data.pkl"
model_path = "/model/sherlock-language-model keras.hdf5"
output_path = "/output/sherlock-language-model keras.hdf5"

# Load data
data = load_training_data(data_path)
x = data["x_padded"]
y = data["y_padded"]
word_to_index = data["word_to_index"]
index_to_word = data["index_to_word"]
vocabulary = data["vocabulary"]
num_sentences = data["num_sentences"]
max_input_len = data["max_input_len"]

# Parameters
vocabulary_size = 15000
hidden_layer = 150
learning_rate = 0.001
num_epoch = 1
batch_size = 50
test_split = 0.02

print("------------------------------------")
print("Using parameters...")
print("Vocabulary size: ", vocabulary_size)
print("Number of Sentences: ", num_sentences)
print("Max Sentence length: ", max_input_len)
print("Batch size: ", batch_size)
print("Hidden layer size: ", hidden_layer)
print("learning rate: ", learning_rate)
print("Epochs: ", num_epoch)

# Split training and test sets
num_test_samples = math.ceil(num_sentences * test_split)
x_test = x[:num_test_samples]
y_test = y[:num_test_samples]
x_train = x[num_test_samples:]
y_train = y[num_test_samples:]

# Create data generators
training_data_generator = KerasBatchGenerator(x_train, y_train, max_input_len, len(x_train), batch_size, vocabulary_size)
test_data_generator = KerasBatchGenerator(x_test, y_test, max_input_len, len(x_test), batch_size, vocabulary_size)
evaluate_data_generator = KerasBatchGenerator(x, y, max_input_len, 100, batch_size, vocabulary_size)

# Build the model
print("------------------------------------")
print('Build model...')
model = Sequential()
model.add(Embedding(vocabulary_size, hidden_layer, input_length=max_input_len, mask_zero=True))
model.add(LSTM(hidden_layer, return_sequences=True))
model.add(TimeDistributed(Dense(vocabulary_size, input_shape=(max_input_len, hidden_layer))))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Load the model
print("------------------------------------")
print('Load model...')
if os.path.isfile(model_path):
    print("Loaded model from {}".format(model_path))
    model = load_model(model_path)

print(model.summary())

# Keep only a single model, the best over test accuracy.
checkpoint = ModelCheckpoint(output_path, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='max')

# Train the model
print("------------------------------------")
print("Training model...")
history = model.fit_generator(training_data_generator.generate(), steps_per_epoch=len(x_train)/batch_size, validation_data=test_data_generator.generate(), validation_steps=len(x_test)/batch_size, epochs=num_epoch, callbacks=[checkpoint])
model.save("/models/sherlock-language-model keras.hdf5", overwrite=True)

# Evaluate the model
print("------------------------------------")
print("Evaluating model...")

# Validation set
scores = model.evaluate_generator(evaluate_data_generator.generate(), steps=10)
print("Validation data: ")
print("Loss: ", scores[0], " Accuracy: ", scores[1])

# Generate a sentence
keras_generate_sentence(model, max_input_len, word_to_index, index_to_word)
