import os

from tflearn.data_utils import pad_sequences

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import math
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, Embedding, TimeDistributed
from keras.layers import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras_batch_generator import KerasBatchGenerator
from utilities import *

file_path = "resources\Complete Works.txt"
data_path = "./data/sherlock-training-data.pkl"
model_path = "./data/sherlock-language-model keras.hdf5"

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
num_epoch = 10
batch_size = 20
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
print(model.summary())

# Train the model
print("------------------------------------")
print("Training model...")
history = model.fit_generator(training_data_generator.generate(), steps_per_epoch=len(x_train)/batch_size, validation_data=test_data_generator.generate(), validation_steps=len(x_test)/batch_size, epochs=num_epoch)
model.save(model_path, overwrite=True)

# Plot history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Evaluate the model
print("------------------------------------")
print("Evaluating model...")
model = load_model(model_path)

# Validation set
scores = model.evaluate_generator(evaluate_data_generator.generate(), steps=10)
print("Validation data: ")
print("Loss: ", scores[0], " Accuracy: ", scores[1])


# Generate a sentence
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# We start the sentence with the start token
sentence_tokens = np.zeros((1, max_input_len), dtype=int)
sentence_tokens[0][0] = word_to_index[sentence_start_token]

sentence = []
# sampled_word = word_to_index[sentence_start_token]
word_index = 1

# Repeat until we get an end token
for i in range(0, max_input_len):

    # Generate some word predictions
    sentence_probs = model.predict_proba(sentence_tokens, batch_size=1, verbose=1)
    word_probs = sentence_probs[0][i]

    sampled_word = np.argmax(word_probs)

    # We don't want to sample unknown words
    if sampled_word == word_to_index[unknown_token]:
        # Remove the unknown token and get second most likely word
        word_probs = np.delete(word_probs, sampled_word)
        sampled_word = np.argmax(word_probs)

    if sampled_word == word_to_index[sentence_end_token]:
        break

    sentence.append(index_to_word[sampled_word])
    sentence_tokens[0][word_index] = sampled_word
    word_index += 1

print(sentence)
