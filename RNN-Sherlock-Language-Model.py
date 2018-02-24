import timeit
import numpy as np
import RNN
import ProcessFile
from Utilities import *

# Parameters
vocabulary_size = 8000
hidden_layer = 100
learning_rate = 0.005
num_epoch = 10

# File Paths
file_path = "resources\The Adventures of Sherlock Holmes (No Titles).txt"
model_path = "./data/rnn-sherlock-language-model hidden_dimension=" + str(hidden_layer) + "  word_dimensions=" + str(vocabulary_size) + ".npz"
data_path = "./data/rnn-sherlock-training-data.pkl"
data = None


def process_file():
    # Process input file and create training sets
    file_data = ProcessFile.ProcessFile(vocabulary_size, file_path)
    print("Processing file: ", file_path)

    # Tokenize the text into sentences
    file_data.tokenize_to_sentences()
    print("Parsed %d sentences." % (len(file_data.sentences)))

    # Tokenize the sentences into words
    file_data.tokenize_to_words()
    print("Found %d unique word tokens." % len(file_data.word_frequency.items()))

    # Generate word to index and index to words
    file_data.generate_word_indexes()
    print("Using vocabulary size %d." % vocabulary_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (file_data.vocabulary[-1][0], file_data.vocabulary[-1][1]))
    print("Example sentence after Pre-processing: '%s'" % file_data.tokenized_sentences[0])

    # Create the training data
    file_data.create_training_data()

    return file_data


# Train a model using Stochastic Gradient Descent
def train(num_examples=-1):
    time_taken = timeit.Timer(lambda: model.train_with_sgd(x_train[:num_examples], y_train[:num_examples], learning_rate, num_epoch)).timeit(number=1)
    print("Time taken (in seconds) for ", num_epoch, " epoch over training data : ", time_taken)
    # 100 training exampls for 50 epoch took 719 seconds


def generate_sentence():
    sentence = model.generate_sentence(word_to_index, index_to_word)
    print("Generated sentence: " + str(sentence))
    return sentence


# Claculate the current loss/error of the model
def calc_loss(number_trainin_examples=1000):
    # Test loss function (limit to 1000 examples to save time)
    print("Expected Loss for random predictions: %f" % np.log(vocabulary_size))
    print("Actual loss: %f" % model.calculate_loss(x_train[:number_trainin_examples], y_train[:number_trainin_examples]))


# Test forward propogation and predictions for one sentence
def test_predictions(sentence_index=0):
    # For the input sentence the model outputs vocabulary_size vector for each word
    output, hidden_state = model.forward_propagation(x_train[sentence_index])
    print("Output dimensions ", output.shape)
    print(output)

    # The highest probability predictions for each word
    predictions = model.predict(x_train[sentence_index])
    print(predictions.shape)
    print(predictions)


# Test Stochastic Gradient Descent for one sentence
def test_sgd(sentence_index=0):
    time_taken = timeit.Timer(lambda: model.sgd_step(x_train[sentence_index], y_train[sentence_index], learning_rate)).timeit(number=1)
    print("Time taken (in seconds) for one step over  a sentence of training data : ", time_taken)


def save_data():
    save_training_data(data_path, data)


def load_data():
    try:
        load_training_data(data_path, data)
    except FileNotFoundError as err:
        print("No saved training data found!")


def save_model():
    save_model_parameters(model_path, model)


def load_model():
    try:
        load_model_parameters(model_path, model)
    except FileNotFoundError as err:
        print("No model file found!")

# Create RNN
np.random.seed(10)
model = RNN.RNN(model_path, vocabulary_size, hidden_layer)

# data = ProcessFile.ProcessFile(vocabulary_size, file_path)

# Load training data
if data is None:
    load_data()
else:
    try:
        data = process_file()
        save_data()
    except FileNotFoundError as err:
        print("No file to process found!")

# Training data
x_train = data.x_train
y_train = data.y_train
word_to_index = data.word_to_index
index_to_word = data.index_to_word


# test_predictions()

# test_sgd()

# calc_loss()

# train()

# save_model()

load_model()

generate_sentence()


