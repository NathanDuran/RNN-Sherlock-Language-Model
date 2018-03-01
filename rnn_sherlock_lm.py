import timeit
import datetime
import string
import re
import rnn
from utilities import *

# Parameters
vocabulary_size = 8000
hidden_layer = 100
learning_rate = 0.005
num_epoch = 1

# File Paths
file_path = "resources\Short Stories.txt"
data_path = "./data/rnn-sherlock-training-data.pkl"
model_path = "./data/rnn-sherlock-language-model " \
             + datetime.datetime.today().strftime('%d-%m-%Y') \
             + " hidden_layers=" + str(hidden_layer) \
             + "  vocabulary=" + str(vocabulary_size) + ".npz"


# Train a model using Stochastic Gradient Descent
def train(num_examples):
    time_taken = timeit.Timer(lambda: model.train_with_sgd(x_train[:num_examples], y_train[:num_examples], learning_rate, num_epoch)).timeit(number=1)
    print("Time taken (in seconds) for ", num_epoch, " epoch over " + str(num_examples) + " training data : ", time_taken)


# Generate a random sentence using the current model
def generate_sentence():

    sentence = ""
    sentence_list = model.generate_sentence(word_to_index, index_to_word)

    for i, word in enumerate(sentence_list):

        if word in ["``", "''"]:
            word = '"'

        if word is '"' or word in string.punctuation:
            sentence += word
        else:
            sentence += " " + word

    sentence = re.sub(r'\s([?.!,"](?:\s|$))', r'\1', sentence)
    sentence.lstrip()
    print("Generated sentence: " + sentence)
    return sentence


# Calculate the current loss/error of the model
def calc_loss(number_trainin_examples=1000):
    # Test loss function (limit to 1000 examples to save time)
    print("Expected Loss for random predictions: %f" % np.log(vocabulary_size))
    print("Actual loss: %f" % model.calculate_loss(x_train[:number_trainin_examples], y_train[:number_trainin_examples]))


# Test forward propagation and predictions for one sentence
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
    print("Time taken (in seconds) for one step over a sentence of training data : ", time_taken)


# Save a model file to model_path
def save_model():
    try:
        save_model_parameters(model_path, model)
    except FileNotFoundError as err:
        print("Error saving model! " + str(err))


# Load a model file from model_path
def load_model():
    try:
        load_model_parameters(model_path, model)
    except FileNotFoundError as err:
        print("No model file found! " + str(err))


# Create RNN
np.random.seed(10)
model = rnn.RNN(model_path, vocabulary_size, hidden_layer)

# Load training data
try:
    data = load_training_data(data_path)
except FileNotFoundError as err:
    print("No saved training data found!")


# Training data
x_train = data["x_train"]
y_train = data["y_train"]
word_to_index = data["word_to_index"]
index_to_word = data["index_to_word"]

load_model()

# test_predictions()
# test_sgd()
# calc_loss()

train(len(x_train))

save_model()

generate_sentence()




