import timeit
import rnn
import lstm
from utilities import *

# Parameters
model_type = "lstm"
vocabulary_size = 15000
hidden_layer = 150
learning_rate = 0.005
num_epoch = 10

# File Paths

data_path = "./data/sherlock-training-data.pkl"
model_path = "./models/sherlock-language-model " + model_type + ".npz"


# Train a model using Stochastic Gradient Descent
def train(num_examples):
    time_taken = timeit.Timer(lambda: model.train_with_sgd(x_train[:num_examples], y_train[:num_examples], learning_rate, num_epoch)).timeit(number=1)
    print("Time taken for ", num_epoch, " epoch over ", num_examples, " training data: ", time_taken, " seconds")


# Calculate the current loss/error of the model
def calc_loss(number_training_examples=1000):
    # Test loss function (limit to 1000 examples to save time)
    print("Expected Loss for random predictions: ", np.log(vocabulary_size))
    print("Actual loss: ", model.calculate_loss(x_train[:number_training_examples], y_train[:number_training_examples]))


# Test forward propagation and predictions for one sentence
def test_predictions(sentence_index=0):
    # For the input sentence the model outputs vocabulary_size vector for each word
    output, hidden_state, cell_state = model.forward_propagation(x_train[sentence_index])
    print("Output dimensions: ", output.shape)
    print(output)

    # The highest probability predictions for each word
    predictions = model.predict(x_train[sentence_index])
    print("Prediction dimensions: ", predictions.shape)
    print(predictions)


# Test Stochastic Gradient Descent for one sentence
def test_sgd(sentence_index=0):
    time_taken = timeit.Timer(lambda: model.sgd_step(x_train[sentence_index], y_train[sentence_index], learning_rate)).timeit(number=1)
    print("Time taken for one step over a sentence of training data: ", time_taken, " seconds")


# Build model
def build_model(type):
    np.random.seed(10)
    if type is "rnn":
        print("Creating RNN Model")
        model = rnn.RNN(model_path, vocabulary_size, hidden_layer)
    else:
        print("Creating LSTM Model")
        model = lstm.LSTM(model_path, vocabulary_size, hidden_layer)
    return model


# Create model
model = build_model(model_type)

# Load model parameters
model = load_model_parameters(model_path, model, type)

# Load training data
data = load_training_data(data_path)
x_train = data["x_train"]
y_train = data["y_train"]
word_to_index = data["word_to_index"]
index_to_word = data["index_to_word"]

# Test model
#test_predictions()
#test_sgd()
#calc_loss()

# Train model - use len(x_train) for full dataset
# train(len(x_train))
train(100)

# Save model after training
save_model_parameters(model_path, model, type)

# Generate a random sentence using the current model
sentence_tokens = model.generate_sentence(word_to_index, index_to_word)
sentence = tokens_to_sentence(sentence_tokens)
print("Generated sentence: " + sentence)




