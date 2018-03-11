import time
import sys
from scipy.special import expit as sigmoid
from utilities import *


class LSTM:

    ##### Initialise #####
    def __init__(self, model_path, vocabulary_size, hidden_layer, back_prop_through_time_truncate=4):

        # Assign instance variables
        self.model_path = model_path
        self.word_dimension = vocabulary_size
        self.hidden_dimension = hidden_layer
        self.bptt_truncate = back_prop_through_time_truncate

        # Randomly initialize the network parameters (weights)
        # in range [- (1/sqrt of n), 1/sqrt of n)] where n = number of incoming connections from previous layer

        # Input weights (vocabulary_size, hidden_layer)
        self.input_weights_g = np.random.uniform(-np.sqrt(1. / self.word_dimension), np.sqrt(1. / self.word_dimension),
                                                 (self.hidden_dimension, self.word_dimension))
        self.input_weights_i = np.random.uniform(-np.sqrt(1. / self.word_dimension), np.sqrt(1. / self.word_dimension),
                                                 (self.hidden_dimension, self.word_dimension))
        self.input_weights_f = np.random.uniform(-np.sqrt(1. / self.word_dimension), np.sqrt(1. / self.word_dimension),
                                                 (self.hidden_dimension, self.word_dimension))
        self.input_weights_o = np.random.uniform(-np.sqrt(1. / self.word_dimension), np.sqrt(1. / self.word_dimension),
                                                 (self.hidden_dimension, self.word_dimension))

        # Hidden weights (hidden_layer, hidden_layer)
        self.hidden_weights_g = np.random.uniform(-np.sqrt(1. / self.hidden_dimension),
                                                  np.sqrt(1. / self.hidden_dimension),
                                                  (self.hidden_dimension, self.hidden_dimension))
        self.hidden_weights_i = np.random.uniform(-np.sqrt(1. / self.hidden_dimension),
                                                  np.sqrt(1. / self.hidden_dimension),
                                                  (self.hidden_dimension, self.hidden_dimension))
        self.hidden_weights_f = np.random.uniform(-np.sqrt(1. / self.hidden_dimension),
                                                  np.sqrt(1. / self.hidden_dimension),
                                                  (self.hidden_dimension, self.hidden_dimension))
        self.hidden_weights_o = np.random.uniform(-np.sqrt(1. / self.hidden_dimension),
                                                  np.sqrt(1. / self.hidden_dimension),
                                                  (self.hidden_dimension, self.hidden_dimension))

        # Bias vector for hidden_layer
        self.bias_g = np.random.uniform(0.5, 1, self.hidden_dimension)
        self.bias_i = np.random.uniform(0.5, 1, self.hidden_dimension)
        self.bias_f = np.random.uniform(0.5, 1, self.hidden_dimension)
        self.bias_o = np.random.uniform(0.5, 1, self.hidden_dimension)

        # Output weights (vocabulary_size, hidden_layer)
        self.output_weights = np.random.uniform(-np.sqrt(1. / self.hidden_dimension),
                                                np.sqrt(1. / self.hidden_dimension),
                                                (self.word_dimension, self.hidden_dimension))

        self.bias_output = np.random.uniform(0.5, 1, self.word_dimension)

    ##### Stochastic Gradient Descent #####
    # Train a LSTM for the given number of epochs using SGD.
    # For each epoch calls sgd_step for each sentence in the training set
    #
    # - model: The LSTM model instance
    # - x_train: The training data set
    # - y_train: The training data labels
    # - learning_rate: Initial learning rate for SGD
    # - num_epoch: Number of times to iterate through the complete dataset
    # - evaluate_loss_after: Evaluate the loss after this many epochs
    def train_with_sgd(model, x_train, y_train, learning_rate=0.005, num_epoch=100, evaluate_loss_after=5):

        # Time
        print("Training started: " + time.asctime(time.localtime(time.time())) + " for ", num_epoch, " epochs")

        # Keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0

        for epoch in range(1, num_epoch + 1):

            # For each training example...
            for i in range(len(y_train)):
                # One SGD step
                model.sgd_step(x_train[i], y_train[i], learning_rate)
                num_examples_seen += 1

            # Display current epoch
            print(time.asctime(time.localtime(time.time())) +
                  ": Number of examples seen = ", num_examples_seen, "epoch = ", epoch)

            # Optionally evaluate the loss
            if epoch % evaluate_loss_after == 0:

                loss = model.calculate_loss(x_train, y_train)
                losses.append((num_examples_seen, loss))
                print(time.asctime(time.localtime(time.time())) +
                      ": Loss after number of examples seen = ", num_examples_seen, "epoch = ", epoch, " : ", loss)

                # Adjust the learning rate if loss increases
                if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                    learning_rate = learning_rate * 0.5
                    print("Setting learning rate to ", learning_rate)
                    sys.stdout.flush()

                # Save model parameters
                save_model_parameters(model.model_path, model, "lstm")

    # Performs one step of SGD.
    # Calls back_prop_through_time for the input sentence
    #
    # - x_sent: The current sentence
    # - y_sent: The current sentence labels
    # - learning_rate: Initial learning rate for SGD
    def sgd_step(self, x_sent, y_sent, learning_rate):

        # Forward propagate over an input sentence
        output, hidden_state, cell_state = self.forward_propagation(x_sent)

        # Calculate the gradients and propagate backwards through cells
        gradients = self.back_prop_through_time(x_sent, y_sent, output, hidden_state, cell_state)

        # Change parameters according to gradients and learning rate
        self.input_weights_g -= learning_rate * gradients["in_g"]
        self.input_weights_i -= learning_rate * gradients["in_i"]
        self.input_weights_f -= learning_rate * gradients["in_f"]
        self.input_weights_o -= learning_rate * gradients["in_o"]

        self.bias_g -= learning_rate * gradients["b_g"]
        self.bias_i -= learning_rate * gradients["b_i"]
        self.bias_f -= learning_rate * gradients["b_f"]
        self.bias_o -= learning_rate * gradients["b_o"]

        self.output_weights -= learning_rate * gradients["out"]
        self.bias_output -= learning_rate * gradients["out_b"]

    ##### Back Propagation Through Time #####
    # Calculate the gradients for the input, output and hidden weights.
    # Because the parameters are shared by all time steps in the network,
    # the gradient at each output depends not only on the calculations of the current time step,
    # but also the previous time steps.
    # Returns a dictionary object of all calculated gradients
    #
    # - x_sent: The current sentence
    # - y_sent: The current sentence labels
    # - output: The output probabilities from forward propagation
    # - hidden_state: The current hidden state (short term memory) of the cell
    # - cell_state: The current cell state (long term memory) of the cell
    def back_prop_through_time(self, x_sent, y_sent, output, hidden_state, cell_state):

        # Length of sentence
        sentence_length = len(y_sent)

        # We accumulate the gradients in these variables
        gradient_input_g = np.zeros(self.input_weights_g.shape)
        gradient_input_i = np.zeros(self.input_weights_i.shape)
        gradient_input_f = np.zeros(self.input_weights_f.shape)
        gradient_input_o = np.zeros(self.input_weights_o.shape)

        gradient_hidden_g = np.zeros(self.hidden_weights_g.shape)
        gradient_hidden_i = np.zeros(self.hidden_weights_i.shape)
        gradient_hidden_f = np.zeros(self.hidden_weights_f.shape)
        gradient_hidden_o = np.zeros(self.hidden_weights_o.shape)

        gradient_bias_g = np.zeros(self.bias_g.shape)
        gradient_bias_i = np.zeros(self.bias_i.shape)
        gradient_bias_f = np.zeros(self.bias_f.shape)
        gradient_bias_o = np.zeros(self.bias_o.shape)

        gradient_out_weights = np.zeros(self.output_weights.shape)
        gradient_bias_out = np.zeros(self.bias_output.shape)

        # Need to accumulate hidden and cell state gradients so keep the previous values after each iteration
        gradient_hidden_prev = np.zeros(hidden_state.shape)
        gradient_hidden = np.zeros(hidden_state.shape)
        gradient_cell_prev = np.zeros(cell_state.shape)
        gradient_cell = np.zeros(cell_state.shape)

        # Change in output
        output_delta = output
        # Effectively ignore the SENTENCE_END token by adding -1
        output_delta[np.arange(len(y_sent)), y_sent] -= 1.

        # For each word from the end of the sentence backwards...
        for t in reversed(np.arange(sentence_length)):

            # Gradient of output weights is the matrix multiplication (outer product) of the,
            # output change vector and hidden state vector (.T = transpose)
            gradient_out_weights += np.outer(output_delta[t], hidden_state[t].T)
            gradient_bias_out += output_delta[t]

            # Gradient for hidden_state
            gradient_hidden[t] = np.dot(output_delta[t], self.output_weights)
            gradient_hidden[t] += gradient_hidden_prev[t]

            # Gradient for cell_state
            gradient_cell[t] = self.hidden_weights_o[t] * gradient_hidden[t] * self.dtanh(cell_state[t])
            gradient_cell[t] += gradient_cell_prev[t]

            # Gradients for hidden_weights_o, hidden_weights_f, hidden_weights_i, hidden_weights_g
            gradient_hidden_o += np.tanh(cell_state[t]) * gradient_hidden[t]
            gradient_hidden_o += np.dot(self.dsigmoid(self.hidden_weights_o[t]), gradient_hidden_o[t])

            gradient_hidden_f += np.dot(cell_state[t - 1], gradient_cell[t])
            gradient_hidden_f += np.dot(self.dsigmoid(self.hidden_weights_f[t]), gradient_hidden_f[t])

            gradient_hidden_i += np.dot(self.hidden_weights_g[t], gradient_cell[t])
            gradient_hidden_i += np.dot(self.dsigmoid(self.hidden_weights_i[t]), gradient_hidden_i[t])

            gradient_hidden_g += np.dot(self.hidden_weights_i[t], gradient_cell[t])
            gradient_hidden_g += np.dot(self.dtanh(self.hidden_weights_g[t]), gradient_hidden_g[t])

            # Gradients for gradient_input_o, gradient_input_f, gradient_input_i, gradient_input_g
            gradient_input_o += np.dot(gradient_input_o[:, x_sent[t]], gradient_hidden_o[t])
            gradient_bias_o += gradient_hidden_o[t]

            gradient_input_f += np.dot(gradient_input_f[:, x_sent[t]].T, gradient_hidden_f[t])
            gradient_bias_f += gradient_hidden_f[t]

            gradient_input_i += np.dot(gradient_input_i[:, x_sent[t]].T, gradient_hidden_i[t])
            gradient_bias_i += gradient_hidden_i[t]

            gradient_input_g += np.dot(gradient_input_g[:, x_sent[t]].T, gradient_hidden_g[t])
            gradient_bias_g += gradient_hidden_g[t]

            # Gate gradients
            gradient_gate_o = np.dot(gradient_hidden_o, self.input_weights_o.T[t])

            gradient_gate_f = np.dot(gradient_hidden_f, self.input_weights_f.T[t])

            gradient_gate_i = np.dot(gradient_hidden_i, self.input_weights_i.T[t])

            gradient_gate_g = np.dot(gradient_hidden_g, self.input_weights_g.T[t])

            # Keep hidden state gradients for next iteration
            gradient_hidden_prev[t] = gradient_gate_o + gradient_gate_g + gradient_gate_i + gradient_gate_f

            # Keep cell state gradient for next iteration
            gradient_cell_prev = np.dot(self.hidden_weights_f, gradient_cell[t])


        gradients = dict(in_f=gradient_input_f, in_i=gradient_input_i, in_g=gradient_input_g, in_o=gradient_input_o,
                         h_f=gradient_hidden_f, h_i=gradient_hidden_i, h_g=gradient_hidden_g, h_o=gradient_hidden_o,
                         b_f=gradient_bias_f, b_i=gradient_bias_i, b_g=gradient_bias_g, b_o=gradient_bias_o,
                         out=gradient_out_weights, out_b=gradient_bias_out)
        return gradients

    ##### Forward Propogation #####
    # Performs one iteration over the input_sentence.
    # Returns the output probabilities and the hidden_state.
    #
    # - input_sentence: The matrix of one hot vectors for all words in the input.
    # - input_sentence[t]: The a one hot vector for the word at that time step.
    #
    # - cell_state: at time t, the long-term 'memory' of the network.
    #   It is calculated based on the forget_gate state and the previous cell_state
    #   plus the input_gate and the normal_gate at the current step..
    #   cell_state = (forget_gate * tanh( cell_state[t - 1] )) + (input_gate * normal_gate)
    #
    # - hidden_state: at time t, the short-term 'memory' of the network.
    #   It is calculated based on the output_gate and the cell_state at the current step..
    #   hidden_state = output_gate * tanh( cell_state[t] )
    #
    # - normal_gate (g): acts like the hidden state of a traditional RNN
    #   It is calculated based on the previous hidden state and the input at the current step plus the bias..
    #   normal_gate = tanh( (input_weights_g * input_sentence[t]) + (hidden_weights_g * hidden_state[t - 1]) + bias_g )
    #
    # - input_gate (i): controls which parts of the normal_gate output should be added to the long-term memory (cell_state)
    #   It is calculated based on the previous hidden state and the input at the current step plus the bias..
    #   normal_gate = sigmoid( (input_weights_i * input_sentence[t]) + (hidden_weights_i * hidden_state[t - 1]) + bias_i )
    #
    # - forget_gate (f): controls which parts of the long-term memory (cell_state) should be erased
    #   It is calculated based on the previous hidden state and the input at the current step plus the bias..
    #   normal_gate = sigmoid( (input_weights_f * input_sentence[t]) + (hidden_weights_f * hidden_state[t - 1]) + bias_f )
    #
    # - output_gate (o): controls which parts of the long-term state (cell_state) should be read and output at this time step
    #   It is calculated based on the previous hidden state and the input at the current step plus the bias..
    #   normal_gate = sigmoid( (input_weights_o * input_sentence[t]) + (hidden_weights_o * hidden_state[t - 1]) + bias_o )
    #
    # - output: is the output at step t.
    #   The next word in a sentence it is a vector of probabilities across the vocabulary.
    #   output = softmax( output_weights * hidden_state[t] + bias_output)
    def forward_propagation(self, input_sentence):

        # The total number of time steps
        steps = len(input_sentence)

        # During forward propagation we save all hidden states (Long and short term memory) because we need them later.
        # We add one additional element for the initial hidden state, which we set to 0
        cell_state = np.zeros((steps + 1, self.hidden_dimension))
        hidden_state = np.zeros((steps + 1, self.hidden_dimension))

        # The outputs at each time step. Again, we save them for later.
        output = np.zeros((steps, self.word_dimension))

        # For each time step word in the sentence
        for t in np.arange(steps):
            # Note that we are indexing inputWeights by input_sentence[t].
            # This is the same as multiplying inputWeights with a one-hot vector.
            # Calculate gates normal_gate, ,i ,f and o at time t
            normal_gate = np.tanh(self.input_weights_g[:, input_sentence[t]] + np.dot(hidden_state[t - 1], self.hidden_weights_g) + self.bias_g)
            input_gate = sigmoid(self.input_weights_i[:, input_sentence[t]] + np.dot(hidden_state[t - 1], self.hidden_weights_i) + self.bias_i)
            forget_gate = sigmoid(self.input_weights_f[:, input_sentence[t]] + np.dot(hidden_state[t - 1], self.hidden_weights_f) + self.bias_f)
            output_gate = sigmoid(self.input_weights_o[:, input_sentence[t]] + np.dot(hidden_state[t - 1], self.hidden_weights_o) + self.bias_o)

            # Calculate hidden and cell state at time t
            cell_state[t] = (forget_gate * cell_state[t - 1]) + (input_gate * normal_gate)
            hidden_state[t] = output_gate * np.tanh(cell_state[t])

            # Calculate output at time t
            output[t] = self.softmax(np.dot(self.output_weights, hidden_state[t]) + self.bias_output)

        return [output, hidden_state, cell_state]

    # Returns the word index with the highest probability for each word in the sentence
    #
    # - input_sentence: The current sentence
    def predict(self, input_sentence):

        output, hidden_state, cell_state = self.forward_propagation(input_sentence)
        return np.argmax(output, axis=1)

    ##### Calculating Loss Function #####
    # Calculate the difference between the predictions and the labeled data.
    # Cross entropy loss, where N = training examples, y = expected word and o = output word predictions
    # loss(y, o) = - sum(y log o )/N
    #
    # - x_train: The training data set
    # - y_train: The training data labels
    def calculate_loss(self, x_train, y_train):

        num_training_examples = 0
        total_loss = 0

        # For each sentence...
        for i in np.arange(len(y_train)):
            # Perform iteration of forward propagation
            output, hidden_state, cell_state = self.forward_propagation(x_train[i])

            # We only care about our prediction of the "correct" words in current sentence
            correct_word_predictions = output[np.arange(len(y_train[i])), y_train[i]]

            # Add to the loss based on how off we were
            total_loss += -1 * np.sum(np.log(correct_word_predictions))

            # Divide the total loss by the number of training examples
            num_training_examples = np.sum((len(i) for i in y_train))

        return total_loss / num_training_examples

    ##### Generate New Sentences #####
    # Generates a random new sentence from a trained model
    # Returns a list of words for generated sentence
    #
    # - word_to_index: Mapping of vocabulary words to index
    # - index_to_word: Mapping of index to vocabulary words
    def generate_sentence(model, word_to_index, index_to_word):

        unknown_token = "UNKNOWN_TOKEN"
        sentence_start_token = "SENTENCE_START"
        sentence_end_token = "SENTENCE_END"

        # We start the sentence with the start token
        new_sentence = [word_to_index[sentence_start_token]]

        # Repeat until we get an end token
        while not new_sentence[-1] == word_to_index[sentence_end_token]:
            next_word_probs = model.forward_propagation(new_sentence)[0]
            sampled_word = word_to_index[unknown_token]

            # We don't want to sample unknown words
            while sampled_word == word_to_index[unknown_token]:
                samples = np.random.multinomial(1, next_word_probs[-1])
                sampled_word = np.argmax(samples)

            new_sentence.append(sampled_word)
        sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
        return sentence_str

    def softmax(self, x):
        xt = np.exp(x - np.max(x))
        return xt / np.sum(xt)

    def dsigmoid(self, x):
        x = sigmoid(x)
        return x * (1 - x)

    def dtanh(self, x):
        x = np.tanh(x)
        return 1 - x * x
