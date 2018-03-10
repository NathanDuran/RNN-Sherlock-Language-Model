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
        self.hidden_weights_g = np.random.uniform(-np.sqrt(1. / self.hidden_dimension), np.sqrt(1. / self.hidden_dimension),
                                                (self.hidden_dimension, self.hidden_dimension))
        self.hidden_weights_i = np.random.uniform(-np.sqrt(1. / self.hidden_dimension), np.sqrt(1. / self.hidden_dimension),
                                                (self.hidden_dimension, self.hidden_dimension))
        self.hidden_weights_f = np.random.uniform(-np.sqrt(1. / self.hidden_dimension), np.sqrt(1. / self.hidden_dimension),
                                                (self.hidden_dimension, self.hidden_dimension))
        self.hidden_weights_o = np.random.uniform(-np.sqrt(1. / self.hidden_dimension), np.sqrt(1. / self.hidden_dimension),
                                                (self.hidden_dimension, self.hidden_dimension))

        # Bias vector for hidden_layer
        self.bias_g = np.random.uniform(0.5, 1, self.hidden_dimension)
        self.bias_i = np.random.uniform(0.5, 1, self.hidden_dimension)
        self.bias_f = np.random.uniform(0.5, 1, self.hidden_dimension)
        self.bias_o = np.random.uniform(0.5, 1, self.hidden_dimension)

        # Output weights (vocabulary_size, hidden_layer)
        self.output_weights = np.random.uniform(-np.sqrt(1. / self.hidden_dimension), np.sqrt(1. / self.hidden_dimension),
                                                (self.word_dimension, self.hidden_dimension))

        self.bias_output = np.random.uniform(0.5, 1, self.word_dimension)



        # print(self.input_weights.shape)
        # print(self.output_weights.shape)
        # print(self.hidden_weights.shape)

    ##### Stochastic Gradient Descent #####
    # Train a RNN for the given number of epochs using SGD.
    # For each epoch calls sgd_step for each sentence in the training set
    #
    # - model: The RNN model instance
    # - x_train: The training data set
    # - y_train: The training data labels
    # - learning_rate: Initial learning rate for SGD
    # - num_epoch: Number of times to iterate through the complete dataset
    # - evaluateLossAfter: Evaluate the loss after this many epochs
    def train_with_sgd(model, x_train, y_train, learning_rate=0.005, num_epoch=100, evaluate_loss_after=5):

        # Timer
        current_time = time.asctime(time.localtime(time.time()))
        print("Training started at : " + current_time + " for " + str(num_epoch) + " epochs")

        # We keep track of the losses so we can plot them later
        losses = []
        num_examples_seen = 0

        for epoch in range(1, num_epoch + 1):

            # For each training example...
            for i in range(len(y_train)):
                # One SGD step
                model.sgd_step(x_train[i], y_train[i], learning_rate)
                num_examples_seen += 1

            # Display current epoch
            print("%s: Number of examples seen = %d epoch = %d" % (current_time, num_examples_seen, epoch))

            # Optionally evaluate the loss
            if epoch % evaluate_loss_after == 0:

                loss = model.calculate_loss(x_train, y_train)
                losses.append((num_examples_seen, loss))
                print("%s: Loss after number of examples seen = %d epoch = %d: %f" % (
                    current_time, num_examples_seen, epoch, loss))

                # Adjust the learning rate if loss increases
                if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                    learning_rate = learning_rate * 0.5
                    print("Setting learning rate to %f" % learning_rate)
                    sys.stdout.flush()

                    # Saving model parameters
                    save_model_parameters(model.model_path, model)

    # Performs one step of SGD.
    # Calls back_prop_through_time for the input sentence
    #
    # - x_sent: The current sentence
    # - y_sent: The current sentence labels
    # - learning_rate: Initial learning rate for SGD
    def sgd_step(self, x_sent, y_sent, learning_rate):

        # Calculate the gradients
        gradients = self.back_prop_through_time(x_sent, y_sent)

        # Change parameters according to gradients and learning rate
        self.input_weights_g -= learning_rate * gradients["in_g"]
        self.input_weights_i -= learning_rate * gradients["in_i"]
        self.input_weights_f -= learning_rate * gradients["in_f"]
        self.input_weights_o -= learning_rate * gradients["in_o"]

        # self.hidden_weights_g -= learning_rate * gradients["h_g"]
        # self.hidden_weights_i -= learning_rate * gradients["h_i"]
        # self.hidden_weights_f -= learning_rate * gradients["h_f"]
        # self.hidden_weights_o -= learning_rate * gradients["h_o"]

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
    # Calls forward_propagation for the input sentence
    #
    # - x_sent: The current sentence
    # - y_sent: The current sentence labels
    def back_prop_through_time(self, x_sent, y_sent):

        # Length of sentence
        sentence_length = len(y_sent)

        # Perform forward propagation
        output, h, c = self.forward_propagation(x_sent)

        # We accumulate the gradients in these variables
        gradient_in_g = np.zeros(self.input_weights_g.shape, dtype=np.float64)
        gradient_in_i = np.zeros(self.input_weights_i.shape)
        gradient_in_f = np.zeros(self.input_weights_f.shape)
        gradient_in_o = np.zeros(self.input_weights_o.shape)

        gradient_hidden_g = np.zeros(self.hidden_weights_g.shape, dtype=np.float64)
        gradient_hidden_i = np.zeros(self.hidden_weights_i.shape, dtype=np.float64)
        gradient_hidden_f = np.zeros(self.hidden_weights_f.shape, dtype=np.float64)
        gradient_hidden_o = np.zeros(self.hidden_weights_o.shape, dtype=np.float64)

        gradient_bias_g = np.zeros(self.bias_g.shape)
        gradient_bias_i = np.zeros(self.bias_i.shape)
        gradient_bias_f = np.zeros(self.bias_f.shape)
        gradient_bias_o = np.zeros(self.bias_o.shape)

        gradient_out_weights = np.zeros(self.output_weights.shape)
        gradient_bias_out = np.zeros(self.bias_output.shape)

        gradient_h_next = np.zeros(h.shape)
        gradient_h = np.zeros(h.shape)
        gradient_c_next = np.zeros(c.shape)
        gradient_c = np.zeros(c.shape)


        # Change in output
        output_delta = output
        # Effectively ignore the SENTENCE_END token by adding -1
        output_delta[np.arange(len(y_sent)), y_sent] -= 1.

        # For each word from the end of the sentence backwards...
        for word in reversed(np.arange(sentence_length)):

            # Gradient of output_weights is the matrix multiplication (outer product) of the,
            # output change vector and hidden state vector (.T = transpose)
            gradient_out_weights += np.outer(output_delta[word], h[word].T)
            gradient_bias_out += output_delta[word]
            # print("GRAD H SHAPE " + str(gradient_h[word].shape))
            # print("OUT D SHAPE" + str(output_delta[word].shape))
            # print("GRAD OUT W SHAPE " + str(self.output_weights[word].T.shape))
            # Gradient for h
            gradient_h[word] = np.dot(output_delta[word], self.output_weights)
            gradient_h[word] += gradient_h_next[word]


            # Gradient for c
            gradient_c[word] = self.hidden_weights_o[word] * gradient_h[word] * self.dtanh(c[word])
            gradient_c[word] += gradient_c_next[word]


            # Gradient for hidden_weights_o ??????????????
            gradient_hidden_o += np.tanh(c[word]) * gradient_h[word]
            gradient_hidden_o += np.dot(self.dsigmoid(self.hidden_weights_o[word]), gradient_hidden_o[word])

            # Gradient for hidden_weights_f
            gradient_hidden_f += np.dot(c[word - 1], gradient_c[word])
            gradient_hidden_f += np.dot(self.dsigmoid(self.hidden_weights_f[word]), gradient_hidden_f[word])

            # Gradient for gradient_hidden_i
            gradient_hidden_i += np.dot(self.hidden_weights_g[word], gradient_c[word])
            gradient_hidden_i += np.dot(self.dsigmoid(self.hidden_weights_i[word]), gradient_hidden_i[word])

            # Gradient for hg
            gradient_hidden_g += np.dot(self.hidden_weights_i[word], gradient_c[word])
            gradient_hidden_g += np.dot(self.dtanh(self.hidden_weights_g[word]), gradient_hidden_g[word])

            # print("H W SHAPE " + str(gradient_in_o.shape))
            # print("G H SHAPE" + str( x_sent))
            # print("G O H SHAPE" + str(gradient_hidden_o.shape))
            # Gate gradients, just a normal fully connected layer gradient
            # gradient_in_o[word] += np.dot(x_sent, gradient_hidden_o[word])
            gradient_in_o += np.dot(gradient_in_o[:, x_sent[word]], gradient_hidden_o[word])
            gradient_bias_o += gradient_hidden_o[word]
            dXo = np.dot(gradient_hidden_o, self.input_weights_o.T[word])

            gradient_in_f += np.dot(gradient_in_f[:, x_sent[word]].T, gradient_hidden_f[word])
            gradient_bias_f += gradient_hidden_f[word]
            dXf = np.dot(gradient_hidden_f, self.input_weights_f.T[word])

            gradient_in_i += np.dot(gradient_in_i[:, x_sent[word]].T, gradient_hidden_i[word])
            gradient_bias_i += gradient_hidden_i[word]
            dXi = np.dot(gradient_hidden_i, self.input_weights_i.T[word])

            gradient_in_g += np.dot(gradient_in_g[:, x_sent[word]].T, gradient_hidden_g[word])
            gradient_bias_g += gradient_hidden_g[word]
            dXg = np.dot(gradient_hidden_g, self.input_weights_g.T[word])

            # As X was used in multiple gates, the gradient must be accumulated here
            dX = dXo + dXg + dXi + dXf

            # Split the concatenated X, so that we get our gradient of h_old
            # gradient_h_next[word] = dX[:, :self.hidden_dimension]
            gradient_h_next[word] = dX
            # Gradient for c_old in c = hf * c_old + hi * hc
            gradient_c_next = np.dot(self.hidden_weights_f, gradient_c[word])

            # # Gradient of hidden_weights is the matrix multiplication (outer product) of the,
            # # delta (change) vector and the previous hidden state vector
            # gradient_hidden_weights += np.outer(delta, hidden_state[bptt_step - 1])
            #
            # # Gradient of input_weights just update the weight for the current word at this step
            # gradient_in_weights[:, x_sent[bptt_step]] += delta
            #
            # # Update delta for next step
            # delta = self.hidden_weights.T.dot(delta) * (1 - hidden_state[bptt_step - 1] ** 2)

        gradients = dict(in_f=gradient_in_f, in_i=gradient_in_i, in_g=gradient_in_g, in_o=gradient_in_o,
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
    # - hidden_state: at time t, the 'memory' of the network.
    #   It is calculated based on the previous hidden state and the input at the current step..
    #   hidden_state = tanh( (inputWeights * inputSentence[t]) + (hiddenWeights * hiddenState - 1) )
    #
    # - output: is the output at step t.
    #   The next word in a sentence it is a vector of probabilities across the vocabulary.
    #   output = softmax( outputWeights * hiddenState )
    def forward_propagation(self, input_sentence):

        # The total number of time steps
        steps = len(input_sentence)
        # print("SENT SHAPE " + str(input_sentence))
        # During forward propagation we save all hidden states because we need them later.
        # We add one additional element for the initial hidden state, which we set to 0

        # Long and short term memory
        c = np.zeros((steps + 1, self.hidden_dimension))
        h = np.zeros((steps + 1, self.hidden_dimension))


        # The outputs at each time step. Again, we save them for later.
        output = np.zeros((steps, self.word_dimension))

        # For each time step word in the sentence
        for t in np.arange(steps):


            g = np.tanh(self.input_weights_g[:, input_sentence[t]] + np.dot(h[t - 1], self.hidden_weights_g) + self.bias_g)
            i = sigmoid(self.input_weights_i[:, input_sentence[t]] + np.dot(h[t - 1], self.hidden_weights_i) + self.bias_i)
            f = sigmoid(self.input_weights_f[:, input_sentence[t]] + np.dot(h[t - 1], self.hidden_weights_f) + self.bias_f)
            o = sigmoid(self.input_weights_o[:, input_sentence[t]] + np.dot(h[t - 1], self.hidden_weights_o) + self.bias_o)

            c[t] = (f * c[t - 1]) + (i * g)
            h[t] = o * np.tanh(c[t])

            output[t] = self.softmax(np.dot(self.output_weights, h[t]) + self.bias_output)

        return [output, h, c]

    # Returns the word index with the highest probability for each word in the sentence
    #
    # - input_sentence: The current sentence
    def predict(self, input_sentence):

        output, h, c = self.forward_propagation(input_sentence)
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
            output, h, c = self.forward_propagation(x_train[i])

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
