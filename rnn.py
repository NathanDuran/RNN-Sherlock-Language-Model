import time
import sys
from utilities import *


class RNN:

    ##### Initialise #####
    def __init__(self, model_path, vocabulary_size, hidden_layer, back_prop_through_time_truncate=4):

        # Assign instance variables
        self.model_path = model_path
        self.word_dimension = vocabulary_size
        self.hidden_dimension = hidden_layer
        self.bptt_truncate = back_prop_through_time_truncate

        # Randomly initialize the network parameters (weights)
        # in range [- (1/sqrt of n), 1/sqrt of n)] where n = number of incoming connections from previous layer

        # Input weights (hidden_layer, vocabulary_size)
        self.input_weights = np.random.uniform(-np.sqrt(1. / self.word_dimension),
                                               np.sqrt(1. / self.word_dimension),
                                               (self.hidden_dimension, self.word_dimension))
        # Output weights (vocabulary_size, hidden_layer)
        self.output_weights = np.random.uniform(-np.sqrt(1. / self.hidden_dimension),
                                                np.sqrt(1. / self.hidden_dimension),
                                                (self.word_dimension, self.hidden_dimension))
        # Hidden weights (hidden_layer, hidden_layer)
        self.hidden_weights = np.random.uniform(-np.sqrt(1. / self.hidden_dimension),
                                                np.sqrt(1. / self.hidden_dimension),
                                                (self.hidden_dimension, self.hidden_dimension))

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
            print("%s: Number of examples seen = %d epoch = %d" % (
                current_time, num_examples_seen, epoch))

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
        gradient_in_weights, gradient_out_weights, gradient_hidden_weights = self.back_prop_through_time(x_sent, y_sent)

        # Change parameters according to gradients and learning rate
        self.input_weights -= learning_rate * gradient_in_weights
        self.output_weights -= learning_rate * gradient_out_weights
        self.hidden_weights -= learning_rate * gradient_hidden_weights

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
        output, hidden_state = self.forward_propagation(x_sent)

        # We accumulate the gradients in these variables
        gradient_in_weights = np.zeros(self.input_weights.shape)
        gradient_out_weights = np.zeros(self.output_weights.shape)
        gradient_hidden_weights = np.zeros(self.hidden_weights.shape)

        # Change in output
        output_delta = output
        # Effectively ignore the SENTENCE_END token by adding -1
        output_delta[np.arange(len(y_sent)), y_sent] -= 1.

        # For each word from the end of the sentence backwards...
        for word in reversed(np.arange(sentence_length)):

            # Gradient of output_weights is the matrix multiplication (outer product) of the,
            # output change vector and hidden state vector (.T = transpose)
            gradient_out_weights += np.outer(output_delta[word], hidden_state[word].T)

            # Initial delta calculation, the change in weights,
            # the dot product (sum) of ( output change vector * (1 -  hidden state vector ^ 2) )
            delta = self.output_weights.T.dot(output_delta[word]) * (1 - (hidden_state[word] ** 2))

            # Backpropagation through time (for at most self.bptt_truncate steps)
            for bptt_step in reversed(np.arange(max(0, word - self.bptt_truncate), word + 1)):
                # Gradient of hidden_weights is the matrix multiplication (outer product) of the,
                # delta (change) vector and the previous hidden state vector
                gradient_hidden_weights += np.outer(delta, hidden_state[bptt_step - 1])

                # Gradient of input_weights just update the weight for the current word at this step
                gradient_in_weights[:, x_sent[bptt_step]] += delta

                # Update delta for next step
                delta = self.hidden_weights.T.dot(delta) * (1 - hidden_state[bptt_step - 1] ** 2)

        return [gradient_in_weights, gradient_out_weights, gradient_hidden_weights]

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

        # During forward propagation we save all hidden states because we need them later.
        # We add one additional element for the initial hidden state, which we set to 0
        hidden_state = np.zeros((steps + 1, self.hidden_dimension))
        # print("HIDDEN SHAPE " + str(hidden_state.shape))
        #hidden_state[-1] = np.zeros(self.hidden_dimension)
        # print("HIDDEN SHAPE " + str(hidden_state.shape))
        # The outputs at each time step. Again, we save them for later.
        output = np.zeros((steps, self.word_dimension))

        # For each time step...
        for t in np.arange(steps):
            # Note that we are indexing inputWeights by input_sentence[t].
            # This is the same as multiplying inputWeights with a one-hot vector.
            hidden_state[t] = np.tanh(
                self.input_weights[:, input_sentence[t]] + self.hidden_weights.dot(hidden_state[t - 1]))
            output[t] = self.softmax(self.output_weights.dot(hidden_state[t]))

        return [output, hidden_state]

    # Returns the word index with the highest probability for each word in the sentence
    #
    # - input_sentence: The current sentence
    def predict(self, input_sentence):

        output, hidden_state = self.forward_propagation(input_sentence)
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
            output, hidden_state = self.forward_propagation(x_train[i])

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
