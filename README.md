# RNN-Sherlock-Language-Model

A Recurrent Neural Network Language Model trained on ['The Adventures of Sherlock Holmes'](https://sherlock-holm.es/ascii/).

The RNN is an adapted version of the one outlined in [this tutorial](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) by Denny Britz.

The data directory contains pre-processed data set (rnn-sherlock-training-data.pkl) and a pre-trained model. The included model has

To generate a sentence with the included pre-trained model simply run rnn_sherlock_lm.py and call load_model() followed by generate_sentence().


# Example of Generated Sentences


# Included Python Files

process_file.py turns the raw text into training data sets and saves it in the /data directory.

utilities.py contains functions for loading and saving the training data and models..

rnn.py contains the rnn code.

rnn_sherlock_lm.py is the main script. It essentially acts as a wrapper to the underlying functions. It will create a RNN and attempt to load a data set from the /data directory.


# Functions in rnn_sherlock_lm.py

train(num_examples) Trains a model using Stochastic Gradient Descent for the number of specified training examples. Use train(len(x_train)) to train on the entire dataset.

generate_sentence() Generates a random sentence using the current model.

calc_loss(number_trainin_examples=1000) Calculates the current loss/error of the model for the specified number of training examples.

test_predictions(sentence_index=0) Tests forward propagation and predictions for one sentence at the specified index.

def test_sgd(sentence_index=0) Tests Stochastic Gradient Descent for one sentence at the specified index.

save_model() Saves a model file to model_path.

load_model() Loads a model file from model_path.