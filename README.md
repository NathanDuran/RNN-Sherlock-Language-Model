# RNN-Sherlock-Language-Model

A Recurrent Neural Network Language Model trained on ['The Stories of Sherlock Holmes'](https://sherlock-holm.es/ascii/).

Contains a traditional RNN and an LSTM. The RNN is an adapted version of the one outlined in [this tutorial](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/) by Denny Britz.

The data directory contains a pre-processed data set (sherlock-training-data.pkl) and pre-trained models.
The included models have...

To generate a sentence with the included pre-trained model simply set the desired model_type (rnn or lstm) and run sherlock_lm.py and call generate_sentence().


# Example of Generated Sentences

## Vanilla RNN:

10 Epoch - said of course, leaning neither, dear employer, Miss side was just to you; but returning groping, but he makes her that round she were an deduction before I could pray of turning I never the road of Oxfordshire of the extreme.

20 Epoch - said of Holmes, Mrs. Holmes, though a moment were repaid in different, certainly from your instant of our word and, and to-day, then, you suppose, Alice to do for where?

30 Epoch - said of a very engineer lens," fascinating, the dear light, much, now, who was let her that we were observed that it was I. from his strength, with a rough during in a nice of good-fortune, together at your with your good villain," said she gravely, on here and no real that it is a hundred feeling, but he answered you look, and was not deserted house your dear Rucastle, Miss Toller?"

40 Epoch - said of Holmes, protruded Toller to introduce any three together.

50 Epoch - said of a serious, for success --" said Holmes," of no means recollect for weeks fact that, and when Holmes suddenly all my trivial, then, you shall tell it out and found.

## Keras LSTM:

10 Epoch - the old man, and the name of the be in the house.

# Included Python Files

process_file.py turns the raw text into training data sets and saves it in the /data directory.

utilities.py contains functions for loading and saving the training data and models.

rnn.py contains the rnn code.

lstm.py contains the lstm code.

sherlock_lm.py is the main script. It essentially acts as a wrapper to the underlying functions. It will create a RNN or LSTM (based on the model_type variable) and attempt to load a data set from the /data directory.

sherlock-training-data.pkl contains 45464 prepared sentences from all of the stories of Sherlock Holmes.

keras_lstm.py Keras implementation of the language model.

# Functions in sherlock_lm.py

train(num_examples) - Trains a model using Stochastic Gradient Descent for the number of specified training examples. Use train(len(x_train)) to train on the entire dataset.

calc_loss(number_trainin_examples=1000) - Calculates the current loss/error of the model for the specified number of training examples.

test_predictions(sentence_index=0) - Tests forward propagation and predictions for one sentence at the specified index.

test_sgd(sentence_index=0) - Tests Stochastic Gradient Descent for one sentence at the specified index.

build_model(type) - creates an empty RNN or LSTM model based on the specified type string ("rnn" or "lstm").


# TODO
Sort generate sentence function

Documentation