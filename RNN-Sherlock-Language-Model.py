import timeit
import numpy as np
import ProcessFile
import RNN

file_path = "resources\The Adventures of Sherlock Holmes (No Titles).txt"
vocabulary_size = 8000
learning_rate = 0.005
num_epoch = 50

# Process input file and create training sets
process_file = ProcessFile.ProcessFile(vocabulary_size, file_path)
print("Processing file: ", file_path)

# Tokenize the text into sentences
process_file.tokenize_to_sentences()
print("Parsed %d sentences." % (len(process_file.sentences)))

# Tokenize the sentences into words
process_file.tokenize_to_words()
print("Found %d unique word tokens." % len(process_file.word_frequency.items()))

# Generate word to index and index to words
process_file.generate_word_indexes()
print("Using vocabulary size %d." % vocabulary_size)
print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (process_file.vocabulary[-1][0], process_file.vocabulary[-1][1]))
print("Example sentence after Pre-processing: '%s'" % process_file.tokenized_sentences[0])

# Create the training data
process_file.create_training_data()
x_train = process_file.x_train
y_train = process_file.y_train

# Create RNN
np.random.seed(10)
model = RNN.RNN(vocabulary_size)

# # Test forward propogation and predictions
# output, hidden_state = model.forward_propagation(x_train[10])
# # For the input sentence the model outputs vocabularySize for each word
# print("Output dimensions ", output.shape)
# print(output)
# # The highest probability predictions for each word
# predictions = model.predict(x_train[10])
# print(predictions.shape)
# print(predictions)
#
# # Test loss function (limit to 1000 examples to save time)
# print("Expected Loss for random predictions: %f" % np.log(vocabulary_size))
# print("Actual loss: %f" % model.calculate_loss(x_train[:1000], y_train[:1000]))
#
# # Test Stochastic Gradient Descent for one sentence
# time_taken = timeit.Timer(lambda: model.sgd_step(x_train[10], y_train[10], learning_rate)).timeit(number=1)
# print("Time taken (in seconds) for one instance of training data : ", time_taken)

time_taken = timeit.Timer(lambda: model.train_with_sgd(x_train[:100], y_train[:100], learning_rate, num_epoch)).timeit(number=1)
print("Time taken (in seconds) for one epoch over training data : ", time_taken)
# 100 training exampls for 50 epoch took 719 seconds


