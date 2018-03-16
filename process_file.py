import itertools
import nltk
from keras.preprocessing.sequence import pad_sequences
from utilities import *

# File paths
file_path = "resources\Short Stories.txt"
data_path = "./data/sherlock-training-data.pkl"

# Sentence tokens
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

# Number of words to hold in vocabulary
vocabulary_size = 8000

print("Processing file: ", file_path)
file = open(file_path).read()

# Split into sentences
sentences = nltk.sent_tokenize(file)

# Remove extra whitespace
sentences = [' '.join(line.split()).strip() for line in sentences]

# Append SENTENCE_START and SENTENCE_END tokens
sentences = ["%s %s %s" % (sentence_start_token, sentence, sentence_end_token) for sentence in sentences]
print("Parsed %d sentences." % (len(sentences)))

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]

# Count the word frequencies
word_frequency = nltk.FreqDist(itertools.chain(*tokenized_sentences))

# Get the most common words and build index to word and word to index vectors
vocabulary = word_frequency.most_common(vocabulary_size - 2)
print("Found %d unique word tokens." % len(word_frequency.items()))

# Generate word to index and index to words (Add the word not the frequency from our vocabulary data)
index_to_word = [x[0] for x in vocabulary]
index_to_word.append(unknown_token)
index_to_word.insert(0, "")

# Dictionary of {word : index} pairs
word_to_index = dict([(word, i) for i, word in enumerate(index_to_word)])

# Replace all words not in our vocabulary with the unknown token
for i, sentence in enumerate(tokenized_sentences):
    for j, word in enumerate(sentence):
        if word not in word_to_index:
            tokenized_sentences[i][j] = unknown_token
print("Using vocabulary size %d." % vocabulary_size)
print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (
    vocabulary[-1][0], vocabulary[-1][1]))
print("Example sentence after Pre-processing: '%s'" % tokenized_sentences[0])

# Example and labeled training sets
x_train = []
y_train = []
# Create the training data
for sentence in tokenized_sentences:
    x = []
    y = []
    # All but the SENTENCE_END token
    for word in sentence[: -1]:
        x.append(word_to_index[word])
    # All but the SENTENCE_START token
    for word in sentence[1:]:
        y.append(word_to_index[word])

    x_train.append(x)
    y_train.append(y)

# For Keras LSTM must pad the sequences to same length and return a numpy array
x_train_pad = pad_sequences(x_train, maxlen=None, padding='post', value=0)
y_train_pad = pad_sequences(y_train, maxlen=None, padding='post', value=0)

num_sentences = x_train_pad.shape[0]
print("Number of Sentences: ", num_sentences)
max_input_len = x_train_pad.shape[1]
print("Max Sentence length: ", max_input_len)

# Save data to file
data = dict(
    x_train=x_train,
    y_train=y_train,
    x_train_pad=x_train_pad,
    y_train_pad=y_train_pad,
    word_to_index=word_to_index,
    index_to_word=index_to_word,
    vocabulary=vocabulary,
    num_sentences=num_sentences,
    max_input_len=max_input_len)

print("Saving training data")
try:
    save_training_data(data_path, data)
except FileNotFoundError as err:
    print("Error saving data " + str(err))

# Write sentences to file
# with open("resources\Short Stories (Sentences).txt", 'w') as file:
#     for line in sentences:
#         file.write(line + "\n")

