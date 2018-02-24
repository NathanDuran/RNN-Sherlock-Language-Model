import itertools
import nltk


class ProcessFile:

    unknown_token = "UNKNOWN_TOKEN"
    sentence_start_token = "SENTENCE_START"
    sentence_end_token = "SENTENCE_END"

    # Raw sentences
    sentences = []
    # Sentences tokenized into words
    tokenized_sentences = []
    # Frequency of each word in training set
    word_frequency = []
    # Total number of words in training set
    vocabulary = []
    # # Word indexes
    # index_to_word = []
    # word_to_index = ()
    # # Example and labeled training sets
    # x_train = []
    # y_train = []

    def __init__(self, vocabulary_size, file_path):

        self.vocabulary_size = vocabulary_size
        self.file = open(file_path).read()
        # Word indexes
        self.index_to_word = []
        self.word_to_index = ()
        # Example and labeled training sets
        self.x_train = []
        self.y_train = []
    # Tokenize the text into sentences
    # Removes extra white space characters
    # Appends start and end tokens
    def tokenize_to_sentences(self):

        # Split into sentences
        self.sentences = nltk.sent_tokenize(self.file)

        # Remove extra whitespace
        self.sentences = [' '.join(line.split()).strip() for line in self.sentences]

        # Append SENTENCE_START and SENTENCE_END tokens
        self.sentences = ["%s %s %s" % (self.sentence_start_token, sentence, self.sentence_end_token) for sentence in self.sentences]
        return self.sentences

    # Tokenize the sentences into words
    # Count word frequencies
    # Build vocabulary
    def tokenize_to_words(self):

        self.tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in self.sentences]

        # Count the word frequencies
        self.word_frequency = nltk.FreqDist(itertools.chain(*self.tokenized_sentences))

        # Get the most common words and build index to word and word to index vectors
        self.vocabulary = self.word_frequency.most_common(self.vocabulary_size - 1)
        return [self.tokenized_sentences, self.word_frequency, self.vocabulary]

    # Generate word to index and index to words
    # Replaces least common words with unkown_token
    def generate_word_indexes(self):

        # Add the word not the frequency from our vocabulary
        self.index_to_word = [x[0] for x in self.vocabulary]
        self.index_to_word.append(self.unknown_token)

        # Dictionary of {word : index} pairs
        self.word_to_index = dict([(word, i) for i, word in enumerate(self.index_to_word)])

        # Replace all words not in our vocabulary with the unknown token
        for i, sentence in enumerate(self.tokenized_sentences):
            for j, word in enumerate(sentence):
                if word not in self.word_to_index:
                    self.tokenized_sentences[i][j] = self.unknown_token

    # Create the training data
    def create_training_data(self):

        for sentence in self.tokenized_sentences:
            x = []
            y = []
            # All but the SENTENCE_END token
            for word in sentence[: -1]:
                x.append(self.word_to_index[word])
            # # All but the SENTENCE_START token
            for word in sentence[1:]:
                y.append(self.word_to_index[word])

            self.x_train.append(x)
            self.y_train.append(y)
