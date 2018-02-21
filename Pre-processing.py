import itertools
import nltk

vocabularySize = 8000
unknownToken = "UNKNOWN_TOKEN"
sentenceStartToken = "SENTENCE_START"
sentenceEndToken = "SENTENCE_END"

# Split into sentences
file_content = open("resources\The Adventures of Sherlock Holmes (No Titles).txt").read()

# Tokenise into sentences
sentences = nltk.sent_tokenize(file_content)

# Remove extra whitespace
sentences = [' '.join(line.split()).strip() for line in sentences]

# Write file with sentences
# with open("resources\The Adventures of Sherlock Holmes (Sentences).txt", 'w') as file:
#
#     for line in sentences:
#         file.write(line + '\n')

# Append SENTENCE_START and SENTENCE_END
sentences = ["%s %s %s" % (sentenceStartToken, s, sentenceEndToken) for s in sentences]
print("Parsed %d sentences." % (len(sentences)))

# Tokenize the sentences into words
tokenizedSentences = [nltk.word_tokenize(s) for s in sentences]

# Count the word frequencies
wordFrequency = nltk.FreqDist(itertools.chain(*tokenizedSentences))
print("Found %d unique word tokens." % len(wordFrequency.items()))

# Get the most common words and build index to word and word to index vectors
vocabulary = wordFrequency.most_common(vocabularySize - 1)

indexToWord = [x[0] for x in vocabulary] # Add the word not the frequency from our vocabulary
indexToWord.append(unknownToken)

wordToIndex = dict([(w, i) for i, w in enumerate(indexToWord)]) # Dictionary of {word : index} pairs

print("Using vocabulary size %d." % vocabularySize)
print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocabulary[-1][0], vocabulary[-1][1]))

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenizedSentences):
    tokenizedSentences[i] = [w if w in wordToIndex else unknownToken for w in sent]

print("Example sentence:\n '%s'" % sentences[0])
print("Example sentence after Pre-processing:\n '%s'" % tokenizedSentences[0])

print("Example sentence:\n '%s'" % sentences[5131])
print("Example sentence after Pre-processing:\n '%s'" % tokenizedSentences[5131])