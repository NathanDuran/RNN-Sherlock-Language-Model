
import numpy as np
from keras.utils import to_categorical


class KerasBatchGenerator(object):

    def __init__(self, x_train, y_train, max_input_len, num_sentences, batch_size, vocabulary_size, skip_step=5):
        self.x_train = x_train
        self.y_train = y_train
        self.max_input_len = max_input_len
        self.num_sentences = num_sentences
        self.batch_size = batch_size
        self.vocabulary_size = vocabulary_size
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.index = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, self.max_input_len))
        y = np.zeros((self.batch_size, self.max_input_len, self.vocabulary_size))

        while True:
            for i in range(self.batch_size):
                if self.index + self.batch_size >= self.num_sentences:
                    # reset the index back to the start of the data set
                    self.index = 0
                x[i, :] = self.x_train[self.index]
                temp_y = self.y_train[self.index]
                # convert all of temp_y into a one hot representation
                y[i, :, :] = to_categorical(temp_y, num_classes=self.vocabulary_size)

                self.index += 1
            yield x, y
