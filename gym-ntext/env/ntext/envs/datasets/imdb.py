
import numpy as np
import logging as log
from keras.datasets import imdb
from keras.preprocessing import sequence
from ntext.envs.datasets.base import NtextDataset


class ImdbDataset(NtextDataset):

    def __init__(self):
        self.maxlen = 200
        self.max_features = 5000
        self.ngrams = 1
        self.skip_top= 100
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def create_ngram_set(self, input_list, ngram_value=2):
        """
        Extract a set of n-grams from a list of integers.

        >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
        {(4, 9), (4, 1), (1, 4), (9, 4)}

        >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
        [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
        """
        return set(zip(*[input_list[i:] for i in range(ngram_value)]))


    def add_ngram(self, sequences, token_indice, ngram_range=2):
        """
        Augment the input list of list (sequences) by appending n-grams values.

        Example: adding bi-gram
        >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
        >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
        >>> add_ngram(sequences, token_indice, ngram_range=2)
        [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

        Example: adding tri-gram
        >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
        >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
        >>> add_ngram(sequences, token_indice, ngram_range=3)
        [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
        """
        new_sequences = []
        for input_list in sequences:
            new_list = input_list[:]
            for ngram_value in range(2, ngram_range + 1):
                for i in range(len(new_list) - ngram_value + 1):
                    ngram = tuple(new_list[i:i + ngram_value])
                    if ngram in token_indice:
                        new_list.append(token_indice[ngram])
            new_sequences.append(new_list)

        return new_sequences

    def load(self):
        # Set parameters:
        # ngram_range = 2 will add bi-grams features

        log.info('Loading data...')
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=self.max_features, skip_top=self.skip_top)
        log.info('{} train sequences'.format(len(x_train)))
        log.info('{} test sequences'.format(len(x_test)))
        log.info('Average train sequence length: {}'.format(
            np.mean(list(map(len, x_train)), dtype=int)))
        log.info('Average test sequence length: {}'.format(
            np.mean(list(map(len, x_test)), dtype=int)))

        if self.ngrams > 1:
            log.info('Adding {}-gram features'.format(self.ngrams))
            # Create set of unique n-gram from the training set.
            ngram_set = set()
            for input_list in x_train:
                for i in range(2, self.ngrams + 1):
                    set_of_ngram = self.create_ngram_set(input_list, ngram_value=i)
                    ngram_set.update(set_of_ngram)

            # Dictionary mapping n-gram token to a unique integer.
            # Integer values are greater than max_features in order
            # to avoid collision with existing features.
            start_index = self.max_features + 1
            token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
            indice_token = {token_indice[k]: k for k in token_indice}

            # max_features is the highest integer that could be found in the dataset.
            self.max_features = np.max(list(indice_token.keys())) + 1

            # Augmenting x_train and x_test with n-grams features
            x_train = self.add_ngram(x_train, token_indice, self.ngrams)
            x_test = self.add_ngram(x_test, token_indice, self.ngrams)
            log.info('Average train sequence length: {}'.format(
                np.mean(list(map(len, x_train)), dtype=int)))
            log.info('Average test sequence length: {}'.format(
                np.mean(list(map(len, x_test)), dtype=int)))

        log.info('Pad sequences (samples x time)')
        x_train = sequence.pad_sequences(x_train, maxlen=self.maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=self.maxlen)
        log.info('x_train shape: {}'.format(x_train.shape))
        log.info('x_test shape: {}'.format(x_test.shape))

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def get_tfidf(self):
        #todo: this is wrong
        from sklearn.feature_extraction.text import TfidfTransformer
        transformer = TfidfTransformer()
        X = transformer.fit_transform(self.x_train)
        X = X.todense()
        return X