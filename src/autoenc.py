#!/usr/bin/env python2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import Adadelta
from sklearn.model_selection import train_test_split

from data import postprocess, trainSet, preprocess, cached_sequence_data, import_sequence, import_matrix


def batch_gen(x_train, batch_size):
    total_samples = x_train.shape[0]
    n_batches = total_samples / batch_size
    i = 0
    while True:
        if i > n_batches: i = 0

        x_in = x_train[i * batch_size:min((i + 1) * batch_size, total_samples), :].toarray()
        x_test = x_train[i * batch_size + 1:min((i + 1) * batch_size + 1, total_samples), :].toarray()

        diff = len(x_in) - len(x_test)
        if diff: x_test = np.vstack((x_test, x_test[-diff:]))

        yield (x_in, x_test)
        i += 1


def one_hot_encode(item, rating, input_dim, do_preprocess=False):
    v = np.zeros((input_dim,), dtype=np.float32)
    v[item] = rating
    if do_preprocess:
        preprocess_ = np.vectorize(preprocess)
        v = preprocess_(v)
    return v


def sequence_encode(seq, input_dim):
    v = np.zeros(input_dim, dtype=np.float32)
    for (it, r) in seq:
        v[it] = r
    return v


def masked_mse(do_preprocess=False):
    def loss(y_true, y_pred):
        zero = 0 if not do_preprocess else preprocess(0)
        m = (y_pred != zero)
        masked_val = K.sum(K.square(y_true[:, m] - y_pred[:, m]))  # / sum(m)
        return masked_val

    return loss


class KerasBaseline(object):
    # Visualize with tensorboard --logdir=/tmp/autoencoder
    """
    Model {
      [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> output]
      (1): nnsparse.SparseLinearBatch(X -> 700)
      (2): nn.Tanh
      (3): nn.Linear(700 -> 500)
      (4): nn.Tanh
      (5): nn.Linear(500 -> 700)
      (6): nn.Tanh
      (7): nn.Linear(700 -> X)
      (8): nn.Tanh
    }

    """

    def __init__(self, input_dim, encoding_dim, weight_reg=0.01):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim

        # input placeholder
        input_d = Input(shape=(input_dim,))
        # "encoded" is the encoded representation of the input
        encoded = Dense(encoding_dim * 2, activation='tanh', W_regularizer=l2(0.025))(input_d)
        encoded = Dropout(0.2)(encoded)
        encoded = Dense(encoding_dim, activation='tanh', W_regularizer=l2(0.2))(encoded)

        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(encoding_dim * 2, activation='tanh', W_regularizer=l2(0.025))(encoded)
        decoded = Dropout(0.2)(decoded)
        decoded = Dense(input_dim, activation='tanh', W_regularizer=l2(0.025))(decoded)

        # this model maps an input to its reconstruction
        self.model = Model(input=input_d, output=decoded)
        print(self.model.summary())
        self.model.compile(optimizer='adadelta',
                           loss='mean_squared_error')
        # loss=masked_mse(True))

    def fit_gen(self, x_train, epochs, n_samples_epoch, batch_size):
        self.model.fit_generator(generator=batch_gen(x_train, batch_size),
                                 nb_epoch=epochs, samples_per_epoch=n_samples_epoch,
                                 callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

    def predict_generator(self, X, batch_size):
        # encode and decode some ratings
        # note that we take them from the *test* set
        batch_ratings = self.model.predict_generator(batch_gen(X, batch_size), 10 * batch_size)
        return batch_ratings

    def score_seq(self, validation_set, do_postprocess=False):
        # Test with and without history sequence
        errors = np.empty(len(validation_set), dtype=np.float32)
        for i, (user, item, rating) in enumerate(validation_set.values):
            # Encode the item we wish to predict
            inp = one_hot_encode(item, preprocess(0), self.input_dim)
            # Reshape it to Keras input format
            inp = inp.reshape((1, self.input_dim))
            # Reconstruct the one-hot encoded vector
            reconstruction = self.model.predict(inp, 1)
            # Evaluate the difference in post-processed
            prediction = reconstruction[0, item]
            if do_postprocess: prediction = postprocess(prediction)
            errors[i] = prediction - rating
            if i < 10:
                print("Rating={} Reconstruction={} -> Error: {} ".format(rating, prediction, errors[i]))

        # Return the rmse
        return np.sqrt(np.mean(np.power(errors, 2)))


if __name__ == '__main__':
    # x_train, val_seq = cached_sequence_data()

    # Load vectors
    x_train, x_test = import_matrix(pr_valid=0.05, do_preprocess=False, return_sparse=False)

    # mask = x_train != preprocess(0)
    # Unbias item vectors by removing the mean rating
    # x_train -= x_train.mean(axis=1)[:, np.newaxis]
    # x_test -= x_test.mean(axis=1)[:, np.newaxis]
    total_samples, input_dim = x_train.shape

    encoding_dim = 350
    epochs = 2
    big_batch_size = total_samples
    batch_size = 50

    autoenc = KerasBaseline(input_dim, encoding_dim)
    autoenc.model.fit(x_train, x_train,
                      batch_size=batch_size,
                      shuffle=True,
                      # sample_weight=mask,
                      # validation_split=0.05,
                      validation_data=(x_test, x_test),
                      # callbacks=[TensorBoard(log_dir='/tmp/autoencoder')],
                      nb_epoch=epochs)

    _, val_seq = cached_sequence_data()
    val_set = val_seq[:1000]
    # autoenc.fit_gen(big_batch_size)

    score = autoenc.score_seq(val_set)
    print("Final validation score on {} held out samples is {}".format(len(val_set), score))
