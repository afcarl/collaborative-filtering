#!/usr/bin/env python2
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import TensorBoard

import matplotlib.pyplot as plt

# this is the size of our encoded representations
from src.data import import_train, import_sequence, postprocess

x_train, x_test = import_sequence(1e5) #(do_preprocess=True)
x_train_recon = np.vstack((x_train[1:], x_train[-1]))
x_test_recon = np.vstack((x_test[1:], x_test[-1]))

input_dim = x_train.shape[1]
encoding_dim = 512  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
epochs = 1
batch_size = 256

# this is our input placeholder
input_d = Input(shape=(input_dim,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='tanh')(input_d)
#encoded = Dense(encoding_dim/2, activation='relu')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(input_dim, activation='tanh')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input=input_d, output=decoded)

# this model maps an input to its encoded representation
encoder = Model(input=input_d, output=encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

autoencoder.fit(x_train, x_train_recon,
                nb_epoch=epochs,
                batch_size=batch_size,
                shuffle=False,
                validation_data=(x_test, x_test_recon),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
# Visualize with tensorboard --logdir=/tmp/autoencoder
# encode and decode some digits
# note that we take them from the *test* set
encoded_ratings = encoder.predict(x_test)
decoded_ratings = decoder.predict(encoded_ratings)

plt.hist(list(map(postprocess, decoded_ratings)))
'''
for k in range(10):
    i, j = np.nonzero(x_test[k])
    for r in range(len(i)):
        print(x_test[i[r], j[r]], decoded_ratings[i[r], j[r]])
'''