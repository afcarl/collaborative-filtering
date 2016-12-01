import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from VariationalAutoencoder import VariationalAutoencoder
from src.data import cached_sequence_data


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, data.shape[0] - batch_size)
    return data[start_index:(start_index + batch_size)]


sequences = cached_sequence_data()
X_train, X_test = train_test_split(sequences, test_size=0.1)

n_samples, n_inp = X_train.shape
training_epochs = 2
batch_size = 50
display_step = 1

autoencoder = VariationalAutoencoder(n_input = n_inp,
                                     n_hidden = 700,
                                     optimizer = tf.train.AdamOptimizer(learning_rate = 0.001))

for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)

        # Fit training using batch data
        cost = autoencoder.partial_fit(batch_xs.toarray())
        # Compute average loss
        avg_cost += cost / n_samples * batch_size

    # Display logs per epoch step
    if epoch % display_step == 0:
        print "Epoch:", '%04d' % (epoch + 1), \
            "cost=", "{:.9f}".format(avg_cost)

print "Total cost: " + str(autoencoder.calc_total_cost(X_test.toarray()))
