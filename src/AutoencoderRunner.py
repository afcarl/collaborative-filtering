import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from VariationalAutoencoder import VariationalAutoencoder
from data import cached_sequence_data, import_matrix, import_tensor, one_hot_encode, get_random_block_from_data
import sklearn.preprocessing as prep

from DenoisingAutoencoder import MaskingNoiseAutoencoder


def min_max_scale(X_train, X_test):
    preprocessor = prep.MaxAbsScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler(with_mean=False).fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test


def main(_):
    sequences = cached_sequence_data(do_preprocess=True)
    X_train, X_test = train_test_split(sequences, test_size=0.1)
    # X_train, X_test = import_matrix(do_preprocess=True)
    # X_train, X_test = min_max_scale(X_train, X_test)
    # X_train, X_test = standard_scale(X_train, X_test)
    with tf.Session() as sess:
        # tf.logging.set_verbosity(tf.logging.INFO)
        # summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, sess.graph)

        n_samples, n_visible = X_train.shape #tf.shape(X_train).eval()
        training_epochs = 20
        batch_size = 50
        display_step = 1

        # autoencoder = VariationalAutoencoder(n_input=n_visible,
        #                                      n_hidden=700,
        #                                      optimizer=tf.train.AdamOptimizer(learning_rate=0.001))

        autoencoder = MaskingNoiseAutoencoder(sess, n_input=n_visible,
                                              n_hidden=700,
                                              transfer_function=tf.tanh,
                                              dropout_probability=0.2,
                                              optimizer=tf.train.AdagradOptimizer(learning_rate=0.01))

        total_batch = int(n_samples / batch_size)
        # X_split = tf.sparse_split(0, total_batch, X_train)
        for epoch in range(training_epochs):
            avg_cost = 0.
            # Loop over all batches
            for i in range(total_batch):
                # Load batch
                batch_xs = get_random_block_from_data(X_train, batch_size).toarray()
                # Unbias the data
                batch_xs -= batch_xs.mean(axis=1)[:, np.newaxis].astype(np.float32)
                # Fit training using batch data
                cost = autoencoder.partial_fit(batch_xs)
                # Compute average loss
                avg_cost += cost / n_samples * batch_size

            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        # Testing (densify the matrix by batch)
        avg_cost = 0.
        for i in range(total_batch):
            batch_xs = X_test[i * batch_size:min((i + 1) * batch_size, n_samples), :].toarray()
            batch_xs -= batch_xs.mean(axis=1)[:, np.newaxis].astype(np.float32)
            # Fit training using batch data
            cost = autoencoder.calc_total_cost(batch_xs)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        # print("Total cost: " + str(autoencoder.calc_total_cost(X_test.toarray())))
        print("Total cost: " + str(np.sqrt(avg_cost)))

        # Sampling some test values
        t_rows, t_cols = X_test.nonzero()
        rating = X_test[t_rows[0], t_cols[0]]
        test = X_test[t_rows[0]].toarray()
        t = autoencoder.transform(test)
        try:
            output = autoencoder.generate(t)
            print(output[0, t_cols[0]], rating)
        except Exception as exc:
            print(exc)


if __name__ == '__main__':
    tf.app.run()
