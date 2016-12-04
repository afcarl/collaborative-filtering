import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import argparse

from VariationalAutoencoder import VariationalAutoencoder
from data import cached_sequence_data, import_matrix, import_tensor, one_hot_encode, get_random_block_from_data, postprocess, \
    sequence_encode
import sklearn.preprocessing as prep

from DenoisingAutoencoder import DenoisingAutoencoder


flags = tf.app.flags
flags.DEFINE_string("save_path", ".", "Directory to write the model.")
flags.DEFINE_integer("hidden", 700, "Latent dimension size.")
flags.DEFINE_float("learning_rate", 0.01, "Initial learning rate.")
flags.DEFINE_integer("batch_size", 50, "Numbers of training examples per mini batch.")
flags.DEFINE_integer("epochs", 1, "Number of training epochs.")
flags.DEFINE_integer("display_step", 1, "Period to display loss.")
flags.DEFINE_integer("n_train", int(1e5), "Number of sequences to train on.")
FLAGS = flags.FLAGS

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
    training_epochs = FLAGS.epochs
    batch_size = FLAGS.batch_size
    display_step = FLAGS.display_step
    hidden_units = FLAGS.hidden
    lr = FLAGS.learning_rate
    n_train = FLAGS.n_train

    X_train, X_test = cached_sequence_data(max_items=n_train, do_preprocess=True)

    # The model uses 32-bit precision floats
    if X_train.dtype == np.float64: X_train = X_train.astype(np.float32)
    # if X_test.dtype == np.float64: X_train = X_test.astype(np.float32)

    with tf.Session() as sess:
        tf.logging.set_verbosity(tf.logging.INFO)
        summary_writer = tf.train.SummaryWriter(FLAGS.save_path, sess.graph)

        n_samples, input_dim = X_train.shape #tf.shape(X_train).eval()
        print('[Session] Training on {} samples'.format(n_samples))

        autoencoder = DenoisingAutoencoder(sess, n_input=input_dim,
                                           n_hidden=hidden_units,
                                           transfer_function=tf.tanh,
                                           dropout_probability=0.2,
                                           optimizer=tf.train.AdagradOptimizer(learning_rate=lr))

        total_batch = int(n_samples / batch_size)
        # X_split = tf.sparse_split(0, total_batch, X_train)
        for epoch in range(training_epochs):
            avg_cost = 0.
            # Loop over all batches
            for i in range(total_batch):
                # Load batch
                batch_xs = get_random_block_from_data(X_train, batch_size).toarray()

                # Get missing values indices (will not participate in activations or backprop)
                missing = batch_xs == 0

                # Fit using batch data
                cost = autoencoder.partial_fit(batch_xs, missing)
                # Compute average loss
                avg_cost += cost / n_samples * batch_size

            # Display loss evolution
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        # Computing the loss on the test set
        avg_cost = 0.
        X_ohe_test = sequence_encode(X_test, input_dim)
        for i in range(total_batch):
            batch_xs = X_ohe_test[i * batch_size:min((i + 1) * batch_size, n_samples), :]
            missing = batch_xs == 0
            # Fit training using batch data
            cost = autoencoder.calc_total_cost(batch_xs, missing)
            # Compute average loss
            avg_cost += cost / n_samples * batch_size

        print("Total cost: " + str(np.sqrt(avg_cost)))

        # Sampling some test values
        print("User, item, rating, reconstr, real rating")
        for i in range(10):
            user, item, rating = X_test.iloc[i]
            one_hot = one_hot_encode(item, rating, input_dim)
            latent_repr = autoencoder.transform(one_hot)
            try:
                output = autoencoder.generate(latent_repr)
                print(user, item, rating, postprocess(output[0, item]), postprocess(rating))
            except Exception as exc:
                print(exc)


if __name__ == '__main__':
    tf.app.run()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
          '--learning_rate',
          type=float,
          default=0.01,
          help='Initial learning rate.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    """
