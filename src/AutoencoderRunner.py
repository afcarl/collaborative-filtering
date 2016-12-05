from time import time
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import argparse

from VariationalAutoencoder import VariationalAutoencoder
from data import cached_sequence_data, import_matrix, import_tensor, one_hot_encode, get_random_block_from_data, postprocess, \
    sequence_encode, import_validation_sparse
import sklearn.preprocessing as prep

from DenoisingAutoencoder import DenoisingAutoencoder


flags = tf.app.flags
flags.DEFINE_string("save_path", os.path.join('..', 'output'), "Directory to write the model.")
flags.DEFINE_integer("hidden", 356, "Latent dimension size.")
flags.DEFINE_float("learning_rate", 0.01, "Initial learning rate.")
flags.DEFINE_float("test_ratio", 0.05, "Ratio of the training set to test on.")
flags.DEFINE_integer("batch_size", 32, "Numbers of training examples per mini batch.")
flags.DEFINE_integer("epochs", 20, "Number of training epochs.")
flags.DEFINE_integer("display_step", 1, "Period to display loss.")
flags.DEFINE_integer("n_train", None, "Number of sequences to train on.")
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
    test_ratio = FLAGS.test_ratio

    X_train, X_test = cached_sequence_data(max_items=n_train, test_ratio=test_ratio, do_preprocess=True)

    # The model uses 32-bit precision floats
    if X_train.dtype == np.float64: X_train = X_train.astype(np.float32)
    # if X_test.dtype == np.float64: X_train = X_test.astype(np.float32)

    print('[Test ] Encoding in sequences...')
    t0 = time()
    X_ohe_test = sequence_encode(X_test)
    print('[Test ] Encoding took {}'.format(time() - t0))

    with tf.Session() as sess:
        # Create a saver for writing training checkpoints.
        # saver = tf.train.Saver()

        tf.logging.set_verbosity(tf.logging.INFO)
        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.train.SummaryWriter(FLAGS.save_path, sess.graph)

        n_samples, input_dim = X_train.shape #tf.shape(X_train).eval()
        print('[Train] Running on {} samples'.format(n_samples))

        autoencoder = DenoisingAutoencoder(sess, n_input=input_dim,
                                           n_hidden=hidden_units,
                                           transfer_function=tf.tanh,
                                           dropout_probability=0.2,
                                           optimizer=tf.train.AdagradOptimizer(learning_rate=lr))

        total_batch = int(n_samples / batch_size)
        # X_split = tf.sparse_split(0, total_batch, X_train)
        for epoch in range(training_epochs):
            t0 = time()
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
                print("[Train] Epoch: {:04d}, cost= {:.9f} ({:.3f}s)".format(epoch + 1,avg_cost, time()-t0))

            # Save a checkpoint and evaluate the model periodically.
            if (epoch + 1) % 10 == 0 or (epoch + 1) == training_epochs:
                # checkpoint_file = os.path.join(FLAGS.save_path, 'model.ckpt')
                # saver.save(sess, checkpoint_file, global_step=epoch)

                # Evaluation on the test set
                avg_cost = 0.
                n_test_samples = X_ohe_test.shape[0]
                test_batches = int(n_test_samples / batch_size)
                print('[Test ] Evaluating on {} samples ({} batches)'.format(n_test_samples, test_batches))
                t0 = time()
                for i in range(test_batches):
                    batch_xs = X_ohe_test[i * batch_size:min((i + 1) * batch_size, n_test_samples), :].toarray()
                    missing = batch_xs == 0
                    # Fit training using batch data
                    cost = autoencoder.calc_total_cost(batch_xs, missing)
                    # Compute average loss
                    avg_cost += cost / n_test_samples * batch_size

                print("[Test ] Total cost: {} ({:.3f}s)".format(np.sqrt(avg_cost), time()-t0))

        # Sampling some test values
        print("User\t Item\t Rating \t Reconstruction")
        for i in range(50):
            user, item, rating = X_test.iloc[i]
            one_hot = one_hot_encode(item, rating, input_dim).reshape(1, -1)
            latent_repr = autoencoder.transform(one_hot)
            output = autoencoder.generate(latent_repr)
            print('{}\t {}\t {}\t {}'.format(user, item, rating, postprocess(output[0, item])))

        tbl, score_sp_sequences = import_validation_sparse()
        user_grp = tbl.groupby('new_user', sort=False)
        tbl['new_rating'] = 0.0
        for i, (user, item_lst) in enumerate(user_grp):
            input_vect = score_sp_sequences[i,:].toarray().reshape(1, -1)
            latent_repr = autoencoder.transform(input_vect)
            output = autoencoder.generate(latent_repr)
            tbl.ix[item_lst.index, 'new_rating'] = postprocess(output[item_lst.new_item.values])
        tbl.to_csv(os.path.join(FLAGS.save_path, 'customeraffinity.predictions'), index=False,
                   columns=['', 'new_user', 'new_item','new_rating'])

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
