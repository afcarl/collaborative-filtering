from time import time
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import argparse

from VariationalAutoencoder import VariationalAutoencoder
from data import cached_sequence_data, import_matrix, import_tensor, one_hot_encode, get_random_block_from_data, \
    sequence_encode, validation_sparse_matrix, R_SHAPE
from Utils import postprocess
import sklearn.preprocessing as prep

from DenoisingAutoencoder import DenoisingAutoencoder

tf.logging.set_verbosity(tf.logging.DEBUG)

flags = tf.app.flags
flags.DEFINE_string("save_path", os.path.join('..', 'logs'), "Directory to write the model.")
flags.DEFINE_integer("hidden", 712, "Latent dimension size.")
flags.DEFINE_float("learning_rate", 0.01, "Initial learning rate.")
flags.DEFINE_float("test_ratio", 0.1, "Ratio of the training set to test on.")
flags.DEFINE_float("hide_ratio", 0.25, "Ratio of the corrupted entries.")
flags.DEFINE_integer("batch_size", 32, "Numbers of training examples per mini batch.")
flags.DEFINE_integer("epochs", 25, "Number of training epochs.")
flags.DEFINE_integer("display_step", 1, "Period to display loss.")
flags.DEFINE_integer("test_step", 5, "Period to test model.")
flags.DEFINE_integer("n_train", None, "Number of sequences to train on.")
flags.DEFINE_integer("prediction_precision", 3, "Digits to keep when doing predictions.")

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
    test_step = FLAGS.test_step
    hidden_units = FLAGS.hidden
    lr = FLAGS.learning_rate
    n_train = FLAGS.n_train
    test_ratio = FLAGS.test_ratio
    hide_ratio = FLAGS.hide_ratio
    digits = FLAGS.prediction_precision
    save_path = FLAGS.save_path

    if tf.gfile.Exists(save_path):
        tf.gfile.DeleteRecursively(save_path)
    tf.gfile.MakeDirs(save_path)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    t0 = time()
    X_train, X_test = cached_sequence_data(max_items=n_train, test_ratio=test_ratio, do_preprocess=True)
    print("[Loaded] {} sequences - took {} s".format(X_train.shape, time() - t0))

    # The model uses 32-bit precision floats
    if X_train.dtype == np.float64: X_train = X_train.astype(np.float32)
    # if X_test.dtype == np.float64: X_train = X_test.astype(np.float32)
    assert X_test.shape[1] == 3561, ValueError('Test set has fewer dimensions than needed.')
    if not os.path.exists(save_path): os.makedirs(save_path)

    with tf.Session() as sess:

        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer = tf.train.SummaryWriter(save_path, sess.graph)

        n_samples, input_dim = X_train.shape  # tf.shape(X_train).eval()
        print('[Train] Running on {} samples'.format(n_samples))

        autoencoder = DenoisingAutoencoder(sess, n_input=input_dim,
                                           n_hidden=hidden_units,
                                           transfer_function=tf.tanh,
                                           dropout_probability=hide_ratio,
                                           optimizer=tf.train.AdagradOptimizer(learning_rate=lr))

        # Create a saver for writing training checkpoints.
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)

        n_train_batches = int(n_samples / batch_size)
        # X_split = tf.sparse_split(0, total_batch, X_train)

        n_test_samples = X_test.shape[0]
        test_batches = int(n_test_samples / batch_size)
        print('[Test ] Test set size is {} samples ({} batches)'.format(n_test_samples, test_batches))

        for epoch in range(training_epochs):
            t0 = time()
            avg_cost = 0.
            # Loop over all batches
            for i in range(n_train_batches):
                # Load batch
                batch_xs = get_random_block_from_data(X_train, batch_size).toarray()

                # Get missing values indices (will not participate in activations or backprop)
                missing = batch_xs == 0

                feed_dict = {autoencoder.x_orig: batch_xs,
                             autoencoder.keep_prob: autoencoder.dropout_probability,
                             autoencoder.missing_mask: missing}

                # cost = sess.run(autoencoder.cost, feed_dict=feed_dict)
                cost, opt = autoencoder.partial_fit(batch_xs, missing)
                # Fit using batch data
                # grads_and_vars = sess.run(autoencoder.optimizer.compute_gradients, feed_dict={}
                # Ask the optimizer to apply the capped gradients.
                # autoencoder.optimizer.apply_gradients(grads_and_vars)

                # with tf.name_scope('train_summaries'):
                tf.summary.scalar('squared_error', cost)
                    # for grad, var in grads_and_vars:
                    #     tf.summary.histogram(var.name, grad)
                # Compute average loss
                avg_cost += cost / n_samples * batch_size

            # Display loss evolution
            if epoch % display_step == 0:
                print("[Train] Epoch: {:04d}, Cost = {:.9f} ({:.3f}s)".format(epoch + 1, avg_cost, time() - t0))
                # Update the events file.
                summary_writer.add_summary(opt, epoch)
                summary_writer.flush()

            # Save a checkpoint and evaluate the model periodically.
            if (epoch + 1) % test_step == 0 or (epoch + 1) == training_epochs:
                checkpoint_file = os.path.join(save_path, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=epoch)
                # Evaluation on the test set
                avg_cost = 0.
                t0 = time()
                for i in range(test_batches):
                    batch_xs = X_test[i * batch_size:min((i + 1) * batch_size, n_test_samples), :].toarray()
                    missing = batch_xs == 0
                    # Fit training using batch data
                    cost = autoencoder.calc_total_cost(batch_xs, missing)
                    # Compute average loss
                    avg_cost += cost / n_test_samples * batch_size

                print("[Test ] Cost = {:.9f} ({:.3f}s)".format(avg_cost, time() - t0))

        summary_writer.close()
        # Sampling some test values
        # print("User\t Item\t Rating \t Reconstruction")
        # for i in range(50):
        #     user, item, rating = X_test.iloc[i]
        #     one_hot = one_hot_encode(item, rating, input_dim).reshape(1, -1)
        #     latent_repr = autoencoder.transform(one_hot)
        #     output = autoencoder.generate(latent_repr)
        #     prediction = np.round(np.clip(postprocess(output[0, item]), 1.0, 5.0), digits)
        #     print('{}\t {}\t {}\t {}'.format(user, item, rating, prediction))

        print('[Valid] Loading validation dataframe & sequences')
        score_df, score_sp_sequences = validation_sparse_matrix()
        user_grp = score_df.groupby('new_user', sort=False)
        score_df['new_rating'] = 0.0
        print('[Valid] Computing predictions')
        for i, (user, item_lst) in enumerate(user_grp):
            if i > 2: break
            input_vect = score_sp_sequences[i, :].toarray().reshape(1, -1)
            latent_repr = autoencoder.transform(input_vect)
            output = autoencoder.generate(latent_repr)
            predictions = np.clip(np.round(postprocess(output[0, item_lst.new_item.values]), digits), 1.0, 5.0)
            score_df.ix[item_lst.index, 'new_rating'] = predictions
        print('[Valid] Flushing predictions to disk')
        score_df.to_csv(os.path.join(save_path, 'customeraffinity.predictions'), index=False,
                        columns=['', 'new_user', 'new_item', 'new_rating'])


def restore_model():
    with tf.Session() as sess:
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        saver.restore(sess, FLAGS.save_path)


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
