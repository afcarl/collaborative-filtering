# -*- coding: utf-8 -*-
import os
from time import time

import numpy as np
import pandas as pd
import tensorflow as tf
from joblib import dump, load
from scipy.sparse import csc_matrix, csr_matrix, coo_matrix
from sklearn.model_selection import train_test_split

from Utils import preprocess

data_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..", "input")

trainSet = os.path.join(data_dir, "customeraffinity.train")
scoreSet = os.path.join(data_dir, "customeraffinity.score")

R_SHAPE = (93705, 3561)


# ---- Data as matrices and SparseTensors

def import_matrix(pr_valid=0.1, do_preprocess=False, return_sparse=True):
    data_frame = pd.read_table(trainSet, sep=',', header=None)

    if do_preprocess:
        data_frame.iloc[:, -1] = data_frame.iloc[:, -1].apply(preprocess)

    train_items, val_items = train_test_split(data_frame, test_size=pr_valid)

    train_mat = to_matrix(train_items)
    valid_mat = to_matrix(val_items)

    if not return_sparse:
        train_mat = train_mat.toarray()
        valid_mat = valid_mat.toarray()

    return train_mat, valid_mat


def to_matrix(tbl, shape=R_SHAPE):
    user_arr = tbl.iloc[:, 0].values
    item_arr = tbl.iloc[:, 1].values
    data = tbl.iloc[:, -1].values

    matrix = csc_matrix((data, (user_arr, item_arr)), dtype=np.float32)

    return matrix


def import_tensor(pr_valid=0.1, do_preprocess=False):
    tbl = pd.read_table(trainSet, sep=',', header=None)

    if do_preprocess:
        tbl.iloc[:, -1] = tbl.iloc[:, -1].apply(preprocess)

    train_items, val_items = train_test_split(tbl, test_size=pr_valid)

    train_mat = to_sparse_tensor(train_items)
    valid_mat = to_sparse_tensor(val_items)

    return train_mat, valid_mat


def to_sparse_tensor(tbl):
    tensor = tf.SparseTensor(list(zip(tbl.iloc[:, 0].values, tbl.iloc[:, 1].values)),
                             tbl.iloc[:, -1].values,
                             list(R_SHAPE))

    return tensor


# ---- Data as sequences of ratings

def dataframe_split(df, test_ratio):
    length = len(df)
    # Generate indices split
    test_indices = np.random.choice(length, int(test_ratio * length), replace=False)
    train_indices = list(set(range(length)).difference(test_indices))

    test_df = df.iloc[test_indices, :]
    train_df = df.iloc[train_indices, :]

    print("[Split] {} train, {} test ".format(len(train_indices), len(test_indices)))
    return train_df, test_df


def import_sequence(max_items=None, do_preprocess=False, test_ratio=0.1):
    # Read training data
    data_frame = pd.read_table(trainSet, sep=',', header=None, nrows=max_items)
    # Preprocess if needed
    if do_preprocess: data_frame.iloc[:, -1].apply(preprocess)

    # Split the table in train-test
    if test_ratio:
        data_frame, data_frame_bis = dataframe_split(data_frame, test_ratio)
    else:
        data_frame_bis = None

    train_seqs = one_more_encoder(data_frame)
    test_seqs = one_more_encoder(data_frame_bis)

    return train_seqs, test_seqs


def one_more_encoder(data_frame):
    # matrix = lil_matrix((n_train, 3), dtype=np.float32)
    tot_n_elts = len(data_frame)
    curr_client = 0
    user_hist = []
    row_indices = []  # np.empty(sp_ids_len, dtype=np.int32)
    col_indices = []  # np.empty(sp_ids_len, dtype=np.int32)
    data = []  # np.empty(sp_ids_len, dtype=np.int32)
    seq_len = 0
    user_bias = 0.0

    previous_index = 0

    for t in range(len(data_frame)):
        user, item, rating = data_frame.iloc[t]

        if not seq_len or user == curr_client:
            user_hist.append((item, rating))
            seq_len += 1
            user_bias = ((seq_len - 1) * user_bias + rating) / float(seq_len)
        else:  # new client
            # Save data
            items, ratings = zip(*user_hist)
            new_n_ratings = previous_index + seq_len
            if new_n_ratings > tot_n_elts:  # Extend datastruct (2 * current length)
                row_indices = np.hstack((row_indices, np.empty(tot_n_elts, dtype=np.float32)))
                col_indices = np.hstack((row_indices, np.empty(tot_n_elts, dtype=np.float32)))
                data = np.hstack((row_indices, np.empty(tot_n_elts, dtype=np.float32)))
                tot_n_elts *= 2

            row_indices[previous_index:new_n_ratings] = np.full(seq_len, t, dtype=np.float32)
            col_indices[previous_index:new_n_ratings] = np.array(items)
            data[previous_index:new_n_ratings] = np.array(ratings) - user_bias

            previous_index = new_n_ratings
            """
            data.extend(map(lambda r: r - user_bias, ratings))
            col_indices.extend(items)
            row_indices.extend([i] * len(items))
            """
            # Reset counters
            user_hist = []
            curr_client = user
            user_bias = 0
            seq_len = 0

        if t % 100000 == 0: print("Processed {} lines".format(t))

    # Construct and return sparse matrix
    matrix = csr_matrix((data[:previous_index], (row_indices[:previous_index], col_indices[:previous_index])),
                        shape=(len(data_frame), R_SHAPE[1]),
                        dtype=np.float32)

    return matrix


def cached_sequence_data(max_items=None, test_ratio=0.1, do_preprocess=False, filename='seq_data', ext='bz2'):
    fp = '{}_{}.{}'.format(filename, 'pproc' if do_preprocess else 'raw', ext)
    data_file = os.path.join(data_dir, fp)
    if os.path.exists(data_file):
        t0 = time()
        sequences, test_dataframe = load(data_file)
        n_sequences = sequences.shape[0]
        n_test_ratings = test_dataframe.shape[0]
        print("[Loaded] {} sequences from cache - took {}s".format(n_sequences, time() - t0))
        if max_items and n_sequences > max_items:
            train_items = np.random.choice(n_sequences, max_items, replace=False)
            sequences = sequences[train_items]
    else:
        sequences, test_dataframe = import_sequence(max_items, do_preprocess, test_ratio)
        dump((sequences, test_dataframe), data_file)
    return sequences, test_dataframe


def future_generator(X, batch_size, sparse=True, back_to_the_future=False):
    total_samples = X.shape[0]
    n_batches = np.ceil(float(total_samples) / batch_size)
    i = 0
    f = 1 if back_to_the_future else 0
    while True:
        if i > n_batches: i = 0

        x_train = X[i * batch_size:min((i + 1) * batch_size, total_samples), :]
        x_test = X[i * batch_size + f:min((i + 1) * batch_size + f, total_samples), :]

        # Eventually extend the target matrix with the last vector of the original matrix
        diff = x_train.shape[0] - x_test.shape[0]
        if diff: x_test = np.vstack((x_test, x_test[-diff:]))

        if sparse:
            yield x_train, x_test
        else:
            yield x_train.toarray(), x_test.toarray()

        i += 1


def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, data.shape[0] - batch_size)
    return data[start_index:(start_index + batch_size), :]


def one_hot_encode(item, rating, input_dim):
    # Equivalent in tensorflow is tf.one_hot
    v = np.zeros(input_dim, dtype=np.float32)
    v[item] = rating
    return v


def sequence_encode(seq):
    usr_grp = seq.groupby(seq.iloc[:, 0], sort=False)
    rows, cols, data = [], [], []
    for i, (user, item_lst) in enumerate(usr_grp):
        n = len(item_lst)
        cols.extend(item_lst.iloc[:, 1].values)
        data.extend(item_lst.iloc[:, 2].values)
        rows.extend([i] * n)

    matrix = csr_matrix((np.array(data), (np.array(rows), np.array(cols))),
                        shape=(len(usr_grp), R_SHAPE[1]),
                        dtype=np.float32)
    return matrix


def validation_sparse_matrix(filename='seq_valid_data', ext='bz2', max_items=None):
    """ Test data """
    score_dataframe = pd.read_table(scoreSet, sep=',', nrows=max_items)

    fp = filename + '.' + ext
    data_file = os.path.join(data_dir, fp)

    try:
        matrix = load(data_file)
    except Exception:
        user_grp = score_dataframe.groupby('new_user', sort=False)
        rows, cols, data = [], [], []
        for i, (user, item_lst) in enumerate(user_grp):
            n = len(item_lst)
            data.extend([preprocess(0)] * n)
            cols.extend(item_lst.new_item.values)
            rows.extend([i] * n)

        matrix = csr_matrix((np.array(data), (np.array(rows), np.array(cols))),
                            shape=(len(user_grp), R_SHAPE[1]),
                            dtype=np.float32)
        dump(matrix, data_file)
    return score_dataframe, matrix


if __name__ == "__main__":
    pass
