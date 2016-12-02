# -*- coding: utf-8 -*-
import os
from time import time

import tensorflow as tf
import numpy as np
import pandas as pd
from joblib import dump, load
from scipy.sparse import csc_matrix, lil_matrix
from sklearn.model_selection import train_test_split

data_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..", "input")

trainSet = os.path.join(data_dir, "customeraffinity.train")
scoreSet = os.path.join(data_dir, "customeraffinity.score")

preprocess = lambda x: (float(x) - 3) / 2
postprocess = lambda x: 2 * x + 3

R_SHAPE = (93705, 3561)


def import_matrix(pr_valid=0.1, do_preprocess=False, return_sparse=True):
    tbl = pd.read_table(trainSet, sep=',', header=None)

    if do_preprocess:
        tbl.iloc[:, -1] = tbl.iloc[:, -1].apply(preprocess)

    train_items, val_items = train_test_split(tbl, test_size=pr_valid)

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
    tensor = tf.SparseTensor(list(zip(
        tbl.iloc[:, 0].values, tbl.iloc[:, 1].values)),
        tbl.iloc[:, -1].values,
        list(R_SHAPE))

    return tensor

# ---- Data as sequences of ratings

def import_sequence(max_items=None, do_preprocess=False):

    t0 = time()
    # Read training data
    tbl = pd.read_table(trainSet, sep=',', header=None)
    # Preprocess if needed
    if do_preprocess: tbl.iloc[:, -1].apply(preprocess)

    n_train = int(max_items) if max_items else len(tbl)
    # matrix = lil_matrix((n_train, 3), dtype=np.float32)
    sp_ids_len = n_train
    curr_client = 0
    user_hist = []
    row_indices = [] #np.empty(sp_ids_len, dtype=np.int32)
    col_indices = [] #np.empty(sp_ids_len, dtype=np.int32)
    data = [] #np.empty(sp_ids_len, dtype=np.int32)
    seq_len = 0
    user_bias = 0

    prev_ptr = 0
    n_ratings = 0 # 2000

    for i in range(n_train):
        user, item, rating = tbl.iloc[i]

        if not seq_len or user == curr_client:
            user_hist.append((item, rating))
            seq_len += 1
            user_bias = ((seq_len-1) * user_bias + rating) / float(seq_len)
        else:  # new client
            # Save data
            items, ratings = zip(*user_hist)
            new_n_ratings = prev_ptr + seq_len
            if new_n_ratings > sp_ids_len:
                row_indices = np.hstack((row_indices, np.empty(sp_ids_len, dtype=np.float32)))
                col_indices = np.hstack((row_indices, np.empty(sp_ids_len, dtype=np.float32)))
                data = np.hstack((row_indices, np.empty(sp_ids_len, dtype=np.float32)))
                sp_ids_len *= 2

            row_indices[prev_ptr:new_n_ratings] = np.full(seq_len, i, dtype=np.float32)
            col_indices[prev_ptr:new_n_ratings] = np.array(items)
            data[prev_ptr:new_n_ratings] = np.array(ratings) - user_bias

            prev_ptr = new_n_ratings
            """
            data.extend(map(lambda r: r - user_bias, ratings))
            col_indices.extend(items)
            row_indices.extend((i,) * len(items))
            """
            # Reset counters
            user_hist = []
            curr_client = user
            user_bias = 0
            seq_len = 0

        if max_items and i > max_items:
            break
        elif i % 100000 == 0:
            print("Processed {} lines".format(i))

    # Return sparse column matrix
    matrix = csc_matrix((data[:prev_ptr], (row_indices[:prev_ptr], col_indices[:prev_ptr])))
    print("[Loaded] {} sequences - took {} s".format(matrix.shape, time() - t0))

    return matrix


def cached_sequence_data(max_items=None, do_preprocess=False, filename='seq_data.bz2'):
    data_file = os.path.join(data_dir, filename)
    if os.path.exists(data_file):
        t0 = time()
        sequences = load(data_file)
        print("[Loaded] {} from cache - took {}".format(data_file, time() - t0))
        if max_items and sequences.shape[0] > max_items:
            sequences = sequences[:max_items]
    else:
        sequences = import_sequence(max_items, do_preprocess)
        dump(sequences, data_file)
    return sequences


def future_generator(X, batch_size, sparse=True, back_to_the_future=False):
    total_samples = X.shape[0]
    n_batches = np.ceil(float(total_samples) / batch_size)
    i = 0
    f = 1 if back_to_the_future else 0
    while True:
        if i > n_batches: i = 0

        x_train = X[i * batch_size:min((i + 1) * batch_size, total_samples), :]
        x_test = X[i * batch_size + f:min((i + 1) * batch_size + f, total_samples), :]

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


def one_hot_encode(item, rating, input_dim, do_preprocess=False):
    # Equivalent in tensorflow is tf.one_hot
    v = np.zeros(input_dim, dtype=np.float32)
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

# ---- Testing data

def import_test():
    tbl = pd.read_table(scoreSet)
    user_arr = tbl.iloc[:, 1].values
    item_arr = tbl.iloc[:, 2].values
    data = np.empty(len(item_arr))
    data.fill(preprocess(0))
    test_matrix = csc_matrix((data, (user_arr, item_arr)))
    return test_matrix


if __name__ == "__main__":
    t = get_score_set()
    t.values.shape
