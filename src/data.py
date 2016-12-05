# -*- coding: utf-8 -*-
import os
from collections import defaultdict
from time import time

import tensorflow as tf
import numpy as np
import pandas as pd
from joblib import dump, load
from scipy.sparse import csc_matrix, lil_matrix, csr_matrix
from sklearn.model_selection import train_test_split

data_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..", "input")

trainSet = os.path.join(data_dir, "customeraffinity.train")
scoreSet = os.path.join(data_dir, "customeraffinity.score")


def preprocess(x):
    pre = lambda x: (float(x) - 3) / 2
    return generic_apply(pre, x)


def postprocess(x):
    post = lambda x: 2 * x + 3
    return generic_apply(post, x)


def generic_apply(func, data):
    if isinstance(data, (int, float, np.int32, np.int64, np.float32, np.float64)):
        return func(data)
    elif isinstance(data, list):
        return [func(v) for v in data]
    elif isinstance(data, np.ndarray):
        vfunc = np.vectorize(func)
        return vfunc(data)
    elif isinstance(data, pd.Series):
        return data.apply(func)
    else:
        raise TypeError('unsupported {}'.format(type(data)))


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

def import_sequence(max_items=None, do_preprocess=False, test_ratio=0.1):
    t0 = time()
    # Read training data
    tbl = pd.read_table(trainSet, sep=',', header=None)
    # Preprocess if needed
    if do_preprocess: tbl.iloc[:, -1].apply(preprocess)

    # Split the table in train-test
    test_tbl = None
    if test_ratio:
        length = len(tbl)
        all_indices = list(range(length))
        test_indices = np.random.choice(all_indices, int(test_ratio * length), replace=False)
        train_indices = list(set(all_indices).difference(test_indices))
        test_tbl = tbl.iloc[test_indices, :]
        tbl = tbl.iloc[train_indices, :]
        print("[Split] {} train, {} test ".format(len(train_indices), len(test_indices)))

    n_train = int(max_items) if max_items else len(tbl)
    # matrix = lil_matrix((n_train, 3), dtype=np.float32)
    tot_n_elts = n_train
    curr_client = 0
    user_hist = []
    row_indices = []  # np.empty(sp_ids_len, dtype=np.int32)
    col_indices = []  # np.empty(sp_ids_len, dtype=np.int32)
    data = []  # np.empty(sp_ids_len, dtype=np.int32)
    seq_len = 0
    user_bias = 0.0

    previous_index = 0
    n_ratings = 0  # 2000

    for i in range(n_train):
        user, item, rating = tbl.iloc[i]

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

            row_indices[previous_index:new_n_ratings] = np.full(seq_len, i, dtype=np.float32)
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

        if max_items and i > max_items:
            break
        elif i % 100000 == 0:
            print("Processed {} lines".format(i))

    # Return sparse column matrix
    matrix = csr_matrix((data[:previous_index], (row_indices[:previous_index], col_indices[:previous_index])),
                        dtype=np.float32)
    print("[Loaded] {} sequences - took {} s".format(matrix.shape, time() - t0))

    return matrix, test_tbl


def cached_sequence_data(max_items=None, test_ratio=0.1, do_preprocess=False, filename='seq_data', ext='bz2'):
    fp = '{}_{}.{}'.format(filename, 'pproc' if do_preprocess else 'raw', ext)
    data_file = os.path.join(data_dir, fp)
    if os.path.exists(data_file):
        t0 = time()
        sequences, test_tbl = load(data_file)
        print("[Loaded] {} from cache - took {}".format(data_file, time() - t0))
        if max_items and sequences.shape[0] > max_items:
            sequences = sequences[:max_items]
    else:
        sequences, test_tbl = import_sequence(max_items, do_preprocess, test_ratio)
        dump((sequences, test_tbl), data_file)
    return sequences, test_tbl


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


def import_validation_sparse(filename='seq_valid_data', ext='bz2', max_items=None):
    """ Test data """
    tbl = pd.read_table(scoreSet, sep=',', nrows=max_items)

    fp = filename+'.'+ext
    data_file = os.path.join(data_dir, fp)

    try:
        matrix = load(data_file)
    except Exception:
        user_grp = tbl.groupby('new_user', sort=False)
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
    return tbl, matrix


if __name__ == "__main__":
    pass
