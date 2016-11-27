# -*- coding: utf-8 -*-
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split

from matrix_factorization import run_nmf, mf_val_rmse, test

data_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], "..", "input")

trainSet = os.path.join(data_dir, "customeraffinity.train")
scoreSet = os.path.join(data_dir, "customeraffinity.score")

preprocess = lambda x: (float(x) - 3) / 2
postprocess = lambda x: 2 * x + 3


def import_matrix(pr_valid=0.1, do_preprocess=False, return_sparse=True):
    train_mat = csr_matrix((93705, 3562), dtype=np.int32)
    valid_mat = csr_matrix((93705, 3562), dtype=np.int32)

    tbl = pd.read_table(trainSet, sep=',', header=None)
    tbl.iloc[:, -1] = tbl.iloc[:, -1].apply(preprocess)

    train_items, val_items = train_test_split(tbl, test_size=pr_valid)

    train_mat = to_matrix(train_items)
    valid_mat = to_matrix(val_items)

    if not return_sparse:
        train_mat = train_mat.toarray()
        valid_mat = valid_mat.toarray()

    return train_mat, valid_mat


def to_matrix(tbl, shape=(93705, 3562)):
    user_arr = tbl.iloc[:, 0].values
    item_arr = tbl.iloc[:, 1].values
    data = tbl.iloc[:, -1].values

    matrix = csc_matrix((data, (user_arr, item_arr)))

    return matrix


# ---- Data as sequences of ratings

def import_sequence(max_items=None, val_ratio=0.1):
    tbl = pd.read_table(trainSet, sep=',', header=None)
    tbl.iloc[:, -1].apply(preprocess)
    train_seq, val_seq = train_test_split(tbl, test_size=val_ratio)
    n_train = max_items or len(train_seq)
    training_set = lil_matrix((n_train, 3562), dtype=np.float32)

    curr_client = 0
    rating_lst = []

    for i in range(n_train):
        user, item, rating = train_seq.iloc[i]

        if user == curr_client:
            rating_lst.append((item, rating))
        else:  # changed clients
            if rating_lst:
                for j in range(len(rating_lst)):
                    for (it, r) in rating_lst[:j]:
                        training_set[i, it] = r
                        # else:
                        #    validation_set.append((rating_lst[:j], rating_lst[j]))
                rating_lst = []
            curr_client = user

        if max_items and i > max_items:
            break
        elif i % 100000 == 0:
            print("Processed {} lines".format(i))

    return training_set, val_seq


def cached_sequence_data(max_items=None):
    data_file = os.path.join(data_dir, 'seq_data.bz2')
    if os.path.exists(data_file):
        train, val = load(data_file)
    else:
        train, val = import_sequence(max_items)
        dump((train, val), data_file)
    return train, val


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
    # training_set_stats()
    # Files paths
    data_file = data_dir + '/matrices.xz'
    nmf_model_file = data_dir + '/model_params'
    p_valid = 0.1

    # --- Load data
    if os.path.exists(data_file):
        training_mat, validation_set = load(data_file)
    else:
        training_mat, validation_set = import_matrix()
        dump((training_mat, validation_set), data_file)

    # Run k-SVD
    p, d, q = svds(training_mat, 40)

    ''' Run and save params '''
    if os.path.exists(nmf_model_file + '.xz'):
        (W, H) = load(nmf_model_file + '.xz')
    else:
        (W, H), err, (_, _, _, _, iters) = run_nmf(training_mat)
        dump((W, H), "{}_{}_{}.xz".format(nmf_model_file, int(err), 1 - p_valid))
        print("Reconstruction error = {} in {} iterations".format(err, iters))

    validation_score = mf_val_rmse(W, H, validation_set)
    print("Validation RMSE={}".format(validation_score))
    test(W, H)
