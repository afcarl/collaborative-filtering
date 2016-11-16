# -*- coding: utf-8 -*-
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds

from .matrix_factorization import run_nmf

data_dir = os.path.split(os.path.realpath(__file__))[0] + "/../input"

trainSet = data_dir + "/customeraffinity.train"
scoreSet = data_dir + "/customeraffinity.score"


def import_train(pr_valid=0.1):
    train_mat = np.zeros((93705, 3562), dtype=np.int32)
    validation_lst = []

    with open(trainSet) as train_file:
        lines = train_file.readlines()
        for line in lines[1:]:
            splits = [int(s) for s in line.split(",")]
            if np.random.binomial(1, pr_valid):
                validation_lst.append(tuple(splits))
            else:
                train_mat[splits[0], splits[1]] = splits[2] - 1

        print("Data has {} samples, {} for validation.".format(len(lines), len(validation_lst)))
    return csc_matrix(train_mat), validation_lst


def import_test():
    tbl = pd.read_table(scoreSet)
    user_arr = tbl.iloc[:, 1].values
    item_arr = tbl.iloc[:, 2].values
    data = np.ones(len(item_arr))
    test_matrix = csc_matrix(data, (user_arr, item_arr))
    return test_matrix


def training_set_stats():
    """
    Load customer affinity data (train and test sets), convert item to sparse matrix
    :return: train and test sparse matrices
    """
    item_multiplicities = defaultdict(int)
    user_multiplicities = defaultdict(int)
    m1, m2 = 0, 0
    s1, s2 = set(), set()
    hist = [0] * 5
    lines = 0
    with open(trainSet) as train_file:
        for line in train_file:
            if not lines:
                lines += 1
                continue
            splits = [int(s) for s in line.split(",")]
            item = splits[1]
            user = splits[0]
            item_multiplicities[item] += 1
            user_multiplicities[user] += 1
            m1 = max(m1, user)
            m2 = max(m2, item)
            hist[splits[2] - 1] += 1
            s1.add(user)
            s2.add(item)
            lines += 1

    n_s1, n_s2 = len(s1), len(s2)

    print("Number of entries: {}".format(lines))
    print("Max ids are {} and {}".format(m1, m2))
    print("Number of unique ids are {} and {}".format(n_s1, n_s2))
    print("Sparsity = {}".format(lines / (n_s1 * n_s2)))  # How sparse is the data?

    plt.scatter(np.arange(5), hist)
    plt.show()

    hist = [round(h / sum(hist), 4) for h in hist]
    print(hist)
    item_dist = np.array(list(item_multiplicities.values()))
    user_dist = np.array(list(user_multiplicities.values()))
    print_moments(item_dist)
    print_moments(user_dist)

    # Number of times each item was seen
    plt.hist(item_dist, bins=100)
    plt.show()
    # Number of times each user was seen
    plt.hist(user_dist, bins=100)
    plt.show()


def print_moments(distrib):
    print('Min = {}, Max = {}, Mean = {}, Median = {}, Var = {}, Std = {}'.format(distrib.min(),
                                                                                  distrib.max(),
                                                                                  distrib.mean().round(1),
                                                                                  np.median(distrib),
                                                                                  distrib.var().round(1),
                                                                                  distrib.std().round(1)))


def validation_rmse(w, h, validation_ratings):
    return np.sqrt(np.mean([(np.dot(w[user], h.T[item]) - rating) ** 2 for (user, item, rating) in validation_ratings]))


def test(w, h):
    with open(scoreSet) as test_file:
        lines = test_file.readlines()
        test_ratings = np.empty(len(lines) - 1, dtype=np.float64)
        for i, line in enumerate(lines[1:]):
            splits = [int(s) for s in line.split(",")]
            test_ratings[i] = np.dot(w[splits[1]], h.T[splits[2]])
        # Plot the distribution
        plt.hist(test_ratings, bins=100)
        plt.show()

        return test_ratings


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
        training_mat, validation_set = import_train()
        dump((training_mat, validation_set), data_file)

    # Run k-SVD
    p, d, q = svds(training_mat, 40)

    ''' Run and save params '''
    if os.path.exists(nmf_model_file + '.xz'):
        (W, H) = load(nmf_model_file + '.xz')
    else:
        (W, H), err, (_, _, _, _, iters) = run_nmf(training_mat, l1_ratio_=0.5, latent_dim=40, init_='nndsvd')
        dump((W, H), "{}_{}_{}.xz".format(nmf_model_file, int(err), 1 - p_valid))
        print("Reconstruction error = {} in {} iterations".format(err, iters))

    validation_score = validation_rmse(W, H, validation_set)
    print("Validation RMSE={}".format(validation_score))
    test(W, H)
