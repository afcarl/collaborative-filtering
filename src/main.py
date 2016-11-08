# -*- coding: utf-8 -*-
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from joblib import dump, load
from scipy.sparse import csc_matrix
from scipy.stats import uniform
from sklearn.decomposition import nmf
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error


data_dir = os.path.split(os.path.realpath(__file__))[0] + "/../input"

trainSet = data_dir + "/customeraffinity.train"
scoreSet = data_dir + "/customeraffinity.score"

def import_data(pr_valid=0.1):
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


def test_set_stats():
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
    plt.hist(item_dist[item_dist < 71], bins=100)
    plt.show()

    plt.hist(item_dist[item_dist > 71], bins=100)
    plt.show()

    # Number of times each user was seen
    plt.hist(user_dist[user_dist < 13], bins=100)
    plt.show()

    plt.hist(user_dist[user_dist > 13], bins=100)
    plt.show()


def print_moments(distrib):
    print('Min = {}, Max = {}, Mean = {}, Median = {}, Var = {}, Std = {}'.format(distrib.min(),
                                                                                  distrib.max(),
                                                                                  distrib.mean().round(1),
                                                                                  np.median(distrib),
                                                                                  distrib.var().round(1),
                                                                                  distrib.std().round(1)))


def run_nmf(matrix, init_='nndsvdar', alpha_=0.01, l1_ratio_=0.1, latent_dim=50):
    """ Find W and H such that  W H.T ~ matrix with error minimized /!\ Chaque run utilise 3860Mb """
    estimator = nmf.NMF(n_components=latent_dim,
                        init=init_,
                        max_iter=2000,
                        alpha=alpha_,  # Regularization coef
                        l1_ratio=l1_ratio_,  # Ratio of l1-norm regularizon (0.0 -> only l2-reg)
                        shuffle=True)
    W = estimator.fit_transform(matrix)  # (n_samples, n_components)
    H = estimator.components_  # (n_components, n_features)
    return (W, H), estimator.reconstruction_err_, (init_, alpha_, l1_ratio_, latent_dim, estimator.n_iter_)  # (W, H),


def run_grid(data):
    params_dict = {
        'alpha': uniform.rvs(size=3),
        'l1_ratio': uniform.rvs(size=3),
    }
    # also, http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html
    # pipelines are efficient! http://scikit-learn.org/stable/modules/pipeline.html
    randomized_search = GridSearchCV(nmf.NMF(max_iter=1000, shuffle=True), param_grid=params_dict, n_jobs=3,
                                     scoring=mean_squared_error)
    randomized_search.fit(data)
    print(randomized_search.__dict__)


def run_grid2(data):
    inits = ['random', 'nndsvd']
    dims = [20, 30, 40]
    alphas = [0.001, 0.01, 0.1, 0.2, 0.5, 1.0]
    l1_ratio = [0.0, 0.5, 1.0]

    grid_results = Parallel(n_jobs=6, verbose=1)(delayed(
        run_nmf)(data, i, a, r, d) for i in inits for a in alphas for r in l1_ratio for d in dims)

    results = {error: params for error, params in grid_results}

    for key in sorted(results):
        _, init, alpha, l1_ratio, dim, n_iter = results[key]
        sys.stdout.write(
            '[NMF] Init={}, Alpha={}, L1-ratio={}, Dim={} : Converged to {} reconstruction error in {} iterations\n'
                .format(init, alpha, l1_ratio, dim, key, n_iter))
        sys.stdout.flush()


def validation_rmse(w, h, validation_ratings):
    return np.sqrt(np.mean([(np.dot(w[user], h.T[item]) - rating) ** 2 for (user, item, rating) in validation_ratings]))


def test(w, h):
    with open('../input/customeraffinity.score') as test_file:
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
    model_file = data_dir + '/model_params'
    p_valid = 0.1

    # --- Load data
    if os.path.exists(data_file):
        training_mat, validation_set = load(data_file)
    else:
        training_mat, validation_set = import_data()
        dump((training_mat, validation_set), data_file)

    ''' Run and save params '''
    if os.path.exists(model_file + '.xz'):
        (W, H) = load(model_file + '.xz')
    else:
        (W, H), err, (_, _, _, _, iters) = run_nmf(training_mat, l1_ratio_=0.5, latent_dim=40, init_='nndsvd')
        dump((W, H), "{}_{}_{}.xz".format(model_file, int(err), 1- p_valid))
        print("RMSE={} in {} iterations".format(err, iters))

    validation_score = validation_rmse(W, H, validation_set)
    print("Validation RMSE={}".format(validation_score))
    test(W, H)
