# -*- coding: utf-8 -*-
import os
from time import time

import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load
from scipy.sparse.linalg import svds
from sklearn.decomposition import nmf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

from src.data import scoreSet, import_matrix, data_dir


def run_nmf(matrix, init_='nndsvdar', alpha_=0.1, l1_ratio_=0.0, latent_dim=100):
    """ Find W and H such that  W H.T ~ matrix with error minimized
    /!\ Each run needs 4Gb of memory
    :param matrix: Matrix to be factorized
    :param init_: Initialization method
    :param alpha_: Regularization coef
    :param l1_ratio_: Ratio of l1-norm regularizon (0.0 -> only l2-reg)
    :param latent_dim:
    :return: latent factorization, reconstruction error and number of iterationsx"""
    estimator = nmf.NMF(n_components=latent_dim, init=init_, max_iter=2000,
                        alpha=alpha_, l1_ratio=l1_ratio_, shuffle=True)
    print(estimator.get_params())
    W = estimator.fit_transform(matrix)  # (n_samples, n_components)
    H = estimator.components_  # (n_components, n_features)
    return (W, H), estimator.reconstruction_err_, estimator.n_iter_


def nmf_wrapper():
    # training_set_stats()
    # Files paths
    data_file = data_dir + '/matrices.bz2'
    nmf_model_file = data_dir + '/model_params'
    p_valid = 0.1

    # --- Load data
    if os.path.exists(data_file):
        training_mat, validation_set = load(data_file)
    else:
        training_mat, validation_set = import_matrix()
        dump((training_mat, validation_set), data_file)

    ''' Run and save params '''
    if os.path.exists(nmf_model_file + '.bz2'):
        print('Loading components from file.')
        (W, H) = load(nmf_model_file + '.bz2')
    else:
        print('Running NMF.')
        (W, H), err, (_, _, _, _, iters) = run_nmf(training_mat)
        dump((W, H), "{}_{}_{}.xz".format(nmf_model_file, int(err), 1 - p_valid))
        print("Reconstruction error = {} in {} iterations".format(err, iters))

    validation_score = mf_val_rmse(W, H, validation_set)
    print("Validation RMSE={}".format(validation_score))
    test(W, H, scoreSet)


def run_grid(data):
    params_dict = {
        # 'alpha': uniform.rvs(size=3),
        # 'l1_ratio': uniform.rvs(size=3),
        'inits': ['random', 'nndsvd'],
        'dims': [20, 30, 40],
        'alphas': [0.001, 0.01, 0.1, 0.2, 0.5, 1.0],
        'l1_ratio': [0.0, 0.5, 1.0]
    }
    clf = GridSearchCV(nmf.NMF(max_iter=1000, shuffle=True),
                       param_grid=params_dict, n_jobs=3,
                       scoring=mean_squared_error)
    clf.fit(data)
    return clf


# ---- Validation functions

def mf_val_rmse(w, h, validation_ratings):
    return np.sqrt(np.mean([(np.dot(w[user], h.T[item]) - rating) ** 2
                            for (user, item, rating) in validation_ratings]))


def test(w, h, scoreSet):
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


def test_svd(max_iters, k=200):
    # Run k-SVD
    training_matrix, test_matrix = import_matrix()
    rows, cols = test_matrix.nonzero()
    test_lst = []
    for i in xrange(rows.shape[0]):
        test_lst.append((rows[i], cols[i], test_matrix[rows[i], cols[i]]))

    t0 = time()
    p, d, q = svds(training_matrix, k, maxiter=max_iters)
    print('[SVD-{}] Computed {} iters in {} s'.format(k, max_iters, time() - t0))
    loss = mf_val_rmse(p.dot(d), q, test_lst)
    print(loss)


if __name__ == "__main__":
    nmf_wrapper()
