# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import nmf
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


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


def run_grid(data):
    params_dict = {
        #'alpha': uniform.rvs(size=3),
        #'l1_ratio': uniform.rvs(size=3),
        'inits': ['random', 'nndsvd'],
        'dims' : [20, 30, 40],
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

        
