# -*- coding: utf-8 -*-
import sys
from scipy.stats import uniform
from sklearn.decomposition import nmf
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error
from joblib import Parallel, delayed


def run_nmf(matrix, init_='nndsvdar', alpha_=0.01, l1_ratio_=0.0, latent_dim=100):
    """ Find W and H such that  W H.T ~ matrix with error minimized
    /!\ Chaque run utilise 3860Mb
    :param matrix: Matrix to be factorized
    :param init_: Initialization method
    :param alpha_: Regularization coef
    :param l1_ratio_: Ratio of l1-norm regularizon (0.0 -> only l2-reg)
    :param latent_dim:
    :return: """
    estimator = nmf.NMF(n_components=latent_dim, init=init_, max_iter=2000,
                        alpha=alpha_, l1_ratio=l1_ratio_, shuffle=True)
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
    clf = GridSearchCV(nmf.NMF(max_iter=1000, shuffle=True), param_grid=params_dict, n_jobs=3,
                                     scoring=mean_squared_error)
    clf.fit(data)
    return clf


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
