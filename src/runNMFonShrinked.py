from nmfCF import matrix_factorization
from scipy.sparse import csc_matrix
from time import time
import pandas as pd
import numpy as np

from nmfCF import matrix_factorization, nmf

def mat_from_rating_sequence_sparse(file, max_items=None, do_preprocess=False):

    t0 = time()
    # Read training data
    tbl = pd.read_csv(file, header=None)

    n_train = int(max_items) if max_items else len(tbl)

    curr_client = 0
    user_hist = []
    # typ: ((index, item), value)
    row_indices = []  # np.empty(nnz_elts, dtype=np.int32)
    col_indices = []
    data = []

    for i in range(n_train):
        user, item, rating = tbl.iloc[i]

        if user == curr_client:
            user_hist.append((item, rating))
            items, ratings = zip(*user_hist)
            n_seq = len(user_hist)
            data.extend(ratings)
            col_indices.extend(items)
            row_indices.extend((i,) * len(items))
        else:  # changed clients
            user_hist = []
            curr_client = user

    # Return sparse column matrix
    matrix = csc_matrix((data, (row_indices, col_indices)))

    return matrix


def mat_from_rating_sequence(file):

    return mat_from_rating_sequence_sparse(file).todense()

if __name__ == "__main__":
    latent_features = 300
    sequenceFile = "input/customeraffinity30k.train"

    sparse = mat_from_rating_sequence_sparse(sequenceFile)
    # matrix = mat_from_rating_sequence(sequenceFile)
    # P,Q, e = matrix_factorization(matrix, latent_features)

    P, Q, e = nmf(sparse, latent_features, max_iter=100, error_limit=1e-6, fit_error_limit=1e-6)

    print(e)


