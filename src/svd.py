# -*- coding: utf-8 -*-
from scipy.sparse.linalg import svds


def compute_svd(sparse_mat, k):
    p, d, q = svds(sparse_mat, k)
    return p, d, q
