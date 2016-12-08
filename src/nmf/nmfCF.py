from numpy import dot
import numpy as np
from scipy import linalg


def matrix_factorization(R, K, steps=5000, alpha=0.0002, beta=0.02):
    R = np.asarray(R)
    P = np.random.rand(R.shape[0], K)
    Q = np.random.rand(R.shape[1], K)
    Q = Q.T

    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - np.dot(P[i, :], Q[:, j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = np.dot(P, Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + np.pow(R[i][j] - np.dot(P[i, :], Q[:, j]), 2)
                    for k in xrange(K):
                        e = e + (beta / 2) * (pow(P[i][k], 2) + pow(Q[k][j], 2))
        if e < 0.001:
            break
    return P, Q.T, e


def nmf_collaborative_filtering(R, K, steps=5000, alpha=0.0002, beta=0.02):
    R = np.asarray(R)
    P = np.random.rand(R.shape[0], K)
    Q = np.random.rand(R.shape[1], K)
    Q = Q.T

    nUser = R.shape[0]
    nItem = R.shape[1]

    rated = R > 0
    for step in xrange(steps):

        for i in xrange(K):
            error = np.sum(R[rated] - np.matmul(P, Q)[rated])

        for i in xrange(K):
            P[:, rated[:, i]] = P[:, rated[:, i]] + alpha * (2 * eij * Q - beta * P)
            Q[rated[i], :] = Q[rated[i], :] + alpha * (2 * eij * P - beta * Q)


def nmf(X, latent_features, max_iter=100, error_limit=1e-6, fit_error_limit=1e-6):
    """
    Decompose X in A*Y
    """
    eps = 1e-5
    print 'Starting NMF decomposition with {} latent features and {} iterations.'.format(latent_features, max_iter)
    X = X.toarray()  # I am passing in a scipy sparse matrix

    # mask
    mask = np.sign(X)

    # initial matrices. A is random [0,1] and Y is A\X.
    rows, columns = X.shape
    A = np.random.rand(rows, latent_features)
    A = np.maximum(A, eps)

    Y = linalg.lstsq(A, X)[0]
    Y = np.maximum(Y, eps)

    # bool_mask = mask.astype(bool);
    # for i in range(columns):
    #     Y[:, i] = linalg.lstsq(A[bool_mask[:, i], :], X[bool_mask[:, i], i])[0]


    masked_X = mask * X
    X_est_prev = dot(A, Y)
    for i in range(1, max_iter + 1):
        # ===== updates =====
        # Matlab: A=A.*(((W.*X)*Y')./((W.*(A*Y))*Y'));
        top = dot(masked_X, Y.T)
        bottom = (dot((mask * dot(A, Y)), Y.T)) + eps
        A *= top / bottom

        A = np.maximum(A, eps)
        # print 'A',  np.round(A, 2)

        # Matlab: Y=Y.*((A'*(W.*X))./(A'*(W.*(A*Y))));
        top = dot(A.T, masked_X)
        bottom = dot(A.T, mask * dot(A, Y)) + eps
        Y *= top / bottom
        Y = np.maximum(Y, eps)
        # print 'Y', np.round(Y, 2)


        # ==== evaluation ====
        if i % 5 == 0 or i == 1 or i == max_iter:
            print 'Iteration {}:'.format(i),
            X_est = dot(A, Y)
            err = mask * (X_est_prev - X_est)
            fit_residual = np.sqrt(np.sum(err ** 2))
            X_est_prev = X_est

            curRes = linalg.norm(mask * (X - X_est), ord='fro')
            print 'fit residual', np.round(fit_residual, 4),
            print 'total residual', np.round(curRes, 4)
            if curRes < error_limit or fit_residual < fit_error_limit:
                break

    return A, Y, fit_residual
