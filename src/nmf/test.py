import numpy as np
from sklearn.decomposition import NMF
from nmfCF import matrix_factorization

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

X = np.matrix("""1, 3;
                 2, 1;
                 2, 1;
                 2, 1;
                 1, 1;
                 2, 1;
                 3, 1;
                 1, 1""")

encoder = OneHotEncoder()
print(X)
print(X[:, 0])

X_hot = encoder.fit_transform(X[:, 0])
"""

X: label
     1
     2
     2
     1



X_hot :

[
    is_1 is_2
    1    0
    0    1
    0    1
    1    0


X =

[

    day_of_week
     dimanche
     lundi
     dimanche
     lundi
     mardi
]




"""
print(X_hot.toarray())


Y = ["lund", "mard", "lund", "mard", "merc"]



encoder = LabelEncoder()


Y = encoder.fit_transform(Y)
print(Y)
exit()

X = [
    [1, 3, 0, 0, 5],
    [1, 3, 0, 0, 0],
    [1, 3, 0, 0, 5],
    [2, 3, 0, 0, 4]
]

# X = [
#      [5,3,0,1],
#      [4,0,0,1],
#      [1,1,0,5],
#      [1,0,0,4],
#      [0,1,5,4],
# ]

# model = NMF(n_components=2, init='random', random_state=0, alpha = 0.0002, l1_ratio = 0)
# W = model.fit_transform(X)
# H = model.components_
# print(W)
# print(H)


# R = np.matmul(W, H)
# print R.shape

# print "RMSE: " +   str(model.reconstruction_err_)
# print(R.round(4))



print "avec le wesh"
P, Q = matrix_factorization(X, 2)
print np.matmul(P, Q.T)

# print model.components_.shape

# print model.n_components


# data = np.asarray([1, 3, 0, 0, 0]).reshape(1, -1)
# print(model.transform(data))
