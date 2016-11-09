# -*- coding: utf-8 -*-
# Author: Nithin Holla
# License: BSD 3-clause
import numpy as np


class AUCB(object):
    def __init__(self, **kwargs):
        # Constants
        self.n_arms = kwargs['K']
        self.n_users = kwargs['N']
        self.gamma = kwargs['gamma']
        self.R = kwargs['R']
        # Variables 
        self.U = None
        self.L = None
        self.N = None
        self.Nb = None
        self.cluster_assign = None
        self.epsilon = None
        self.past_users = None
        self.means = None

    def initialize(self):
        self.means = np.zeros((self.n_users, self.n_arms))

        self.U = np.empty((self.n_users, self.n_arms), dtype=np.float64)
        self.U.fill(np.infty)

        self.L = np.empty((self.n_users, self.n_arms), dtype=np.float64)
        self.L.fill(-np.infty)

        self.N = np.zeros((self.n_users, self.n_arms), dtype=np.int32)
        self.Nb = np.zeros(self.n_users, dtype=np.int32)
        self.cluster_assign = np.zeros(self.n_users, dtype=np.int32)
        self.epsilon = np.zeros((self.n_users, self.n_users))
        self.past_users = set()

    # Choose the best arm as per the A-UCB
    def select_arm(self, *args):
        user = args[0]
        possible_clusters = set()
        c_users = []
        if self.Nb[user] == 0:
            possible_clusters.add(user + 1)
        else:
            for past_user in self.past_users:
                # Calculate epsilon for user pairs in past users
                temp = np.sqrt((2 * self.gamma * np.log(self.Nb[past_user] + 1)) / (np.log(self.Nb[user] + 1))) - 1
                self.epsilon[user, past_user] = np.append(temp, 0).max()

                # Obtain the clusters to which the user might belong
                U_scaled = self.means[past_user] + (1 + self.epsilon[user, past_user]) * \
                                                   (self.U[past_user] - self.means[past_user])
                L_scaled = self.means[past_user] - (1 + self.epsilon[user, past_user]) * \
                                                   (self.means[past_user] - self.L[past_user])

                if self.check_grouping(self.L[user], self.U[user],self.L[past_user], self.U[past_user], L_scaled, U_scaled):
                    clustered_users = np.where(self.cluster_assign == self.cluster_assign[past_user])[0]
                    flag = True
                    for member in clustered_users:
                        if member != past_user and member != user:
                            U_scaled = self.means[member] + (1 + self.epsilon[user, member]) * \
                                                            (self.U[member] - self.means[member])
                            L_scaled = self.means[member] - (1 + self.epsilon[user, member]) * \
                                                            (self.means[member] - self.L[member])
                            if not self.check_grouping(self.L[user], self.U[user], self.L[member], self.U[member],
                                                       L_scaled, U_scaled):
                                flag = False
                                break
                    if flag and past_user != user:
                        possible_clusters.add(self.cluster_assign[past_user])

                if len(possible_clusters) == 0:
                    possible_clusters.add(user + 1)

        # Obtain the biggest cluster and assign the user to it
        clust_id = 0
        biggest_cluster_len = 0
        for cluster in possible_clusters:
            c_users = np.where(self.cluster_assign == cluster)
            if len(c_users) > biggest_cluster_len:
                biggest_cluster_len = len(c_users)
                clust_id = cluster
                c_users = np.where(self.cluster_assign == clust_id)[0]
                if user not in c_users:
                    c_users = np.hstack([c_users, user])
        self.cluster_assign[user] = clust_id

        # Aggregate means and the upper bounds from users in the cluster
        mean_aggr = np.zeros(self.n_arms)
        U_aggr = np.zeros(self.n_arms)
        if len(c_users) > 1:  # Nusers in biggest cluster
            for arm in range(self.n_arms):
                mean_aggr[arm] = np.ravel(np.sum(self.means[c_users, arm] * self.N[c_users, arm]) / \
                                          (np.sum(self.N[c_users, arm])) + 1)
                b = np.sqrt(2 * np.log(np.sum(self.Nb[c_users]) ** 3) / (np.sum(self.N[c_users, arm]) + 1))
                U_aggr[arm] = mean_aggr[arm] + self.R * b
            indices = np.where(U_aggr == np.max(U_aggr))[0]
            return indices[np.random.randint(0, len(indices))]
        else:
            for arm in range(self.n_arms):
                if self.N[user, arm] == 0:
                    return arm

            return self.U[user].argmax()

    # Check the condition for grouping (subset condition)
    @staticmethod
    def check_grouping(set1l, set1u, set2l, set2u, set_exp_l, set_exp_u):
        flag = False
        for i in range(len(set1l)):
            if set1l[i] < set_exp_l[i] or set1u[i] > set_exp_u[i]:
                return False
            if (set2l[i] <= set1l[i] <= set2u[i]) or (set2u[i] > set1u[i] > set2l[i]):
                flag = True

        return flag

    # Compute the upper and lower bounds
    def compute_bounds(self, user):
        for i in range(self.n_arms):
            draws_constant = 2 * np.log(self.Nb[user] ** 3 + 1)
            if self.N[user, i] == 0:
                self.U[user, i] = self.means[user, i] + self.R * np.sqrt(draws_constant)
                self.L[user, i] = self.means[user, i] - self.R * np.sqrt(draws_constant)
            else:
                self.U[user, i] = self.means[user, i] + self.R * np.sqrt(draws_constant / self.N[user, i])
                self.L[user, i] = self.means[user, i] - self.R * np.sqrt(draws_constant / self.N[user, i])

    # Update the class variables after observation of reward
    def update(self, usr, arm, reward):
        self.N[usr, arm] += 1
        self.Nb[usr] += 1
        n = self.N[usr, arm]
        self.means[usr, arm] = ((n - 1) / float(n)) * self.means[usr, arm] + (1 / float(n)) * reward
        self.past_users.add(usr)
        self.compute_bounds(usr)

    def __str__(self):
        return 'Latent Bandits'
