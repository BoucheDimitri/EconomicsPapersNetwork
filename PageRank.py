import numpy as np
import pandas as pd
import os
import scipy.sparse as sparse
import scipy.sparse.linalg as linalg
import CitNet.GraphCN as GraphCN
import networkx as nx
import scipy.stats as ss

# Values for test
row = np.array([0, 0, 1, 2, 2, 2])
col = np.array([0, 2, 2, 0, 1, 2])
data = np.array([1, 2, 3, 4, 5, 6])

a_wodang = sparse.csc_matrix((data, (row, col)), (4, 4))
a_wdang = sparse.csc_matrix((data, (row, col)), (5, 5))


def get_dangvec(adj_mat):
    """
    Returns the vector of dangling nodes (0 if False, 1 if Trues)
    :param adj_mat: (scipy.sparse.csc.csc_matrix) n x n
    :return: (scipy.sparse.csc.csc_matrix) n x 1
    """
    dang_vec = sparse.csc_matrix(adj_mat.sum(1) == 0, dtype=int)
    return dang_vec


def get_dangmat(adj_mat):
    """
    Returns the matrix of dangling nodes mandatory score spreading
    :param adj_mat: (scipy.sparse.csc.csc_matrix) n x n
    :return: (scipy.sparse.csc.csc_matrix) n x n
    """
    dang_vec = get_dangvec(adj_mat)
    one_vec = sparse.csc_matrix(np.ones(adj_mat.shape[0]))
    dang_mat = dang_vec.dot(one_vec) * 1/adj_mat.shape[0]
    return dang_mat


def get_hmat(adj_mat):
    """
    Returns the H mat (normalized adj_mat + mandatory score spreading)
    :param adj_mat: (scipy.sparse.csc.csc_matrix) n x n
    :return: (scipy.sparse.csc.csc_matrix) n x n
    """
    int_mat = get_dangmat(adj_mat) + adj_mat
    h_mat = sparse.csc_matrix(int_mat / int_mat.sum(axis=1))
    return h_mat


def get_gmat(adj_mat, theta):
    """
    Returns the G mat ((1-theta) * 1/N.1.1T + theta * H)
    :param adj_mat: (scipy.sparse.csc.csc_matrix) n x n
    :param theta: (numeric) damping factor
    :return: (scipy.sparse.csc.csc_matrix) n x n
    """
    n = adj_mat.shape[0]
    h_mat = get_hmat(adj_mat)
    one_vec = sparse.csc_matrix(np.ones(n)).T
    g_mat = (1-theta)/n * one_vec.dot(one_vec.T) + theta * h_mat
    return g_mat


def get_pagerank(adj_mat, theta=.85, epsilon=1e-03, max_iter=20):

    """
    Returns the vector of pagerank scores
    :param adj_mat: (scipy.sparse.csc.csc_matrix) n x n
    :param theta: (numeric) damping factor
    :return: vector of pagerank scores n x 1
    """
    n = adj_mat.shape[0]
    g_mat = get_gmat(adj_mat, theta)
    pr_vec = sparse.csc_matrix(np.ones(n))/n
    norm_iter = adj_mat.shape[0]
    n = adj_mat.shape[0]

    i = 0
    norm = []

    while (norm_iter > epsilon*n) and (i < max_iter):
        pr_iter = pr_vec.dot(g_mat)
        norm_iter = linalg.norm(pr_vec - pr_iter)
        pr_vec = pr_iter

        i += 1
        norm += [norm_iter]
        print("iter {0}: {1}".format(i, norm_iter))
    return pr_vec.T

####
# Unfortunately, much less efficient than the networkx implementation
####
