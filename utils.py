import numpy as np
from pathlib import Path
import scipy.sparse as sp
from typing import *
import json

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    return False



def load_data(data_set) -> Tuple:
    """
    :param data_set: 数据集名称
    :return: 邻接矩阵和ripple similarity矩阵
    """
    adj_path = Path(__file__).parent / 'data' / (data_set + '_adj.npz')
    ripple_path = Path(__file__).parent / 'data' / (data_set + '_ripple.npz')
    label_path = Path(__file__).parent / 'data' / (data_set + '_labels.npy')
    adj_mat = sp.load_npz(adj_path).astype('float')
    ripple_mat = sp.load_npz(ripple_path)
    labels = np.load(label_path)
    # adj_mat=normalize(adj_mat)
    adj_mat = np.array(adj_mat.todense())
    ripple_mat = np.array(ripple_mat.todense())
    # adj_mat = helper(adj_mat)
    ripple_mat = helper(ripple_mat)
    # ripple_mat += np.diag(np.ones(ripple_mat.shape[0]))
    # if adj_mat[0][0] == 0:
    #     adj_mat += np.diag(np.ones(ripple_mat.shape[0], dtype='int'))
    if max(np.diag(adj_mat))==0:
        adj_mat += np.diag(np.ones(ripple_mat.shape[0], dtype='int'))
    if data_set in {'brazil-airports', 'europe-airports', 'usa-airports'}:
        feat_mat = np.eye(adj_mat.shape[0])
    else:
        feat_path = Path(__file__).parent / 'data' / (data_set + '_features.npz')
        feat_mat = sp.load_npz(feat_path)
        feat_mat = np.array(feat_mat.todense())
    return adj_mat, ripple_mat, feat_mat, labels


def normalize_sp(mx: sp.coo_matrix) -> sp.coo_matrix:
    rows_sum = np.array(mx.sum(1)).astype('float')  # 对每一行求和
    rows_inv = np.power(rows_sum, -1).flatten()  # 求倒数
    rows_inv[np.isinf(rows_inv)] = 0  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    rows_mat_inv = sp.diags(rows_inv)  # 构建对角元素为r_inv的对角矩阵
    mx = rows_mat_inv.dot(mx)  # .dot(cols_mat_inv)
    return mx


def normalize_np(mx: np.ndarray) -> np.ndarray:
    # 对每一行进行归一化
    rows_sum = np.array(mx.sum(1)).astype('float')  # 对每一行求和
    rows_inv = np.power(rows_sum, -1).flatten()  # 求倒数
    rows_inv[np.isinf(rows_inv)] = 0  # 如果某一行全为0，则r_inv算出来会等于无穷大，将这些行的r_inv置为0
    # rows_inv = np.sqrt(rows_inv)
    rows_mat_inv = np.diag(rows_inv)  # 构建对角元素为r_inv的对角矩阵
    mx = rows_mat_inv.dot(mx)  # .dot(cols_mat_inv)
    return mx


def helper(mat: np.ndarray):
    # 每一行除上最大值，进行缩放
    max_vals = np.max(mat, axis=1).astype('float')
    max_vals = np.power(max_vals, -1)
    max_vals[np.isinf(max_vals)] = 0
    return np.diag(max_vals).dot(mat)


def load_params(dataset):
    '''
    导入某个数据集的参数
    :param dataset:
    :return:
    '''
    path = Path(__file__).parent / 'params' / (dataset + '.json')
    f = open(path, 'r')
    params = json.load(f)
    return params


def random_surfing(adj: np.ndarray, epochs: int, alpha: float) -> np.ndarray:
    """
    :param adj: 邻接矩阵，numpy数组,没有经过处理
    :param epochs: 最大迭代次数
    :param alpha: random surf 过程继续的概率
    :return: numpy数组
    """
    # N = adj.shape[0]
    # # 在此进行归一化
    # P0, P = np.eye(N), np.eye(N)
    # mat = np.zeros((N, N))
    # for _ in range(epochs):
    #     P = alpha * adj.dot(P) + (1 - alpha) * P0
    #     mat = mat + P
    # return mat
    # A = normalize_np(adj)
    A = adj.copy()
    # P = np.diag(np.ones(A.shape[0]))
    P = A.copy()
    for _ in range(epochs):
        p1=alpha * A.dot(adj)
        P = p1 + (1 - alpha) * adj
    # 对结果进行正则化
    # mean_val = np.mean(P, axis=0)
    # std_val = np.std(P, axis=0)
    # P = (P - mean_val) / std_val
    # min_val = np.min(P, axis=0)
    # max_val = np.max(P, axis=0)
    # P = (P - min_val) / max_val
    return P


def shuffle(N, num_train, num_val):
    idxs = np.random.permutation(N)
    return idxs[:num_train], idxs[N-num_val:]
