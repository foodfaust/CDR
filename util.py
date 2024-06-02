# import dgl
from IPython import embed
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import pickle
import numpy as np
from torch.utils.data import Dataset
import torch.utils.data as data
import scipy.sparse as sp
import os
import gc
import configparser

from scipy.sparse import csr_matrix, lil_matrix
from tqdm import tqdm
import csv
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import normalize


# from typing import Any
# from torch.utils.data.dataloader import _DatasetKind
# import warnings
# from sklearn.utils import shuffle as reset


# from data.youshu.pro import pretrain_data_pro


# np.random.seed(1)


def beta_cat(left_top, right_top, left_bottom, right_bottom):
    beta_top = sp.hstack((left_top, right_top))
    beta_bottom = sp.hstack((left_bottom, right_bottom))
    beta = sp.vstack((beta_top, beta_bottom))
    return beta


def matrix_cat(n_node_u, n_node_i, inter_M, inter_M_t):
    top_left = np.eye(n_node_u, dtype=int)
    bottom_right = np.eye(n_node_i, dtype=int)
    top = np.concatenate((top_left, inter_M), axis=1)
    bottom = np.concatenate((inter_M_t, bottom_right), axis=1)
    A = csr_matrix(np.concatenate((top, bottom), axis=0))
    return A


def get_ub_constraint_mat(params):
    train_matrix = np.load(f"./data/{params['dataset']}/train_matrix.npy", allow_pickle=True).tolist()

    # n_user = params['n_user']
    # num_neighbors = params['ub_neighbor_num']

    # num_neighbors = max(np.diff(train_matrix.indptr))
    # res_mat = torch.zeros((n_user, num_neighbors))
    # res_sim_mat = torch.zeros((n_user, num_neighbors))

    # res_mat_50 = torch.zeros((n_user, 50))
    # res_sim_mat_50 = torch.zeros((n_user, 50))
    #
    # res_mat_100 = torch.zeros((n_user, 100))
    # res_sim_mat_100 = torch.zeros((n_user, 100))
    #
    # res_mat_150 = torch.zeros((n_user, 150))
    # res_sim_mat_150 = torch.zeros((n_user, 150))
    #
    # res_mat_200 = torch.zeros((n_user, 200))
    # res_sim_mat_200 = torch.zeros((n_user, 200))

    items_D = np.sum(train_matrix, axis=0).reshape(-1)
    users_D = np.sum(train_matrix, axis=1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

    # constraint_mat = torch.from_numpy(beta_uD.dot(beta_iD))  # n_user * m_item

    constraint_mat = beta_uD.dot(beta_iD)
    # all_constraint_mat = torch.from_numpy(normalize(beta_uD.dot(beta_iD), axis=1, norm='max'))
    res_sim_mat = normalize(constraint_mat.getA() * train_matrix.toarray(), axis=1, norm='max')
    res_sim_mat = torch.from_numpy(res_sim_mat)

    # for i in tqdm(range(n_user)):
    #     row = normalize(constraint_mat[i].getA() * train_matrix.getrow(i).toarray()[0], axis=1, norm='max')
    #     row_sims, row_idxs = torch.topk(torch.from_numpy(row), num_neighbors)
    #     res_mat[i] = row_idxs
    #     res_sim_mat[i] = row_sims

    # row_sims, row_idxs = torch.topk(torch.from_numpy(row), 50)
    # res_mat_50[i] = row_idxs
    # res_sim_mat_50[i] = row_sims
    #
    # row_sims, row_idxs = torch.topk(torch.from_numpy(row), 100)
    # res_mat_100[i] = row_idxs
    # res_sim_mat_100[i] = row_sims
    #
    # row_sims, row_idxs = torch.topk(torch.from_numpy(row), 150)
    # res_mat_150[i] = row_idxs
    # res_sim_mat_150[i] = row_sims
    #
    # row_sims, row_idxs = torch.topk(torch.from_numpy(row), 200)
    # res_mat_200[i] = row_idxs
    # res_sim_mat_200[i] = row_sims
    all_constraint_mat = torch.from_numpy(normalize(constraint_mat, axis=1, norm='max'))
    return res_sim_mat.float(), all_constraint_mat


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


# def train_test_split(data, valid_ratio=0.8, test_ratio=0.2, shuffle=True, random_state=None):
#
#     if shuffle:
#         data = reset(data, random_state=random_state)
#
#     test = data[:int(len(data) * test_ratio)].reset_index(drop=True)
#     valid = data[int(len(data) * test_ratio):int(len(data) * (test_ratio+valid_ratio))].reset_index(drop=True)
#     train = data[int(len(data) * (test_ratio+valid_ratio)):].reset_index(drop=True)
#
#     train_data = np.array(train).tolist()
#     valid_data = np.array(valid).tolist()
#     test_data = np.array(test).tolist()
#
#     valid_users = set(i[0] for i in valid_data)
#     test_users = set(i[0] for i in test_data)
#     return train_data, valid_data, test_data, valid_users, test_users


# def load_data(args):
#     try:
#         print('Loading preprocessing files...')
#         UB = load_obj(f'./data/{args.dataname}/UB')
#         # UI = load_obj(f'./data/{args.dataname}/UI')
#         # BI = load_obj(f'./data/{args.dataname}/BI')
#
#         # UIB = UI * BI.T
#         # UIU = UI * UI.T
#         # BIB = BI * BI.T
#
#         # UIU = csr_matrix(UIU + np.eye(UIU.shape[0], dtype=int))
#         # BIB = csr_matrix(BIB + np.eye(BIB.shape[0], dtype=int))
#         #
#         # UIB.data = (UIB.data > 0).astype(int)
#         # UIU.data = (UIU.data > 0).astype(int)
#         # BIB.data = (BIB.data > 0).astype(int)
#
#         # top = np.concatenate((UIU.todense(), UIB.todense()), axis=1)
#         # bottom = np.concatenate((UIB.T.todense(), BIB.todense()), axis=1)
#         # graph = csr_matrix(np.concatenate((top, bottom), axis=0))
#
#         # beta_uiu = load_obj(f'./data/{args.dataname}/beta_uiu')
#         # beta_uib = load_obj(f'./data/{args.dataname}/beta_uib')
#         # beta_biu = load_obj(f'./data/{args.dataname}/beta_biu')
#         # beta_bib = load_obj(f'./data/{args.dataname}/beta_bib')
#
#         beta_uiu_pos = load_obj(f'./data/{args.dataname}/beta_uiu_pos')
#         beta_uib_pos = load_obj(f'./data/{args.dataname}/beta_uib_pos')
#         beta_biu_pos = load_obj(f'./data/{args.dataname}/beta_biu_pos')
#         beta_bib_pos = load_obj(f'./data/{args.dataname}/beta_bib_pos')
#
#         beta_uiu_neg = load_obj(f'./data/{args.dataname}/beta_uiu_neg')
#         beta_uib_neg = load_obj(f'./data/{args.dataname}/beta_uib_neg')
#         beta_biu_neg = load_obj(f'./data/{args.dataname}/beta_biu_neg')
#         beta_bib_neg = load_obj(f'./data/{args.dataname}/beta_bib_neg')
#     except:
#         print('load failed')
#         beta_uiu_pos, beta_uiu_neg, beta_uib_pos, beta_uib_neg, beta_biu_pos, beta_biu_neg, beta_bib_pos, beta_bib_neg = process_data(args)
#
#     # construct degree matrix for graphmf
#     args.n_user = UB.shape[0]
#     args.n_bundle = UB.shape[1]
#     args.n_node = args.n_user + args.n_bundle
#
#     valid_data, test_data, u_valid, u_test, beta_ub_pos, beta_ub_neg, valid_mask, test_mask, valid_ground_truth_list, test_ground_truth_list = train_test_split(args, UB)
#
#     if args.train_ratio > 0:
#         beta_uib_pos = beta_uib_pos + beta_ub_pos
#         beta_biu_pos = beta_biu_pos + beta_ub_pos.T
#         beta_uib_neg = beta_uib_neg + beta_ub_neg
#         beta_biu_neg = beta_biu_neg + beta_ub_neg.T
#
#     # beta_u = np.concatenate((beta_uiu.todense(), beta_uib.todense()), axis=1)
#     # beta_b = np.concatenate((beta_biu.todense(), beta_bib.todense()), axis=1)
#     # beta = np.concatenate((beta_u, beta_b), axis=0)
#
#     beta_u_pos = np.concatenate((beta_uiu_pos.todense(), beta_uib_pos.todense()), axis=1)
#     beta_b_pos = np.concatenate((beta_biu_pos.todense(), beta_bib_pos.todense()), axis=1)
#     beta_pos = csr_matrix(np.concatenate((beta_u_pos, beta_b_pos), axis=0))
#
#     beta_u_neg = np.concatenate((beta_uiu_neg.todense(), beta_uib_neg.todense()), axis=1)
#     beta_b_neg = np.concatenate((beta_biu_neg.todense(), beta_bib_neg.todense()), axis=1)
#     beta_neg = csr_matrix(np.concatenate((beta_u_neg, beta_b_neg), axis=0))
#
#     u_train = np.array(beta_pos.nonzero()[0], dtype='int64')
#     b_train = np.array(beta_pos.nonzero()[1], dtype='int64')
#
#     # np.array(list(set(u_test)), dtype='int64')
#
#     train_data = np.column_stack((u_train, b_train))
#
#     # interacted_items = [[] for _ in range(args.n_node)]
#
#     probs = (np.ones((args.n_node,args.n_node))-(beta_pos!=0).astype(int)).A
#     # for (u, i) in train_data:
#     #     probs[u][i]=0
#     probs /= np.sum(probs,axis=1).reshape(args.n_node,1)
#
#
#     return train_data, valid_data, test_data, u_valid, u_test, beta_pos, beta_neg, probs, valid_mask, test_mask, valid_ground_truth_list, test_ground_truth_list


def calculate_weight_v1(matrix_h, matrix_t, delta, dataset, metapath):
    n_head = matrix_h.shape[0]
    n_tail = matrix_t.shape[0]
    n_middle = matrix_h.shape[1]
    matrix_t2 = matrix_t + matrix_t
    # beta_h2t = csr_matrix((n_head, n_tail))
    alpha = lil_matrix((n_head, n_tail))
    beta = lil_matrix((n_head, n_tail))
    for i in tqdm(range(n_tail)):
        # for i in range(n_tail):
        re_t = matrix_t2[i].todense()
        neighbor = matrix_h + re_t.repeat(n_head, axis=0)
        t2_neighbor = (neighbor == 2).astype(int)
        co_neighbor = (neighbor == 3).astype(int)
        sum_t2_neighbor = np.sum(np.multiply(t2_neighbor, delta[:, i].reshape(1, n_middle).repeat(n_head, axis=0)),
                                 axis=1)
        sum_co_neighbor = np.sum(np.multiply(co_neighbor, delta[:, i].reshape(1, n_middle).repeat(n_head, axis=0)),
                                 axis=1)
        # beta_h2t[:, i] = sum_t2_neighbor - sum_co_neighbor
        alpha[:, i] = sum_co_neighbor
        beta[:, i] = sum_t2_neighbor

    # pstore(beta_h2t, file_name)
    pstore(alpha.tocsr(), f'./data/{dataset}/alpha_{metapath}')
    pstore(beta.tocsr(), f'./data/{dataset}/beta_{metapath}')

    return alpha, beta


def calculate_weight_v2(matrix_h, matrix_t, delta):
    start_time = time.time()

    matrix_h_ex = csr_matrix(np.ones(matrix_h.shape) - matrix_h)

    matrix_h = scipy_sparse_mat_to_torch_sparse_tensor(matrix_h)
    matrix_h_ex = scipy_sparse_mat_to_torch_sparse_tensor(matrix_h_ex)
    matrix_t_delta = scipy_sparse_mat_to_torch_sparse_tensor(matrix_t.multiply(delta))

    alpha = torch.sparse.mm(matrix_h, matrix_t_delta)
    beta = torch.sparse.mm(matrix_h_ex, matrix_t_delta)

    train_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
    print(train_time)


def calculate_weight(matrix_h, matrix_t, delta, metapath):
    n_head = matrix_h.shape[0]
    n_tail = matrix_t.shape[1]

    start_time = time.time()

    matrix_h_ex = np.ones(matrix_h.shape) - matrix_h

    matrix_h_tensor = scipy_sparse_mat_to_torch_sparse_tensor(matrix_h)
    matrix_h_ex = torch.FloatTensor(matrix_h_ex)
    matrix_t_delta = scipy_sparse_mat_to_torch_sparse_tensor(matrix_t.multiply(delta))
    matrix_t_delta_beta = torch.FloatTensor(matrix_t.multiply(delta).todense())
    alpha = torch.sparse.mm(matrix_h_tensor, matrix_t_delta)
    beta = torch.mm(matrix_h_ex, matrix_t_delta_beta)

    alpha = csr_matrix(((alpha.coalesce().values()), alpha.coalesce().indices()), shape=(n_head, n_tail))
    beta = csr_matrix(beta.numpy())

    train_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
    print(f'Calculate alpha_{metapath} and beta_{metapath}: {train_time}')
    return alpha, beta


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mtr):
    """
    将scipy的sparse matrix转换成torch的sparse tensor.
    """
    sparse_mx = sparse_mtr.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    del sparse_mx
    return torch.sparse.FloatTensor(indices, values, shape)


def calculate_delta(matrix):
    items_D = np.sum(matrix, axis=0).reshape(-1)
    users_D = np.sum(matrix, axis=1).reshape(-1)

    if not users_D.all() > 0:
        users_D += 1
    if not items_D.all() > 0:
        items_D += 1

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

    # beta_uD = (1 / np.sqrt(users_D + 1)).reshape(-1, 1)
    # beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

    beta = beta_uD.dot(beta_iD)
    # beta = torch.from_numpy(beta_uD.dot(beta_iD))
    return beta


def matrix_ht_set(matrix):
    h, t = matrix.nonzero()
    return set(h), set(t)


def train_test_split(args, UI_train, UI_valid, UI_test):
    u_train, i_train = UI_train.nonzero()
    u_valid, i_valid = UI_valid.nonzero()
    u_test, i_test = UI_test.nonzero()

    # list2txt('weeplaces_train.csv', u_train, i_train, delimiter=',')
    # list2txt('weeplaces_valid.csv', u_valid, i_valid, delimiter=',')
    # list2txt('weeplaces_test.csv', u_test, i_test, delimiter=',')

    train_data = np.column_stack((u_train, i_train))
    valid_data = np.column_stack((u_valid, i_valid))
    test_data = np.column_stack((u_test, i_test))

    u_valid = np.array(list(set(u_valid)), dtype='int64')
    u_test = np.array(list(set(u_test)), dtype='int64')

    valid_mask = torch.zeros(args.n_head, args.n_tail)

    for (u, i) in train_data:
        valid_mask[u][i] = -np.inf

    test_mask = valid_mask.clone()
    valid_ground_truth_list = [[] for _ in range(args.n_head)]
    for (u, i) in valid_data:
        valid_ground_truth_list[u].append(i)
        test_mask[u][i] = -np.inf

    test_ground_truth_list = [[] for _ in range(args.n_head)]
    for (u, i) in test_data:
        test_ground_truth_list[u].append(i)

    return u_valid, u_test, valid_mask, test_mask, valid_ground_truth_list, test_ground_truth_list


'''
Useful functions
'''


def pload(path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    print('load path = {} object'.format(path))
    return res


def list2txt(filename, u, i, v=None, delimiter='\t'):
    with open(f'{filename}', 'w') as f:
        writer = csv.writer(f, delimiter=delimiter)
        if type(v) == type(None):
            writer.writerows(zip(u, i))
        else:
            writer.writerows(zip(u, i, v))


def pstore(x, path):
    with open(path, 'wb') as f:
        pickle.dump(x, f)
    print('store object in path = {} ok'.format(path))


class RecData(Dataset):

    def __init__(self, args, u_train, b_train):
        # neg_candidates = np.arange(args.n_node)
        # data_size = u_train.shape[0]
        # neg_items = np.tile(np.random.choice(neg_candidates, (7363, args.n_negative), replace=True), (2887, 1))#data_size//args.n_node
        # tail = np.random.choice(neg_candidates, (data_size % args.n_node, args.n_negative), replace=True)
        # neg_items = np.r_[neg_items,tail]
        self.users = u_train
        self.item = b_train
        self.device = args.device
        self.neg_candidates = np.arange(args.n_node)
        self.n_negative = args.n_negative

    def __getitem__(self, idx):
        assert idx < len(self.users)
        # neg_items = np.random.choice(self.neg_candidates, self.n_negative, replace=True)
        # neg_items = torch.tensor(neg_items, device=self.device)
        return self.users[idx], self.item[idx]  # , neg_items

    def __len__(self):
        return len(self.users)


class dp():  # dataloader_param
    n_node = None
    device = None
    n_negative = None
    neg_candidates = None

    def __init__(self, device, n_negative, n_node):
        dp.device = device
        dp.n_negative = n_negative
        dp.neg_candidates = np.arange(n_node)


# def Sampling(x, args.n_bundle, args.n_negative, interacted_items):
#
#     n_bundle = params['n_bundle']
#     neg_ratio = params['negative_num']
#
#
#     neg_candidates = np.arange(n_bundle)
#
#     # if params['sampling_sift_pos']:
#     neg_items = []
#     neg_items_w = []
#     for u in batch_users:
#         probs = np.ones(n_bundle)
#         probs[interacted_items[u]] = 0
#         probs /= np.sum(probs)
#
#         u_neg_items = np.random.choice(neg_candidates, size=neg_ratio, p=probs, replace=True).reshape(1, -1)
#
#         neg_items.append(u_neg_items)
#         neg_items_w.append(1-all_constraint_mat[u][u_neg_items])
#
#     neg_items = np.concatenate(neg_items, axis=0)
#     neg_items_w = torch.stack(neg_items_w)
#     # else:
#     #     neg_items = np.random.choice(neg_candidates, (len(batch_users[0]), neg_ratio), replace=True)
#
#     neg_items = torch.from_numpy(neg_items)
#
#     return neg_items, neg_items_w  # users, pos_items, neg_items

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')
