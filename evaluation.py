from IPython import embed
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from collections import Counter
import torch.utils.data as data
import scipy.sparse as sp
import os
import gc
import configparser
import time
import argparse
from torch.utils.tensorboard import SummaryWriter

def hit(gt_item, pred_items):
    if gt_item in pred_items:
        return 1
    return 0


def ndcg(gt_item, pred_items):
    if gt_item in pred_items:
        index = pred_items.index(gt_item)
        return np.reciprocal(np.log2(index + 2))
    return 0


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k

    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall_n = np.where(recall_n != 0, recall_n, 1)
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1. / np.arange(1, k + 1))
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)


def test_one_batch(sorted_items,groundTrue, k):
    # sorted_items = X[0].numpy()
    # groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    ret = RecallPrecision_ATk(groundTrue, r, k)

    return ret['precision'], ret['recall'], NDCGatK_r(groundTrue, r, k)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def get_entropy(item_matrix):
    """Get shannon entropy through the top-k recommendation list.

    Args:
        item_matrix(numpy.ndarray): matrix of items recommended to users.

    Returns:
        float: the shannon entropy.
    """

    item_count = dict(Counter(item_matrix.flatten()))
    total_num = item_matrix.shape[0] * item_matrix.shape[1]
    result = 0.0
    for cnt in item_count.values():
        p = cnt / total_num
        result += -p * np.log(p)
    return result #/ len(item_count)

def get_coverage(item_matrix, num_items):
    """Get the coverage of recommended items over all items

    Args:
        item_matrix(numpy.ndarray): matrix of items recommended to users.
        num_items(int): the total number of items.

    Returns:
        float: the `coverage` metric.
    """
    unique_count = np.unique(item_matrix).shape[0]
    return unique_count / num_items

def get_gini(item_matrix, num_items):
    """Get gini index through the top-k recommendation list.

    Args:
        item_matrix(numpy.ndarray): matrix of items recommended to users.
        num_items(int): the total number of items.

    Returns:
        float: the gini index.
    """
    item_count = dict(Counter(item_matrix.flatten()))
    sorted_count = np.array(sorted(item_count.values()))
    num_recommended_items = sorted_count.shape[0]
    total_num = item_matrix.shape[0] * item_matrix.shape[1]
    idx = np.arange(num_items - num_recommended_items + 1, num_items + 1)
    gini_index = np.sum((2 * idx - num_items - 1) * sorted_count) / total_num
    gini_index /= num_items
    return gini_index