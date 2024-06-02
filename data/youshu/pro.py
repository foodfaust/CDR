import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip
# from utils import filter_Kcore
from scipy.sparse import csr_matrix
import pandas as pd
import scipy.sparse as sp
import csv
import numpy
import torch
from collections import defaultdict

from numpy import array as matrix, arange
import pickle


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


np.random.seed(2021)


def list2txt(filename, u, i, v=None):
    with open(f'{filename}.txt', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        if v == None:
            writer.writerows(zip(u, i))
        else:
            writer.writerows(zip(u, i, v))


# K-core user_core item_core
def check_Kcore(user_items, user_core, item_core):
    user_count = defaultdict(int)
    item_count = defaultdict(int)
    for user, items in user_items.items():
        for item in items:
            user_count[user] += 1
            item_count[item] += 1

    for user, num in user_count.items():
        if num < user_core:
            return user_count, item_count, False
    for item, num in item_count.items():
        if num < item_core:
            return user_count, item_count, False
    return user_count, item_count, True  # 已经保证Kcore


# 循环过滤 K-core
def filter_Kcore(user_items, user_core, item_core):  # user 接所有items
    user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    while not isKcore:
        for user, num in user_count.items():
            if user_count[user] < user_core:  # 直接把user 删除
                user_items.pop(user)
            else:
                for item in user_items[user]:
                    if item_count[item] < item_core:
                        user_items[user].remove(item)
        user_count, item_count, isKcore = check_Kcore(user_items, user_core, item_core)
    return user_items, user_count, item_count


def pro_file(data_file, k_sqe=None, k_sqe2=None):
    kv_only = {}
    if isinstance(data_file, str):
        lines = open(data_file).readlines()
        for line in tqdm(lines[0:]):
            key, value = line.strip().split('\t')
            tag = True
            if k_sqe != None:
                tag = key in k_sqe
            if k_sqe2 != None:
                tag = tag and (value in k_sqe2)
            if tag:
                if key in kv_only:
                    if value not in kv_only[key]:
                        kv_only[key].append(value)
                    else:
                        continue
                else:
                    kv_only[key] = []
                    kv_only[key].append(value)
    else:
        lines = data_file
        for line in tqdm(lines[0:]):
            key, value = line
            tag = True
            if k_sqe != None:
                tag = key in k_sqe
            if k_sqe2 != None:
                tag = tag and (value in k_sqe2)
            if tag:
                if key in kv_only:
                    if value not in kv_only[key]:
                        kv_only[key].append(value)
                    else:
                        continue
                else:
                    kv_only[key] = []
                    kv_only[key].append(value)
    kv_datas_only = []
    for key in kv_only.keys():
        for g in kv_only[key]:
            kv_datas_only.append([key, g])
    return kv_datas_only


def data2dict(input):
    if isinstance(input, str):
        datas = []
        lines = open(input).readlines()
        for line in tqdm.tqdm(lines[0:]):
            row1, row2 = line.strip().split('\t')
            datas.append((row1, row2))
    else:
        datas = input
    dic = {}
    for data in datas:
        key, value = data
        if key in dic:
            if value not in dic[key]:
                dic[key].append(value)
            else:
                continue
        else:
            dic[key] = []
            dic[key].append(value)
    return dic


def dict2list(dic):
    datas = []
    for key in dic.keys():
        for value in dic[key]:
            datas.append([key, value])
    return datas


def pretrain_data_pro(u_core, b_core):
    # u_core = pretrain_kcore[0]
    # # i_core = pretrain_kcore[1]
    # b_core = pretrain_kcore[1]

    UI = load_obj('user_item')
    UB = load_obj('user_list')
    BI = load_obj('list_item')

    UIB = UI * BI.T

    u = UIB.nonzero()[0]
    b = UIB.nonzero()[1]  # + n_user
    uib = [list(t) for t in zip(u, b)]

    # u = UI.nonzero()[0]
    # i = UI.nonzero()[1]  # + n_user
    # ui = [list(t) for t in zip(u, i)]

    u = UB.nonzero()[0]
    b = UB.nonzero()[1]  # + n_user
    ub = [list(t) for t in zip(u, b)]

    # b = BI.nonzero()[0]
    # i = BI.nonzero()[1]  # + n_user
    # bi = [list(t) for t in zip(b, i)]

    ub_dict = data2dict(ub)
    ub_dict_new, u_count, b_count = filter_Kcore(ub_dict, user_core=1, item_core=1)
    uib_new = pro_file(uib, k_sqe=u_count, k_sqe2=b_count)

    uib_dict = data2dict(uib_new)
    uib1, u_count, b_count = filter_Kcore(uib_dict, user_core=u_core, item_core=b_core)
    ub_datas1 = pro_file(ub, k_sqe=u_count, k_sqe2=b_count)
    ub_dict1 = data2dict(ub_datas1)
    ub_dict_new1, u_count, b_count = filter_Kcore(ub_dict1, user_core=1, item_core=1)
    uib_data = pro_file(uib, k_sqe=u_count, k_sqe2=b_count)

    # uib_dict1 = data2dict(uib_data)
    # uib2, u_count, b_count = filter_Kcore(uib_dict1, user_core=u_core, item_core=b_core)

    uib_data = pd.DataFrame(uib_data, columns=['user_id', 'bundle_id'])
    uib_data['user_id'] = uib_data['user_id'].astype('category')
    uib_data['bundle_id'] = uib_data['bundle_id'].astype('category')
    uib_uid = uib_data['user_id'].cat.codes.values
    uib_bid = uib_data['bundle_id'].cat.codes.values
    user_num = uib_uid.max() + 1
    bundle_num = uib_bid.max() + 1

    uid_dict_new2old = dict(enumerate(uib_data['user_id'].cat.categories))
    bid_dict_new2old = dict(enumerate(uib_data['bundle_id'].cat.categories))

    train_mat = sp.dok_matrix((user_num, bundle_num), dtype=np.float32)
    for i, j in tqdm(list(zip(uib_uid, uib_bid))):
        train_mat[i, j] = UIB[uid_dict_new2old[i], bid_dict_new2old[j]]
    # uib_datas = dict2list(uib2)
    # bi_datas = pro_file(bi, k_sqe=b_count, k_sqe2=None)
    ub_datas = pro_file(ub, k_sqe=u_count, k_sqe2=b_count)
    # ui_datas = pro_file(ui, k_sqe=u_count, k_sqe2=None)

    ub_data = pd.DataFrame(ub_datas, columns=['user_id', 'bundle_id'])
    ub_data['user_id'] = ub_data['user_id'].astype('category')
    ub_data['bundle_id'] = ub_data['bundle_id'].astype('category')
    ub_uid = ub_data['user_id'].cat.codes.values
    ub_bid = ub_data['bundle_id'].cat.codes.values

    # v = np.ones(len(uib_uid), dtype=int)
    list2txt(f'uib_{u_core}_{b_core}', uib_uid, uib_bid)
    list2txt(f'ub_{u_core}_{b_core}', ub_uid, ub_bid)
    np.save(f"train_matrix.npy", train_mat.tocsr())
    # np.savetxt(f'uib_{u_core}_{b_core}.txt', uib_data, fmt="%s", delimiter='\t')
    # np.savetxt(f'ui_{u_core}_{b_core}.txt', ui_datas, fmt="%s", delimiter='\t')
    # np.savetxt(f'ub_{u_core}_{b_core}.txt', ub_datas, fmt="%s", delimiter='\t')
    # np.savetxt(f'bi_{u_core}_{b_core}.txt', bi_datas, fmt="%s", delimiter='\t')
    print('over!!!')


def data_pro(u_core, b_core):
    ui_1 = arange(6).reshape(2, 3)
    bi_1 = arange(12).reshape(3, 4).T

    ui_2 = arange(6).reshape(2, 3)
    bi_2 = arange(12).reshape(3, 4).T

    beta_iu = arange(6).reshape(3,2)
    beta_ib = arange(12).reshape(3, 4)

    beta_uiu = np.empty((ui_1.shape[0],ui_2.shape[0]))
    beta_uib = np.empty((ui_1.shape[0],bi_2.shape[0]))
    beta_biu = np.empty((bi_1.shape[0],ui_2.shape[0]))
    beta_bib = np.empty((bi_1.shape[0], bi_2.shape[0]))

    for i in range(bi_2.shape[0]):
        for j in range(ui_1.shape[0]):
            neighbor = ui_1[j] + bi_2[i]
            u_neighbor = (neighbor == 1).astype(int)
            b_neighbor = (neighbor == 2).astype(int)
            co_neighbor = (neighbor == 3).astype(int)
            beta_uib[j,i] = sum(b_neighbor * beta_ib[:,i]) - sum(co_neighbor * beta_ib[:,i])
            beta_biu[i,j] = sum(u_neighbor * beta_iu[j]) - sum(co_neighbor * beta_ib[j])

    for i in range(ui_2.shape[0]):
        for j in range(ui_1.shape[0]):
            neighbor = ui_1[j] + ui_2[i]
            u2_neighbor = (neighbor == 2).astype(int)
            co_neighbor = (neighbor == 3).astype(int)
            beta_uiu[j, i] = sum(u2_neighbor * beta_iu[:, i]) - sum(co_neighbor * beta_iu[:, i])

    for i in range(bi_2.shape[0]):
        for j in range(bi_1.shape[0]):
            neighbor = bi_1[j] + bi_2[i]
            b2_neighbor = (neighbor == 2).astype(int)
            co_neighbor = (neighbor == 3).astype(int)
            beta_bib[j, i] = sum(b2_neighbor * beta_ib[:, i]) - sum(co_neighbor * beta_ib[:, i])



    ui_re_b = ui[:,:,np.newaxis].repeat(ib.shape[1], axis=2)
    ib_re_u = ib[np.newaxis, :].repeat(ui.shape[0], axis=0)

    neighbor_matrix = ui_re_b + ib_re_u

    u_neighbor = (neighbor_matrix == 1).astype(int)
    b_neighbor = (neighbor_matrix == 2).astype(int)
    co_neighbor = (neighbor_matrix == 3).astype(int)

    b = matrix([2, 2])









    # u_core = pretrain_kcore[0]
    # # i_core = pretrain_kcore[1]
    # b_core = pretrain_kcore[1]

    UI = load_obj('user_item')
    UB = load_obj('user_list')
    BI = load_obj('list_item')

    ui_i = UI.T.nonzero()[0]
    bi_i = BI.T.nonzero()[0]  # + n_user

    set_ui_i = set(ui_i)
    set_bi_i = set(bi_i)

    items = list(set_ui_i & set_bi_i)
    items.sort()

    UI = UI[:, items]
    BI = BI[:, items]

    ui_u = UI.nonzero()[0]
    bi_b = BI.nonzero()[0]  # + n_user

    users = list(set(ui_u))
    bundles = list(set(bi_b))
    users.sort()
    bundles.sort()

    UI = UI[users, :]
    BI = BI[bundles, :]

    UB = UB[users,:]
    UB = UB[:,bundles]

    uib = [list(t) for t in zip(u, b)]

    # u = UI.nonzero()[0]
    # i = UI.nonzero()[1]  # + n_user
    # ui = [list(t) for t in zip(u, i)]

    u = UB.nonzero()[0]
    b = UB.nonzero()[1]  # + n_user
    ub = [list(t) for t in zip(u, b)]

    # b = BI.nonzero()[0]
    # i = BI.nonzero()[1]  # + n_user
    # bi = [list(t) for t in zip(b, i)]

    ub_dict = data2dict(ub)
    ub_dict_new, u_count, b_count = filter_Kcore(ub_dict, user_core=1, item_core=1)
    uib_new = pro_file(uib, k_sqe=u_count, k_sqe2=b_count)

    uib_dict = data2dict(uib_new)
    uib1, u_count, b_count = filter_Kcore(uib_dict, user_core=u_core, item_core=b_core)
    ub_datas1 = pro_file(ub, k_sqe=u_count, k_sqe2=b_count)
    ub_dict1 = data2dict(ub_datas1)
    ub_dict_new1, u_count, b_count = filter_Kcore(ub_dict1, user_core=1, item_core=1)
    uib_data = pro_file(uib, k_sqe=u_count, k_sqe2=b_count)

    # uib_dict1 = data2dict(uib_data)
    # uib2, u_count, b_count = filter_Kcore(uib_dict1, user_core=u_core, item_core=b_core)

    uib_data = pd.DataFrame(uib_data, columns=['user_id', 'bundle_id'])
    uib_data['user_id'] = uib_data['user_id'].astype('category')
    uib_data['bundle_id'] = uib_data['bundle_id'].astype('category')
    uib_uid = uib_data['user_id'].cat.codes.values
    uib_bid = uib_data['bundle_id'].cat.codes.values
    user_num = uib_uid.max() + 1
    bundle_num = uib_bid.max() + 1

    uid_dict_new2old = dict(enumerate(uib_data['user_id'].cat.categories))
    bid_dict_new2old = dict(enumerate(uib_data['bundle_id'].cat.categories))

    train_mat = sp.dok_matrix((user_num, bundle_num), dtype=np.float32)
    for i, j in tqdm(list(zip(uib_uid, uib_bid))):
        train_mat[i, j] = UIB[uid_dict_new2old[i], bid_dict_new2old[j]]
    # uib_datas = dict2list(uib2)
    # bi_datas = pro_file(bi, k_sqe=b_count, k_sqe2=None)
    ub_datas = pro_file(ub, k_sqe=u_count, k_sqe2=b_count)
    # ui_datas = pro_file(ui, k_sqe=u_count, k_sqe2=None)

    ub_data = pd.DataFrame(ub_datas, columns=['user_id', 'bundle_id'])
    ub_data['user_id'] = ub_data['user_id'].astype('category')
    ub_data['bundle_id'] = ub_data['bundle_id'].astype('category')
    ub_uid = ub_data['user_id'].cat.codes.values
    ub_bid = ub_data['bundle_id'].cat.codes.values

    # v = np.ones(len(uib_uid), dtype=int)
    list2txt(f'uib_{u_core}_{b_core}', uib_uid, uib_bid)
    list2txt(f'ub_{u_core}_{b_core}', ub_uid, ub_bid)
    np.save(f"train_matrix.npy", train_mat.tocsr())
    # np.savetxt(f'uib_{u_core}_{b_core}.txt', uib_data, fmt="%s", delimiter='\t')
    # np.savetxt(f'ui_{u_core}_{b_core}.txt', ui_datas, fmt="%s", delimiter='\t')
    # np.savetxt(f'ub_{u_core}_{b_core}.txt', ub_datas, fmt="%s", delimiter='\t')
    # np.savetxt(f'bi_{u_core}_{b_core}.txt', bi_datas, fmt="%s", delimiter='\t')
    print('over!!!')


# if __name__ == '__main__':
# data_pro(1, 1)
