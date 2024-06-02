import pandas as pd

from util import beta_cat, calculate_delta, calculate_weight, train_test_split
from scipy.sparse import csr_matrix
import numpy as np


def data_param_prepare(args):
    data = pd.read_csv(f'./data/{args.dataname}/head_tail.csv', sep=',', header=None,
                       names=['head_id', 'tail_id'])
    data['head_id'] = data['head_id'].astype('category')
    data['tail_id'] = data['tail_id'].astype('category')
    head_id = data['head_id'].cat.codes.values
    tail_id = data['tail_id'].cat.codes.values
    head_num = head_id.max() + 1
    tail_num = tail_id.max() + 1

    args.n_head, args.n_tail = head_num, tail_num
    args.n_node = args.n_head + args.n_tail

    link = np.ones(len(data['head_id']))

    tuple_train = csr_matrix((link, (head_id, tail_id)), shape=(head_num, tail_num))
    tuple_valid = csr_matrix((link, (head_id, tail_id)), shape=(head_num, tail_num))
    tuple_test = csr_matrix((link, (head_id, tail_id)), shape=(head_num, tail_num))

    float_mask = np.random.permutation(np.linspace(0, 1, len(data['head_id'])))
    tuple_train.data[float_mask >= args.split_ratio['train']] = 0
    tuple_valid.data[(float_mask < args.split_ratio['train']) | (float_mask > (1 - args.split_ratio['test']))] = 0
    tuple_test.data[float_mask <= (1 - args.split_ratio['test'])] = 0

    link = np.ones(len(tuple_train.nonzero()[0]))
    tuple_train = csr_matrix((link, tuple_train.nonzero()), shape=(head_num, tail_num))

    link = np.ones(len(tuple_valid.nonzero()[0]))
    tuple_valid = csr_matrix((link, tuple_valid.nonzero()), shape=(head_num, tail_num))

    link = np.ones(len(tuple_test.nonzero()[0]))
    tuple_test = csr_matrix((link, tuple_test.nonzero()), shape=(head_num, tail_num))

    if 'tuple' in args.require_data:
        alpha_hh_tuple, beta_hh_tuple, alpha_ht_tuple, beta_ht_tuple, alpha_th_tuple, beta_th_tuple, alpha_tt_tuple, beta_tt_tuple = process_tuple(
            tuple_train)
    else:
        alpha_hh_tuple, beta_hh_tuple = csr_matrix((args.n_head, args.n_head)), csr_matrix((args.n_head, args.n_head))
        alpha_ht_tuple, beta_ht_tuple = csr_matrix((args.n_head, args.n_tail)), csr_matrix((args.n_head, args.n_tail))
        alpha_th_tuple, beta_th_tuple = csr_matrix((args.n_tail, args.n_head)), csr_matrix((args.n_tail, args.n_head))
        alpha_tt_tuple, beta_tt_tuple = csr_matrix((args.n_tail, args.n_tail)), csr_matrix((args.n_tail, args.n_tail))

    if 'member' in args.require_data:
        alpha_hh_member, beta_hh_member, alpha_ht_member, beta_ht_member, alpha_th_member, beta_th_member, alpha_tt_member, beta_tt_member = process_member(
            args.dataname,
            head_num,
            tail_num)
    else:
        alpha_hh_member, beta_hh_member = csr_matrix((args.n_head, args.n_head)), csr_matrix((args.n_head, args.n_head))
        alpha_ht_member, beta_ht_member = csr_matrix((args.n_head, args.n_tail)), csr_matrix((args.n_head, args.n_tail))
        alpha_th_member, beta_th_member = csr_matrix((args.n_tail, args.n_head)), csr_matrix((args.n_tail, args.n_head))
        alpha_tt_member, beta_tt_member = csr_matrix((args.n_tail, args.n_tail)), csr_matrix((args.n_tail, args.n_tail))

    valid, test, valid_mask, test_mask, valid_ground_truth_list, test_ground_truth_list = train_test_split(args,
                                                                                                           tuple_train,
                                                                                                           tuple_valid,
                                                                                                           tuple_test)

    alpha_tuple = beta_cat(alpha_hh_tuple, alpha_ht_tuple, alpha_th_tuple, alpha_tt_tuple)
    beta_tuple = beta_cat(beta_hh_tuple, beta_ht_tuple, beta_th_tuple, beta_tt_tuple)

    alpha_member = beta_cat(alpha_hh_member, alpha_ht_member, alpha_th_member, alpha_tt_member)
    beta_member = beta_cat(beta_hh_member, beta_ht_member, beta_th_member, beta_tt_member)

    head_train = {'tuple': np.array(alpha_tuple.nonzero()[0], dtype='int64'),
                  'member': np.array(alpha_member.nonzero()[0], dtype='int64')}
    tail_train = {'tuple': np.array(alpha_tuple.nonzero()[1], dtype='int64'),
                  'member': np.array(alpha_member.nonzero()[1], dtype='int64')}

    alpha = {'tuple': alpha_tuple, 'member': alpha_member}
    beta = {'tuple': beta_tuple, 'member': beta_member}

    return head_train, tail_train, valid, test, alpha, beta, valid_mask, test_mask, valid_ground_truth_list, test_ground_truth_list


def process_tuple(GI_train):
    delta_gi, delta_ig = calculate_delta(GI_train), calculate_delta(GI_train.T)
    alpha_gig, beta_gig = calculate_weight(GI_train, GI_train.T, delta_ig, 'gig')
    alpha_igi, beta_igi = calculate_weight(GI_train.T, GI_train, delta_gi, 'igi')
    alpha_gi = csr_matrix(np.multiply(delta_gi, GI_train.todense()))
    alpha_ig = csr_matrix(np.multiply(delta_ig, GI_train.T.todense()))

    beta_gi, beta_ig = csr_matrix(delta_gi), csr_matrix(delta_ig)

    return alpha_gig, beta_gig, alpha_gi, beta_gi, alpha_ig, beta_ig, alpha_igi, beta_igi


def process_member(dataset, group_num, item_num):
    gu_data = pd.read_csv(f'./data/{dataset}/head_member.csv', sep=',', header=None,
                          names=['group_id', 'user_id'])

    ui_data = pd.read_csv(f'./data/{dataset}/member_tail.csv', sep=',', header=None,
                          names=['user_id', 'item_id'])

    user_num = max(max(gu_data['user_id']), max(ui_data['user_id'])) + 1

    link = np.ones(len(gu_data['user_id']))
    GU = csr_matrix((link, (gu_data['group_id'], gu_data['user_id'])), shape=(group_num, user_num))

    link = np.ones(len(ui_data['item_id']))
    UI = csr_matrix((link, (ui_data['user_id'], ui_data['item_id'])), shape=(user_num, item_num))

    delta_ug, delta_ui = calculate_delta(GU.T), calculate_delta(UI)
    alpha_gug, beta_gug = calculate_weight(GU, GU.T, delta_ug, 'gug')
    alpha_gui, beta_gui = calculate_weight(GU, UI, delta_ui, 'gui')
    alpha_iug, beta_iug = calculate_weight(UI.T, GU.T, delta_ug, 'iug')
    alpha_iui, beta_iui = calculate_weight(UI.T, UI, delta_ui, 'iui')
    return alpha_gug, beta_gug, alpha_gui, beta_gui, alpha_iug, beta_iug, alpha_iui, beta_iui
