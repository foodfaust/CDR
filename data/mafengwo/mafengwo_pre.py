from util import list2txt
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
import pickle

def pstore(x, path):
    with open(path, 'wb') as f:
        pickle.dump(x, f)
    print('store object in path = {} ok'.format(path))

gu_file = open(f'groupMember.txt')
gu_data = gu_file.readlines()
gu_file.close()
groups, users = [], []
for line in gu_data:
    g, us = line.strip().split(' ')
    for u in us.split(','):
        groups.append(g)
        users.append(u)
list2txt(f'head_member.csv', groups, users, v=None, delimiter=',')

gi_train_file = open(f'groupRatingTrain.txt')
gi_train_data = gi_train_file.readlines()
gi_train_file.close()
groups, items = [], []
for line in gi_train_data:
    g, i = line.strip().split(' ')
    groups.append(g)
    items.append(i)

gi_test_file = open(f'groupRatingTest.txt')
gi_test_data = gi_test_file.readlines()
gi_test_file.close()
for line in gi_test_data:
    g, i = line.strip().split(' ')
    groups.append(g)
    items.append(i)
list2txt(f'head_tail.csv', groups, items, v=None, delimiter=',')

ui_train_file = open(f'userRatingTrain.txt')
ui_train_data = ui_train_file.readlines()
ui_train_file.close()
users, items = [], []
for line in ui_train_data:
    u, i = line.strip().split(' ')
    users.append(u)
    items.append(i)

ui_test_file = open(f'userRatingTest.txt')
ui_test_data = ui_test_file.readlines()
ui_test_file.close()
for line in ui_test_data:
    u, i = line.strip().split(' ')
    users.append(u)
    items.append(i)
list2txt(f'member_tail.csv', users, items, v=None, delimiter=',')

gi_data = pd.read_csv(f'head_tail.csv', sep=',', header=None,
                      names=['group_id', 'item_id'])

gu_data = pd.read_csv(f'head_member.csv', sep=',', header=None,
                      names=['group_id', 'user_id'])

ui_data = pd.read_csv(f'member_tail.csv', sep=',', header=None,
                      names=['user_id', 'item_id'])

g_num = max(gi_data['group_id'].max(), gu_data['group_id'].max()) + 1
u_num = int(max(gu_data['user_id'].max(), ui_data['user_id'].max())) + 1
i_num = max(gi_data['item_id'].max(), ui_data['item_id'].max()) + 1

link = np.ones(len(gi_data['group_id']))
GI = csr_matrix((link, (gi_data['group_id'], gi_data['item_id'])), shape=(g_num, i_num))

link = np.ones(len(gu_data['group_id']))
GU = csr_matrix((link, (gu_data['group_id'], gu_data['user_id'])), shape=(g_num, u_num))

link = np.ones(len(ui_data['user_id']))
UI = csr_matrix((link, (ui_data['user_id'], ui_data['item_id'])), shape=(u_num, i_num))

gi_g, gi_i = GI.nonzero()
gi_g, gi_i = set(gi_g), set(gi_i)

gi_i = list(gi_i)
gi_i.sort()
GI = GI[:, gi_i]
UI = UI[:, gi_i]

gu_g, gu_u = GU.nonzero()

ui_u, ui_i = UI.nonzero()

gi_g, gi_i = GI.nonzero()


list2txt(f'head_member.csv', gu_g, gu_u, v=None, delimiter=',')
list2txt(f'member_tail.csv', ui_u, ui_i, v=None, delimiter=',')
list2txt(f'head_tail.csv', gi_g, gi_i, v=None, delimiter=',')

