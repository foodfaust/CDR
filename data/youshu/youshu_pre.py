from util import list2txt
import pickle

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def pstore(x, path):
    with open(path, 'wb') as f:
        pickle.dump(x, f)
    print('store object in path = {} ok'.format(path))


UI = load_obj(f'user_item')
UB = load_obj(f'user_list')
BI = load_obj(f'list_item')

ub_u, ub_b = UB.nonzero()
ub_u, ub_b = list(set(ub_u)), list(set(ub_b))
ub_u.sort()
ub_b.sort()

UB = UB[ub_u, :]
UB = UB[:, ub_b]
UI = UI[ub_u, :]
BI = BI[ub_b, :]

ub_u, ub_b = UB.nonzero()

ui_u, ui_i = UI.nonzero()

bi_b, bi_i = BI.nonzero()

list2txt(f'head_member.csv', ui_u, ui_i, v=None, delimiter=',')
list2txt(f'member_tail.csv', bi_i, bi_b, v=None, delimiter=',')
list2txt(f'head_tail.csv', ub_u, ub_b, v=None, delimiter=',')
