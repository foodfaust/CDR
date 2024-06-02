import argparse
def parse_args_mafengwo():
    parser = argparse.ArgumentParser(description='ECTR')
    parser.add_argument('--dataname', type=str, default='mafengwo', help='Name of dataset.')
    parser.add_argument("--embedding_dim", type=int, default=64, help='Hidden layer dim.')
    parser.add_argument("--batch_size", type=dict, default={'member': 512, 'tuple': 96}, help='batch_size')#96
    parser.add_argument("--seed", type=int, default=2022, help='seed')
    parser.add_argument('--split_ratio', type=dict, default={'train': 0.05, 'test': 0.2}, help='split_ratio')
    parser.add_argument("--require_data", type=list, default=['member','tuple'], help='member、tuple')
    parser.add_argument('--τ', type=dict, default={'member': 3.8, 'tuple': 1}, help='item_temp')
    parser.add_argument('--top_k', type=list, default=[10, 20, 30], help='top_k')
    parser.add_argument('--patience', type=int, default=10, help='Patient epochs to wait before early stopping.')
    parser.add_argument('--γ', type=float, default=1e-4, help='gamma')
    return parser.parse_args()


def parse_args_youshu():
    parser = argparse.ArgumentParser(description='ECTR')
    parser.add_argument('--dataname', type=str, default='youshu', help='Name of dataset.')
    parser.add_argument("--embedding_dim", type=int, default=64, help='Hidden layer dim.')
    parser.add_argument("--batch_size", type=dict, default={'member': 512, 'tuple': 128}, help='batch_size')
    parser.add_argument("--seed", type=int, default=2022, help='seed')
    parser.add_argument('--split_ratio', type=dict, default={'train': 0.05, 'test': 0.2}, help='split_ratio')
    parser.add_argument("--require_data", type=list, default=['member', 'tuple'], help='member、tuple')
    parser.add_argument('--τ', type=dict, default={'member': 1, 'tuple': 0.3}, help='item_temp')
    parser.add_argument('--top_k', type=list, default=[10, 20, 30], help='top_k')
    parser.add_argument('--patience', type=int, default=10, help='Patient epochs to wait before early stopping.')
    parser.add_argument('--γ', type=float, default=1e-4, help='gamma')
    return parser.parse_args()

def parse_args_lastfm():
    print('-----------------------------------------------')
    parser = argparse.ArgumentParser(description='my_model')
    parser.add_argument('--dataname', type=str, default='lastfm', help='Name of dataset.')
    parser.add_argument("--embedding_dim", type=int, default=64, help='Hidden layer dim.')
    parser.add_argument("--batch_size", type=dict, default={'item': 32, 'social': 32}, help='batch_size')  # 4096
    parser.add_argument("--test_batch_size", type=int, default=2048, help='test_batch_size')
    parser.add_argument("--max_epoch", type=int, default=10000, help='max_epoch')
    parser.add_argument("--seed", type=int, default=2022, help='seed')
    parser.add_argument('--split_ratio', type=dict, default={'train': 0.05, 'test': 0.2}, help='split_ratio')
    parser.add_argument('--num_workers', type=dict, default={'item': 0, 'social': 0}, help='num_workers')
    parser.add_argument("--require_data", type=list, default=['item'], help='social、item')  # social、item
    parser.add_argument('--τ', type=dict, default={'item': 3, 'social': 1.2}, help='item_temp')

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate of mvgrl.')
    parser.add_argument('--wd', type=float, default=0., help='Weight decay of mvgrl.')
    parser.add_argument('--top_k', type=list, default=[10,20,30], help='top_k')
    parser.add_argument('--patience', type=int, default=20, help='Patient epochs to wait before early stopping.')

    parser.add_argument('--γ', type=float, default=1e-4, help='gamma')
    parser.add_argument('--pos_weight', type=float, default=0.5, help='pos_weight')
    parser.add_argument('--neg_weight', type=float, default=0.5, help='neg_weight')
    parser.add_argument('--initial_weight', type=float, default=1e-4, help='initial_weight')

    return parser.parse_args()