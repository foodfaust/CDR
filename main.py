from tqdm import tqdm
import random
from config import *
from data_processing import data_param_prepare
from evaluation import *
from model import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def pstore(x, path):
    with open(path, 'wb') as f:
        pickle.dump(x, f)
    print('store object in path = {} ok'.format(path))

def pload(path):
    with open(path, 'rb') as f:
        res = pickle.load(f)
    print('load path = {} object'.format(path))
    return res

def collate(batch):
    batch = torch.stack(batch, dim=1)
    return batch[0], batch[1]

def train(args, model, train_loader, h_valid, h_test, valid_mask, test_mask,
          valid_ground_truth_list, test_ground_truth_list, data_type):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_recall = 0
    early_stop_count = 0
    early_stop = False

    batches = len(train_loader.dataset) // args.batch_size[data_type]
    if len(train_loader.dataset) % args.batch_size[data_type] != 0:
        batches += 1
    print('Total training batches = {}'.format(batches))

    all_start_time = time.time()
    epoch = 0
    while True:
        epoch += 1
        print(f'epoch:{epoch}')
        model.train()
        start_time = time.time()

        for batch, (users, pos_items) in enumerate(tqdm(train_loader)):
            model.zero_grad()
            loss = model(users, pos_items)
            loss.backward()
            optimizer.step()

        train_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))

        need_valid = True

        if need_valid:
            start_time = time.time()
            F1_score, Precision, Recall, NDCG, Entropy, Item_Coverage, Gini_Index = test(model, h_valid,
                                                                                         valid_ground_truth_list,
                                                                                         valid_mask,
                                                                                         [args.top_k[-1]])
            test_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
            all_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - all_start_time))
            print(
                'The time for epoch {} is: Time-consuming = {}, train time = {}, test time = {}'.format(epoch, all_time,
                                                                                                        train_time,
                                                                                                        test_time))
            print(
                "Top{:d} \t F1-score: {:5f} \t Precision: {:.5f}\t Recall: {:.5f}\tNDCG: {:.5f}\tEntropy: {:.5f}\t Item_Coverage: {:.5f}\t Gini_Index: {:.5f}".format(
                    args.top_k[-1], F1_score[args.top_k[-1]], Precision[args.top_k[-1]], Recall[args.top_k[-1]],
                    NDCG[args.top_k[-1]], Entropy[args.top_k[-1]], Item_Coverage[args.top_k[-1]],
                    Gini_Index[args.top_k[-1]]))

            if Recall[args.top_k[-1]] > best_recall:
                best_recall = Recall[args.top_k[-1]]
                early_stop_count = 0
                torch.save(model.state_dict(), args.model_save_path)
                print(f'Saving current best:{args.model_save_path}')
            else:
                early_stop_count += 1
                if early_stop_count >= args.patience:
                    early_stop = True
        print(f'early_stop_count:{early_stop_count}')
        if early_stop:
            print('##########################################')
            all_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - all_start_time))
            print('Early stop is triggered at {} epochs. Time-consuming = {}'.format(epoch, all_time))
            print('The best model is saved at {}'.format(args.model_save_path))
            break
    model.load_state_dict(torch.load(args.model_save_path))
    print(f'Loading model structure and parameters from :{args.model_save_path}')
    F1_score, Precision, Recall, NDCG, Entropy, Item_Coverage, Gini_Index = test(model, h_test,
                                                                                 test_ground_truth_list,
                                                                                 test_mask, args.top_k)
    print('Test Results:')
    for k in args.top_k:
        print(
            "Top{:d} \t F1-score: {:5f} \t Precision: {:.5f}\t Recall: {:.5f}\tNDCG: {:.5f}\tEntropy: {:.5f}\t Item_Coverage: {:.5f}\tGini_Index: {:.5f}".format(
                k, F1_score[k], Precision[k], Recall[k], NDCG[k], Entropy[k], Item_Coverage[k], Gini_Index[k]))
    return model


def test(model, test_u, test_ground_truth_list, mask, topk):
    with torch.no_grad():
        model.eval()
        F1_score, Recall, Precision, NDCG, Entropy, Item_Coverage, Gini_Index = dict(), dict(), dict(), dict(), dict(), dict(), dict()
        rating_all = model.test_foward(range(args.n_head))
        rating_all = rating_all.cpu()
        rating_all += mask
        for k in topk:
            _, rating_list_all = torch.topk(rating_all, k=k)

            rat = rating_list_all.numpy()
            Entropy[k] = get_entropy(rat)
            Item_Coverage[k] = get_coverage(rat, args.n_tail)
            Gini_Index[k] = get_gini(rat, args.n_tail)

            rat = rat[test_u]
            groudtrue = [test_ground_truth_list[u] for u in test_u]

            precision, recall, ndcg = test_one_batch(rat, groudtrue, k)
            Precision[k] = precision / len(test_u)
            Recall[k] = recall / len(test_u)
            NDCG[k] = ndcg / len(test_u)
            F1_score[k] = 2 * (Precision[k] * Recall[k]) / (Precision[k] + Recall[k])

    return F1_score, Precision, Recall, NDCG, Entropy, Item_Coverage, Gini_Index


if __name__ == "__main__":

    args = parse_args_mafengwo()

    if torch.cuda.is_available():
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        gpu_id = int(np.argmax(memory_available))
        args.device = 'cuda:{}'.format(gpu_id)
    else:
        args.device = 'cpu'

    param_grid = {
        'require_data': [['member', 'tuple']]
    }

    for require_data in param_grid['require_data']:
        setup_seed(args.seed)
        args.train_time = str(time.strftime('%m-%d-%H-%M'))
        args.model_save_path = f'./save/{args.dataname}/{args.train_time}.pt'

        print(param_grid)
        args.require_data = require_data
        print(f'require_data:{require_data}')
        print('###################### Tuple Recommendation ######################')

        print('1. Loading Configuration...')
        h_train, t_train, h_valid, h_test, alpha, beta, valid_mask, test_mask, valid_ground_truth_list, test_ground_truth_list = data_param_prepare(
            args)
        train_data = np.column_stack((h_train[args.require_data[0]], t_train[args.require_data[0]]))
        train_data = torch.from_numpy(train_data)

        train_loader = data.DataLoader(train_data, batch_size=args.batch_size[args.require_data[0]], shuffle=True,
                                       collate_fn=collate)

        print('Load Configuration OK, show them below')
        print(args)

        model = CDRec(args, alpha[args.require_data[0]], beta[args.require_data[0]], args.τ[args.require_data[0]])
        model = model.to(args.device)

        model = train(args, model, train_loader, h_valid, h_test, valid_mask, test_mask,
                      valid_ground_truth_list, test_ground_truth_list, args.require_data[0])
        print('########################### Pre-Training end! ###########################')
        if len(args.require_data) > 1:
            model.τ = args.τ[args.require_data[1]]
            model.alpha = torch.tensor(alpha[args.require_data[1]].todense(), device=args.device)
            model.beta = torch.tensor(beta[args.require_data[1]].todense(), device=args.device)
            model.pre_embeds = model.embeds.weight.detach().clone()

            train_data = np.column_stack((h_train[args.require_data[1]], t_train[args.require_data[1]]))
            train_data = torch.from_numpy(train_data)
            train_loader = data.DataLoader(train_data, batch_size=args.batch_size[args.require_data[1]], shuffle=True,
                                           collate_fn=collate)
            train(args, model, train_loader, h_valid, h_test, valid_mask, test_mask,
                  valid_ground_truth_list, test_ground_truth_list, args.require_data[1])
        print('############################# Training end! #############################')
