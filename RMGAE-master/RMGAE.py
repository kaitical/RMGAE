import argparse
import statistics
import time
import setting
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from sklearn.preprocessing import normalize
from setting.config import get_config


def parse_args():
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--file', type=str, default='mag_eng', help='Dataset file name (without path)')
    parser.add_argument('--hidden_sizes', nargs='+', type=int, default=[128], help='Hidden layer sizes')
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_norm', default=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--display_step', type=int, default=25)
    parser.add_argument('--balance_loss', default=True)
    parser.add_argument('--stochastic_loss', default=True)
    parser.add_argument('--mask_ratio', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


class NF1(object):
    def __init__(self, communities, ground_truth):
        self.communities = communities
        self.ground_truth = ground_truth

    def get_f1(self):
        gt_coms = {cid: nodes for cid, nodes in enumerate(self.ground_truth)}
        ext_coms = {cid: nodes for cid, nodes in enumerate(self.communities)}

        f1_list = []
        for cid, nodes in gt_coms.items():
            tmp = [self.__compute_f1(nodes2, nodes) for _, nodes2 in ext_coms.items()]
            f1_list.append(np.max(tmp))

        f2_list = []
        for cid, nodes in ext_coms.items():
            tmp = [self.__compute_f1(nodes, nodes2) for _, nodes2 in gt_coms.items()]
            f2_list.append(np.max(tmp))

        return (np.mean(f1_list) + np.mean(f2_list)) / 2

    def __compute_f1(self, c, gt):
        c = set(c)
        gt = set(gt)
        try:
            precision = len(c & gt) / len(c)
            recall = len(c & gt) / len(gt)
            z = 2 * (precision * recall) / (precision + recall)
            return float("%.2f" % z)
        except ZeroDivisionError:
            return 0.0


def get_nmi(gnn, x_norm, adj_norm, Z_gt):
    gnn.eval()
    Z = F.relu(gnn(x_norm, adj_norm))
    Z_pred = Z.cpu().detach().numpy() > 0.5
    return setting.metrics.overlapping_nmi(Z_pred, Z_gt)


def get_overlapping_f1(Z_pred, Z_gt):
    pred_coms = [np.where(Z_pred[:, k])[0].tolist() for k in range(Z_pred.shape[1])]
    gt_coms = [np.where(Z_gt[:, k])[0].tolist() for k in range(Z_gt.shape[1])]
    return NF1(pred_coms, gt_coms).get_f1()


def train(args, A, x_norm, Z_gt, sampler, gnn, adj_norm, decoder, opt):
    val_loss = np.inf
    validation_fn = lambda: val_loss
    early_stopping = setting.train.NoImprovementStopping(validation_fn, patience=10)
    model_saver = setting.train.ModelSaver(gnn)

    for epoch, batch in enumerate(sampler):
        if epoch > args.max_epochs:
            break
        if epoch % args.display_step == 0:
            with torch.no_grad():
                gnn.eval()
                Z = F.relu(gnn(x_norm, adj_norm))
                val_loss = decoder.loss_full(Z, A)
                nmi = get_nmi(gnn, x_norm, adj_norm, Z_gt)
                print(f'Epoch {epoch:4d}, loss.full = {val_loss:.4f}, nmi = {nmi:.2f}')
                early_stopping.next_step()
                if early_stopping.should_save():
                    model_saver.save()
                if early_stopping.should_stop():
                    print(f'Breaking due to early stopping at epoch {epoch}')
                    break

        gnn.train()
        opt.zero_grad()
        Z = F.relu(gnn(x_norm, adj_norm))
        ones_idx, zeros_idx, mask_idx = batch

        if args.stochastic_loss:
            loss = decoder.loss_batch(Z, ones_idx, zeros_idx, mask_idx)
        else:
            loss = decoder.loss_full(Z, A, mask_idx)
        loss += setting.utils.l2_reg_loss(gnn, scale=args.weight_decay)
        loss.backward()
        opt.step()

    model_saver.restore()
    gnn.eval()
    Z = F.relu(gnn(x_norm, adj_norm))
    Z_pred = Z.cpu().detach().numpy() > 0.5
    nmi = setting.metrics.overlapping_nmi(Z_pred, Z_gt)
    f1 = get_overlapping_f1(Z_pred, Z_gt)
    print(f'Final NMI = {nmi:.3f}, Overlapping F1 = {f1:.3f}')
    return nmi, f1


if __name__ == "__main__":
    args = parse_args()
    torch.set_default_dtype(torch.float32)
    torch.set_default_device(args.device)

    dataset_path = f"data/{args.file}.npz"
    #
    batch_size ,mask_size ,args.mask_ratio ,args.alpha = get_config(args.file)

    loader = setting.data.load_dataset2(dataset_path, mask_ratio=args.mask_ratio)
    A, X, Z_gt, mask_idx = loader['A'], loader['X'], loader['Z'], loader['masked_edges']
    N, K = Z_gt.shape

    x_norm = normalize(X)
    x_norm = setting.utils.to_sparse_tensor(x_norm).to(args.device)
    sampler = setting.sampler2.get_edge_sampler2(A, batch_size, batch_size, mask_idx, mask_size, num_workers=5)
    gnn = setting.nn.GCN(x_norm.shape[1], args.hidden_sizes, K, batch_norm=args.batch_norm, dropout=args.dropout).to(args.device)
    adj_norm = gnn.normalize_adj(A)
    decoder = setting.nn.MaskGAEBerpoDecoder(N, A.nnz, balance_loss=args.balance_loss, a=args.alpha)
    opt = torch.optim.Adam(gnn.parameters(), lr=args.lr)

    res, f1_scores, times = [], [], []
    max_nmi = 0

    for loops in range(10):
        print('loop ', loops)
        st = time.time()
        nmi, f1 = train(args, A, x_norm, Z_gt, sampler, gnn, adj_norm, decoder, opt)
        et = time.time()

        elapsed_time = et - st
        res.append(nmi)
        f1_scores.append(f1)
        times.append(elapsed_time)

        if nmi > max_nmi:
            max_nmi = nmi

        for layer in gnn.layers:
            layer.reset_parameters()

        if loops > 2:
            avg_nmi = statistics.mean(res)
            std_nmi = statistics.stdev(res)
            avg_time = statistics.mean(times)
            avg_f1 = statistics.mean(f1_scores)
            print(f'Average NMI after {loops} loops: {avg_nmi:.3f}')
            print(f'Average F1 after {loops} loops: {avg_f1:.3f}')
            print(f'Standard deviation NMI: {std_nmi:.4f}')
            print(f'Average Time: {avg_time:.2f} seconds')

    print(f'\nFinal average NMI for {args.file}: {avg_nmi:.3f}')
    print(f'Final average F1 for {args.file}: {avg_f1:.3f}')
    print(f'Maximum NMI achieved: {max_nmi:.3f}')