"""
这个是基于surf和attention的模型的main

"""
from model import *
from utils import *
import argparse

import warnings
import torch
from train import *

warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=3e-4,
                    help='The learning rate')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--epochs', type=int, default=400,
                    help='Number of epochs to train.')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--print_every', type=int, default=10)
parser.add_argument('--hidden_dim', type=int, default=400)
parser.add_argument('--embedding_dim', type=int, default=32)
# num_feat, num_hidden, n_class
# l_c参数
parser.add_argument('--alpha', type=float, default=0.01)
# l_d参数
parser.add_argument('--beta', type=float, default=1e-2)
parser.add_argument('--sigma', type=float, default=1)
parser.add_argument('--surf_epoch', type=int, default=2)
# BlogCatalog 0.01, l_s参数
parser.add_argument('--theta', type=float, default=0.03)
parser.add_argument('--l_s', default=True, type=str2bool)
parser.add_argument('--l_t', default=True, type=str2bool)
parser.add_argument('--l_d', default=True, type=str2bool)

args = parser.parse_args()
adj_mat, ripple_mat, feat_mat, labels = load_data(args.dataset)
N = adj_mat.shape[0]
num_feat = feat_mat.shape[1]
surf_adj_mat = random_surfing(adj_mat, epochs=args.surf_epoch, alpha=0.5)
ripple_mat = normalize_np(ripple_mat)
adj_mat = normalize_np(adj_mat)

ripple_mat = torch.from_numpy(ripple_mat).float()
adj_mat = torch.from_numpy(adj_mat).float()
feat_mat = torch.from_numpy(feat_mat).float()
surf_adj_mat = torch.from_numpy(surf_adj_mat).float()
print('surfing矩阵构造完成')
labels = torch.LongTensor(labels)
GPU = args.cuda and torch.cuda.is_available()
if GPU:
    torch.cuda.set_device(1)
if args.dataset == 'pubmed':
    GPU = False
#torch.cuda.set_device(1)
lr = args.lr
epochs = args.epochs
model = SurfRippleGCN(num_feat=num_feat, num_hidden=args.hidden_dim, alpha=args.alpha, beta=args.beta, theta=args.theta,
                      embedding_dim=args.embedding_dim, n_class=labels.max().item() + 1, num_layer=args.surf_epoch)
# model = DeepRipple(input_dim=N, hidden_dims=params['hidden_dims'], output_dim=params['output_dim'],)
# # 训练得到模型，该模型为采用了attention的自编码器模型
if __name__ == '__main__':
    print('节点数量:{}'.format(N))
    print('特征个数:{}'.format(num_feat))
    print(adj_mat.shape)
    idx_train, idx_val = shuffle(N, N//5, N // 3)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    # print(ripple_mat.shape) train(model, surfing_mat, ripple_mat,b=1, epochs=epochs, lr=lr,
    # batch_size=args.batch_size, print_every=args.print_every, GPU=GPU)
    print('args.l_t:{}'.format(args.l_t))
    trainSurfRippleGCN(model, adj_a=adj_mat, adj_r=ripple_mat, surf_adj=surf_adj_mat, Y=labels,
                       idx_train=idx_train, idx_val=idx_val,
                       epochs=epochs,
                       lr=lr, print_every=args.print_every, GPU=GPU, feat_X=feat_mat, l_d=args.l_d, l_s=args.l_s,
                       l_t=args.l_t)
    model = model.cpu()
    _, _, _, _, w, embs = model.emb(feat_mat, adj_mat, ripple_mat, require_weight=True)
    embs = embs.cpu().detach().numpy()
    save_path = Path(__file__).parent / 'surf_ripple_result' / (args.dataset + '_outVec.txt')
    w = w.detach().numpy()
    np.savetxt(save_path, embs)
    weights1 = w[:, 0]
    weights2 = w[:, 1]
    weights3 = w[:, 2]
    # w_a, w_r, w_c
    print('邻接矩阵上GCN权重:{}'.format(np.mean(weights1)))
    print('ripple矩阵上GCN权重:{}'.format(np.mean(weights2)))
    print('combined部分GCN权重:{}'.format(np.mean(weights3)))
    # print(np.mean(weights3))
   # print(w[:20, :])
    # out1, out2 = model(feat_X,surfing_mat, ripple_mat)
    # print('ppmi:')
    # print(out1[0][:15])
    # print(surfing_mat[0][:15])
    # # print(out2)
    # print('---------------')
    # print('ripple_mat:')
    # print(out2[0][:15])
    # print(ripple_mat[0][:15])
