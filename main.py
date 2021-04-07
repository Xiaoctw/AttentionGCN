from model import *
from utils import *
import argparse
import warnings
import torch
from train import *
#没有套用AMGCN框架的模型的main
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()

parser.add_argument('--lr', type=float, default=3e-4,
                    help='The learning rate')
parser.add_argument('--cuda', type=bool, default=True)
parser.add_argument('--epochs', type=list, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--dataset', type=str, default='BlogCatalog')
parser.add_argument('--print_every', type=int, default=50)
parser.add_argument('--hidden_dim', type=int, default=400)
parser.add_argument('--embedding_dim', type=int, default=32)


torch.cuda.set_device(1)
args = parser.parse_args()
adj_mat, ripple_mat, feat_mat, labels = load_data(args.dataset)
N = adj_mat.shape[0]
num_feat = feat_mat.shape[1]
ripple_mat = normalize_np(ripple_mat)
adj_mat = normalize_np(adj_mat)
# ppmi_mat = PPMI_matrix(surfing_mat)
# ppmi_mat = torch.from_numpy(ppmi_mat).float()
ripple_mat = torch.from_numpy(ripple_mat).float()
adj_mat = torch.from_numpy(adj_mat).float()
feat_mat = torch.from_numpy(feat_mat).float()
labels = torch.LongTensor(labels)
GPU = args.cuda and torch.cuda.is_available()

lr = args.lr
epochs = args.epochs
model = RippleGCN(num_feat=num_feat, num_hidden=args.hidden_dim,
                  embedding_dim=args.embedding_dim, n_class=labels.max().item() + 1)
# model = DeepRipple(input_dim=N, hidden_dims=params['hidden_dims'], output_dim=params['output_dim'],)
#
#
# # 训练得到模型，该模型为采用了attention的自编码器模型
if __name__ == '__main__':
    print('节点数量:{}'.format(N))
    print('特征个数:{}'.format(num_feat))
    print(adj_mat.shape)
    idx_train = range((N//5)*2)
    idx_val = range(N // 2, )
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    # print(ripple_mat.shape)
    # train(model, surfing_mat, ripple_mat,b=1, epochs=epochs, lr=lr, batch_size=args.batch_size, print_every=args.print_every,
    #       GPU=GPU)
    trainRippleGCN(model, adj1=adj_mat, adj2=ripple_mat, Y=labels, idx_train=idx_train, idx_val=idx_val, epochs=epochs,
                   lr=lr, print_every=args.print_every, GPU=GPU, feat_X=feat_mat)
    model = model.cpu()
    w, embs = model.emb(feat_mat, adj_mat, ripple_mat, require_weight=True)
    embs = embs.cpu().detach().numpy()
    save_path = Path(__file__).parent / 'result' / (args.dataset + '_outVec.txt')
    np.savetxt(save_path, embs)
    print(w[:20, :])
    # out1, out2 = model(feat_X,surfing_mat, ripple_mat)
    # print('ppmi:')
    # print(out1[0][:15])
    # print(surfing_mat[0][:15])
    # # print(out2)
    # print('---------------')
    # print('ripple_mat:')
    # print(out2[0][:15])
    # print(ripple_mat[0][:15])
