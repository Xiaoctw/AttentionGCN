import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch
import torch.utils.data as Data
import time
import torch.optim as optim


class DataSet(Data.Dataset):

    def __init__(self, mat1, mat2):
        self.mat1 = mat1
        self.mat2 = mat2
        self.num_node = self.mat1.shape[0]

    def __getitem__(self, index):
        return self.mat1[index], self.mat2[index]

    def __len__(self):
        return self.num_node


class OneDataSet(Data.Dataset):

    def __init__(self, mat1):
        self.mat1 = mat1
        self.num_node = self.mat1.shape[0]

    def __getitem__(self, index):
        return self.mat1[index]

    def __len__(self):
        return self.num_node


def trainRippleGCN(model: nn.Module, adj1, adj2, Y, idx_train, idx_val, epochs, lr, print_every, GPU=False,
                   feat_X=None):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, )
    optimizer.zero_grad()
    if feat_X is None:
        X = torch.FloatTensor(np.diag(np.ones(Y.shape[0])))
    else:
        X = feat_X
    if GPU:
        model = model.cuda()
        adj1 = adj1.cuda()
        adj2 = adj2.cuda()
        X = X.cuda()
        Y = Y.cuda()
    pre_time = time.time()
    for epoch in range(epochs):
        output = model(X, adj1, adj2)
        loss = F.nll_loss(output[idx_train], Y[idx_train])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % print_every == 0 or epoch == 0:
            model.eval()
            tem_time = time.time()
            print(' epoch:{}, loss:{:.4f}, time:{:.4f}, accuracy:{}'.format(epoch + 1, loss,
                                                                            (tem_time - pre_time) / print_every,
                                                                            accuracy(output[idx_val], Y[idx_val])))
            pre_time = tem_time
            model.train()
    return model


def trainNewRippleGCN(model: nn.Module, adj1, adj2, Y, idx_train, idx_val, epochs, lr,
                      print_every,
                      GPU=False,
                      feat_X=None):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, )
    optimizer.zero_grad()
    if feat_X is None:
        X = torch.FloatTensor(np.diag(np.ones(Y.shape[0])))
    else:
        X = feat_X
    if GPU:
        model = model.cuda()
        adj1 = adj1.cuda()
        adj2 = adj2.cuda()
        X = X.cuda()
        Y = Y.cuda()
    pre_time = time.time()
    for epoch in range(epochs):
        # output = model(X, adj1, adj2)
        lt, lc, ld, output = model.compute_loss(X, adj1, adj2, Y, idx_train)
        loss = lt + lc + ld
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % print_every == 0 or epoch == 0:
            model.eval()
            tem_time = time.time()
            print(' epoch:{}, lt:{:.4f}, lc:{:.4f}, ld:{:.4f}, time:{:.4f}, accuracy:{}'.format(epoch + 1, lt, lc, ld,
                                                                                                (
                                                                                                        tem_time - pre_time) / print_every,
                                                                                                accuracy(
                                                                                                    output[idx_val],
                                                                                                    Y[idx_val])))
            pre_time = tem_time
            model.train()
    return model


def trainSurfRippleGCN(model: nn.Module, adj_a, adj_r, surf_adj, Y, idx_train, idx_val, epochs, lr,
                       print_every,
                       GPU=False,
                       feat_X=None,l_d=True,l_t=True,l_s=True):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    optimizer.zero_grad()
    if feat_X is None:
        X = torch.FloatTensor(np.diag(np.ones(Y.shape[0])))
    else:
        X = feat_X
    if GPU:
        model = model.cuda()
        adj_a = adj_a.cuda()
        adj_r = adj_r.cuda()
        surf_adj = surf_adj.cuda()
        X = X.cuda()
        Y = Y.cuda()
    pre_time = time.time()
    pre_val = None
    no_improve = 0
    weight1s, weight2s, weight3s = [], [], []
    for epoch in range(epochs):
        # output = model(X, adj1, adj2)
      #  print('l_t:{}'.format(l_t))
        lt, lc, ld, ls, output = model.compute_loss(X, adj_a, adj_r, surf_adj, Y, idx_train,l_s=l_s,l_d=l_d,l_t=l_t)
        loss = lt + lc + ld + ls
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc_val = accuracy(output[idx_val], Y[idx_val])
        if (epoch + 1) % print_every == 0 or epoch == 0:
            model.eval()
            tem_time = time.time()
            print(' epoch:{}, lt:{:.4f}, lc:{:.4f}, ld:{:.4f}, ls:{:4f}, time:{:.4f}, accuracy:{}'.format(epoch + 1, lt,
                                                                                                          lc, ld, ls,
                                                                                                          (
                                                                                                                      tem_time - pre_time) / print_every,
                                                                                                          acc_val))
            pre_time = tem_time
            model.train()
        if (epoch + 1) % (2 * print_every) == 0 or epoch == 0:
            model.eval()
            w = model.attn_weight(X, adj_a, adj_r)
            weights1 = w[:, 0]
            weights2 = w[:, 1]
            weights3 = w[:, 2]
            weight1s.append(torch.mean(weights1).item())
            weight2s.append(torch.mean(weights2).item())
            weight3s.append(torch.mean(weights3).item())
            model.train()

        if pre_val is None or acc_val>pre_val:
            no_improve=0
        else:
            no_improve+=1
        if no_improve==3:
            #break
            pass
        pre_val = acc_val
    print('邻接矩阵上GCN权重:{}'.format(weight1s))
    print('combined部分GCN权重:{}'.format(weight2s))
    print('ripple矩阵上GCN权重:{}'.format(weight3s))
    return model


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)  # 使用type_as(tesnor)将张量转换为给定类型的张量。
    correct = preds.eq(labels).double()  # 记录等于preds的label eq:equal
    correct = correct.sum()
    return correct / len(labels)
