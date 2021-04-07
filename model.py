from abc import ABC

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import math


class GraphConvolution(nn.Module):
    """
    GCN的某一层
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.norm = nn.BatchNorm1d(out_features)
        self.init_parameters()

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        output = self.norm(output)
        return output

    def init_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        nn.init.xavier_uniform_(self.weight, gain=1)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -stdv, stdv)


class GCN(nn.Module):
    """
    GCN的整个模型
    """
    def __init__(self, nfeat, nhid, nclass, dropout):  # 底层节点的参数，feature的个数；隐层节点个数；最终的分类数
        super(GCN, self).__init__()  # super()._init_()在利用父类里的对象构造函数
        self.gc1 = GraphConvolution(nfeat, nhid)  # gc1输入尺寸nfeat，输出尺寸nhid
        self.gc2 = GraphConvolution(nhid, nclass)  # gc2输入尺寸nhid，输出尺寸ncalss
        self.dropout = dropout

    # 输入分别是特征和邻接矩阵。最后输出为输出层做log_softmax变换的结果
    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))  # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
        x = F.dropout(x, self.dropout, training=self.training)  # x要dropout
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def savector(self, x, adj):
        x = F.relu(self.gc1(x, adj))  # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
        x = F.dropout(x, self.dropout, training=self.training)  # x要dropout
        x = self.gc2(x, adj)
        return x


class RippleGCN(nn.Module):
    def __init__(self, num_feat, num_hidden, n_class, embedding_dim=32, dropout=0.5):
        super(RippleGCN, self).__init__()
        self.dropout = dropout
        self.gc1 = GraphConvolution(num_feat, num_hidden)
        self.gc2 = GraphConvolution(num_hidden, embedding_dim)
        self.gc3 = GraphConvolution(num_feat, num_hidden)
        self.gc4 = GraphConvolution(num_hidden, embedding_dim)
        self.W1 = nn.Parameter(torch.FloatTensor(embedding_dim, embedding_dim))
        self.b1 = nn.Parameter(torch.FloatTensor(embedding_dim))
        self.W2 = nn.Parameter(torch.FloatTensor(embedding_dim, embedding_dim))
        self.b2 = nn.Parameter(torch.FloatTensor(embedding_dim))
        self.Q = nn.Parameter(torch.FloatTensor(embedding_dim, 1))
        # 得到嵌入向量之后再进行一步全连接
        self.linear = nn.Linear(embedding_dim, n_class)
        self.norm1 = nn.BatchNorm1d(embedding_dim)
        self.norm2 = nn.BatchNorm1d(n_class)
        stdv = 1. / math.sqrt(embedding_dim)
        nn.init.xavier_uniform_(self.W1, gain=1)
        nn.init.xavier_uniform_(self.W2, gain=1)
        nn.init.uniform_(self.b1, -stdv, stdv)
        nn.init.uniform_(self.b2, -stdv, stdv)
        nn.init.uniform_(self.Q, -stdv, stdv)

    def forward(self, x, adj1, adj2):
        x = self.emb(x, adj1, adj2)
        x = self.linear(x)
        x = self.norm2(x)
        return torch.log_softmax(x, dim=1)

    def emb(self, x, adj1, adj2, require_weight=False):
        x1 = F.leaky_relu(self.gc1(x, adj1), negative_slope=0.2)  # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
        x1 = F.dropout(x1, self.dropout, training=self.training)  # x要dropout
        x1 = self.gc2(x1, adj1)
        x2 = F.leaky_relu(self.gc3(x, adj2), negative_slope=0.2)  # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
        x2 = F.dropout(x2, self.dropout, training=self.training)  # x要dropout
        x2 = self.gc4(x2, adj2)
        w1 = F.leaky_relu((torch.mm(x1, self.W1) + self.b1).mm(self.Q), negative_slope=0.2)
        w2 = F.leaky_relu((torch.mm(x2, self.W2) + self.b2).mm(self.Q), negative_slope=0.2)
        w = torch.softmax(torch.cat((w1, w2), dim=1), dim=1)
        # print(x1.shape)
        # print(w[:,0].shape)
        emb = w[:, 0].reshape(-1, 1) * x1 + w[:, 1].reshape(-1, 1) * x2
        emb = self.norm1(emb)
        # print(w)
        if require_weight:
            return w, emb
        return emb


class AmRippleGCN(nn.Module):
    def __init__(self, num_feat, num_hidden, n_class, alpha, beta, embedding_dim=32, dropout=0.5):
        super(AmRippleGCN, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.alpha = alpha
        self.beta = beta
        for i in range(1, 4):
            setattr(self, 'gc{}'.format(2 * i - 1), GraphConvolution(num_feat, num_hidden))
            setattr(self, 'gc{}'.format(2 * i), GraphConvolution(num_hidden, embedding_dim))
            setattr(self, 'W{}'.format(i), nn.Parameter(torch.FloatTensor(embedding_dim, embedding_dim)))
            setattr(self, 'b{}'.format(i), nn.Parameter(torch.FloatTensor(embedding_dim)))
        self.Q = nn.Parameter(torch.FloatTensor(embedding_dim, 1))
        self.linear = nn.Linear(embedding_dim, n_class)
        self.norm1 = nn.BatchNorm1d(embedding_dim)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1. / math.sqrt(self.embedding_dim)
        nn.init.xavier_uniform_(self.W1, gain=1)
        nn.init.xavier_uniform_(self.W2, gain=1)
        nn.init.xavier_uniform_(self.W3, gain=1)
        nn.init.xavier_uniform_(self.linear.weight, gain=1)
        # nn.init.xavier_uniform_(self)
        nn.init.uniform_(self.b1, -stdv, stdv)
        nn.init.uniform_(self.b2, -stdv, stdv)
        nn.init.uniform_(self.b3, -stdv, stdv)
        nn.init.uniform_(self.Q, -stdv, stdv)
        self.norm1.weight.data.fill_(1)
        self.norm1.bias.data.zero_()

    def emb(self, x, adj1, adj2, require_weight=False):
        x1 = F.leaky_relu(self.gc1(x, adj1), negative_slope=0.2)  # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
        x1 = F.dropout(x1, self.dropout, training=self.training)  # x要dropout
        x1 = self.gc2(x1, adj1)
        x2 = F.leaky_relu(self.gc3(x, adj2), negative_slope=0.2)  # adj即公式Z=softmax(A~Relu(A~XW(0))W(1))中的A~
        x2 = F.dropout(x2, self.dropout, training=self.training)  # x要dropout
        x2 = self.gc4(x2, adj2)
        x1_c = F.leaky_relu(self.gc5(x, adj1), negative_slope=0.2)
        x1_c = F.dropout(x1_c, self.dropout, training=self.training)
        x1_c = self.gc6(x1_c, adj1)
        x2_c = F.leaky_relu(self.gc5(x, adj2), negative_slope=0.2)
        x2_c = F.dropout(x2_c, self.dropout, training=self.training)
        x2_c = self.gc6(x2_c, adj1)
        x_c = (x1_c + x2_c) / 2
        w1 = F.leaky_relu((torch.mm(x1, self.W1) + self.b1).mm(self.Q), negative_slope=0.2)
        w2 = F.leaky_relu((torch.mm(x2, self.W2) + self.b2).mm(self.Q), negative_slope=0.2)
        w3 = F.leaky_relu((torch.mm(x_c, self.W3) + self.b3).mm(self.Q), negative_slope=0.2)
        w = torch.softmax(torch.cat((w1, w2, w3), dim=1), dim=1)
        # print(x1.shape)
        # print(w[:,0].shape)
        emb = w[:, 0].reshape(-1, 1) * x1 + w[:, 1].reshape(-1, 1) * x2 + w[:, 2].reshape(-1, 1) * x_c
        emb = self.norm1(emb)
        if require_weight:
            return x1, x2, x1_c, x2_c, w, emb
        return x1, x2, x1_c, x2_c, emb

    def forward(self, x, adj1, adj2):
        x1, x2, x1_c, x2_c, emb = self.emb(x, adj1, adj2)
        x = self.linear(emb)
        # x=self.norm2(x)
        return x1, x2, x1_c, x2_c, emb, torch.log_softmax(x, dim=1)

    def compute_loss(self, x, adj1, adj2, Y, idx_train):
        x1, x2, x1_c, x2_c, emb, output = self.forward(x, adj1, adj2)
        l_t = F.nll_loss(output[idx_train], Y[idx_train])
        # combined的向量尽可能接近
        l_c = torch.mean(torch.pow(torch.mm(x1_c, x1_c.t()) - torch.mm(x2_c, x2_c.t()), 2))
        k_x1 = self.kernel(x1)
        k_x1_c = self.kernel(x1_c)
        k_x2 = self.kernel(x2)
        k_x2_c = self.kernel(x2_c)
        # 这里的向量尽可能远离
        l_d = self.hsic(k_x1, k_x1_c) + self.hsic(k_x2, k_x2_c)
        # print('l_t{},l_c:{},l_d:{}'.format(l_t,l_c,l_d))
        return l_t, self.alpha * l_c, self.beta * l_d, output

    def hsic(self, kX, kY):
        kXY = torch.mm(kX, kY)
        n = kXY.shape[0]
        # print(kXY.shape)
        h = torch.trace(kXY) / (n * n) + torch.mean(kX) * torch.mean(kY) - 2 * torch.mean(kXY) / n
        return h * n ** 2 / (n - 1) ** 2

    def kernel(self, X):
        n = X.shape[0]
        square_x = torch.pow(X, 2)
        sum_square_x = torch.sum(square_x, 1)
        sum_mat = sum_square_x.view(n, 1) + sum_square_x.view(1, n)
        sum_mat = sum_mat - 2 * torch.mm(X, X.t())
        return sum_mat


class SurfRippleGCN(nn.Module):
    def __init__(self, num_feat, num_hidden, n_class, alpha, beta, theta, num_layer=2, embedding_dim=32, dropout=0.5):
        super(SurfRippleGCN, self).__init__()
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.num_layer = num_layer
        if self.num_layer==1:
            setattr(self, 'gc_a_{}'.format(1), GraphConvolution(num_feat, embedding_dim))
            setattr(self, 'gc_c_{}'.format(1), GraphConvolution(num_feat, embedding_dim))
            setattr(self, 'gc_r_{}'.format(1), GraphConvolution(num_feat, embedding_dim))
        else:
            setattr(self, 'gc_a_{}'.format(1), GraphConvolution(num_feat, num_hidden))
            setattr(self, 'gc_c_{}'.format(1), GraphConvolution(num_feat, num_hidden))
            setattr(self, 'gc_r_{}'.format(1), GraphConvolution(num_feat, num_hidden))
            for i in range(2, num_layer):
                setattr(self, 'gc_a_{}'.format(i), GraphConvolution(num_hidden, num_hidden))
                setattr(self, 'gc_c_{}'.format(i), GraphConvolution(num_hidden, num_hidden))
                setattr(self, 'gc_r_{}'.format(i), GraphConvolution(num_hidden, num_hidden))
            setattr(self, 'gc_a_{}'.format(num_layer), GraphConvolution(num_hidden, embedding_dim))
            setattr(self, 'gc_c_{}'.format(num_layer), GraphConvolution(num_hidden, embedding_dim))
            setattr(self, 'gc_r_{}'.format(num_layer), GraphConvolution(num_hidden, embedding_dim))
        self.W_a = nn.Parameter(torch.FloatTensor(embedding_dim, embedding_dim))
        self.b_a = nn.Parameter(torch.FloatTensor(embedding_dim))
        self.W_c = nn.Parameter(torch.FloatTensor(embedding_dim, embedding_dim))
        self.b_c = nn.Parameter(torch.FloatTensor(embedding_dim))
        self.W_r = nn.Parameter(torch.FloatTensor(embedding_dim, embedding_dim))
        self.b_r = nn.Parameter(torch.FloatTensor(embedding_dim))
        # for i in range(1, 4):
        #     setattr(self, 'gc{}'.format(2 * i - 1), GraphConvolution(num_feat, num_hidden))
        #     setattr(self, 'gc{}'.format(2 * i), GraphConvolution(num_hidden, embedding_dim))
        #     setattr(self, 'W{}'.format(i), nn.Parameter(torch.FloatTensor(embedding_dim, embedding_dim)))
        #     setattr(self, 'b{}'.format(i), nn.Parameter(torch.FloatTensor(embedding_dim)))
        self.Q = nn.Parameter(torch.FloatTensor(embedding_dim, 1))
        self.linear = nn.Linear(embedding_dim, n_class)
        self.norm1 = nn.BatchNorm1d(embedding_dim)
        self.init_parameters()

    def init_parameters(self):
        stdv = 1. / math.sqrt(self.embedding_dim)
        nn.init.xavier_uniform_(self.W_a, gain=1)
        nn.init.xavier_uniform_(self.W_r, gain=1)
        nn.init.xavier_uniform_(self.W_c, gain=1)
        nn.init.xavier_uniform_(self.linear.weight, gain=1)
        # nn.init.xavier_uniform_(self)
        nn.init.uniform_(self.b_a, -stdv, stdv)
        nn.init.uniform_(self.b_c, -stdv, stdv)
        nn.init.uniform_(self.b_r, -stdv, stdv)
        nn.init.uniform_(self.Q, -stdv, stdv)
        self.norm1.weight.data.fill_(1)
        self.norm1.bias.data.zero_()

    def emb(self, x, adj_a, adj_r, require_weight=False):
        x_a = F.leaky_relu((getattr(self, 'gc_a_{}'.format(1))(x, adj_a)), negative_slope=0.2)
        x_a = F.dropout(x_a, self.dropout, training=self.training)
        for i in range(2, self.num_layer + 1):
            x_a = F.leaky_relu((getattr(self, 'gc_a_{}'.format(i))(x_a, adj_a)), negative_slope=0.2)
            if i < self.num_layer:
                x_a = F.dropout(x_a, self.dropout, training=self.training)  # x要dropout
        x_r = F.leaky_relu((getattr(self, 'gc_r_{}'.format(1))(x, adj_r)), negative_slope=0.2)
        x_r = F.dropout(x_r, self.dropout, training=self.training)
        for i in range(2, self.num_layer + 1):
            x_r = F.leaky_relu((getattr(self, 'gc_r_{}'.format(i))(x_r, adj_r)), negative_slope=0.2)
            if i < self.num_layer:
                x_r = F.dropout(x_r, self.dropout, training=self.training)  # x要dropout
        x_c_a = F.leaky_relu((getattr(self, 'gc_c_{}'.format(1))(x, adj_a)), negative_slope=0.2)
        x_c_a = F.dropout(x_c_a, self.dropout, training=self.training)
        for i in range(2, self.num_layer + 1):
            x_c_a = F.leaky_relu((getattr(self, 'gc_c_{}'.format(i))(x_c_a, adj_a)), negative_slope=0.2)
            if i < self.num_layer:
                x_c_a = F.dropout(x_c_a, self.dropout, training=self.training)  # x要dropout
        x_c_r = F.leaky_relu((getattr(self, 'gc_c_{}'.format(1))(x, adj_r)), negative_slope=0.2)
        x_c_r = F.dropout(x_c_r, self.dropout, training=self.training)
        for i in range(2, self.num_layer + 1):
            x_c_r = F.leaky_relu((getattr(self, 'gc_c_{}'.format(i))(x_c_r, adj_r)), negative_slope=0.2)
            if i < self.num_layer:
                x_c_r = F.dropout(x_c_r, self.dropout, training=self.training)  # x要dropout
        x_c = (x_c_r + x_c_a) / 2
        w_a = F.leaky_relu((torch.mm(x_a, self.W_a) + self.b_a).mm(self.Q), negative_slope=0.2)
        w_r = F.leaky_relu((torch.mm(x_r, self.W_r) + self.b_r).mm(self.Q), negative_slope=0.2)
        w_c = F.leaky_relu((torch.mm(x_c, self.W_c) + self.b_c).mm(self.Q), negative_slope=0.2)
        w = torch.softmax(torch.cat((w_a, w_r, w_c), dim=1), dim=1)
        emb = w[:, 0].reshape(-1, 1) * x_a + w[:, 1].reshape(-1, 1) * x_r + w[:, 2].reshape(-1, 1) * x_c
        emb = self.norm1(emb)
        if require_weight:
            return x_a, x_r, x_c_a, x_c_r, w, emb
        return x_a, x_r, x_c_a, x_c_r, emb

    def attn_weight(self, x, adj_a, adj_r):
        x_a = F.leaky_relu((getattr(self, 'gc_a_{}'.format(1))(x, adj_a)), negative_slope=0.2)
        x_a = F.dropout(x_a, self.dropout, training=self.training)
        for i in range(2, self.num_layer + 1):
            x_a = F.leaky_relu((getattr(self, 'gc_a_{}'.format(i))(x_a, adj_a)), negative_slope=0.2)
            if i < self.num_layer:
                x_a = F.dropout(x_a, self.dropout, training=self.training)  # x要dropout
        x_r = F.leaky_relu((getattr(self, 'gc_r_{}'.format(1))(x, adj_r)), negative_slope=0.2)
        x_r = F.dropout(x_r, self.dropout, training=self.training)
        for i in range(2, self.num_layer + 1):
            x_r = F.leaky_relu((getattr(self, 'gc_r_{}'.format(i))(x_r, adj_r)), negative_slope=0.2)
            if i < self.num_layer:
                x_r = F.dropout(x_r, self.dropout, training=self.training)  # x要dropout
        x_c_a = F.leaky_relu((getattr(self, 'gc_c_{}'.format(1))(x, adj_a)), negative_slope=0.2)
        x_c_a = F.dropout(x_c_a, self.dropout, training=self.training)
        for i in range(2, self.num_layer + 1):
            x_c_a = F.leaky_relu((getattr(self, 'gc_c_{}'.format(i))(x_c_a, adj_a)), negative_slope=0.2)
            if i < self.num_layer:
                x_c_a = F.dropout(x_c_a, self.dropout, training=self.training)  # x要dropout

        x_c_r = F.leaky_relu((getattr(self, 'gc_c_{}'.format(1))(x, adj_r)), negative_slope=0.2)
        x_c_r = F.dropout(x_c_r, self.dropout, training=self.training)
        for i in range(2, self.num_layer + 1):
            x_c_r = F.leaky_relu((getattr(self, 'gc_c_{}'.format(i))(x_c_r, adj_r)), negative_slope=0.2)
            if i < self.num_layer:
                x_c_r = F.dropout(x_c_r, self.dropout, training=self.training)  # x要dropout

        x_c = (x_c_r + x_c_a) / 2
        w_a = F.leaky_relu((torch.mm(x_a, self.W_a) + self.b_a).mm(self.Q), negative_slope=0.2)
        w_r = F.leaky_relu((torch.mm(x_r, self.W_r) + self.b_r).mm(self.Q), negative_slope=0.2)
        w_c = F.leaky_relu((torch.mm(x_c, self.W_c) + self.b_c).mm(self.Q), negative_slope=0.2)
        w = torch.softmax(torch.cat((w_a, w_r, w_c), dim=1), dim=1)
        return w

    def forward(self, x, adj_a, adj_r):
        x1, x2, x1_c, x2_c, emb = self.emb(x, adj_a, adj_r)
        x = self.linear(emb)
        return x1, x2, x1_c, x2_c, emb, torch.log_softmax(x, dim=1)

    def compute_loss(self, x, adj_a, adj_r, surf_adj_a, Y, idx_train, l_d=True, l_t=True, l_s=True):
        x1, x2, x1_c, x2_c, emb, output = self.forward(x, adj_a, adj_r)
        # print('model:{}'.format(l_t))
        if l_t:
            l_t = F.nll_loss(output[idx_train], Y[idx_train])
        else:
            l_t = 0
        # combined的向量尽可能接近
        l_c = torch.mean(torch.pow(torch.mm(x1_c, x1_c.t()) - torch.mm(x2_c, x2_c.t()), 2))
        fit_mat = torch.mm(emb, emb.t())
        if l_s:
            l_s = torch.mean((fit_mat - surf_adj_a).pow(2)) + torch.mean(torch.abs(fit_mat - surf_adj_a))
        else:
            l_s = 0
        # l_c=0
        if l_d:
            k_x1 = self.kernel(x1)
            k_x1_c = self.kernel(x1_c)
            k_x2 = self.kernel(x2)
            k_x2_c = self.kernel(x2_c)
            # 这里的向量尽可能远离
            l_d = self.hsic(k_x1, k_x1_c) + self.hsic(k_x2, k_x2_c)
        else:
            l_d = 0
        # print('l_t{},l_c:{},l_d:{}'.format(l_t,l_c,l_d))
        return l_t, self.alpha * l_c, self.beta * l_d, self.theta * l_s, output

    def hsic(self, kX, kY):
        kXY = torch.mm(kX, kY)
        n = kXY.shape[0]
        # print(kXY.shape)
        h = torch.trace(kXY) / (n * n) + torch.mean(kX) * torch.mean(kY) - 2 * torch.mean(kXY) / n
        return h * n ** 2 / (n - 1) ** 2

    def kernel(self, X):
        n = X.shape[0]
        square_x = torch.pow(X, 2)
        sum_square_x = torch.sum(square_x, 1)
        sum_mat = sum_square_x.view(n, 1) + sum_square_x.view(1, n)
        sum_mat = sum_mat - 2 * torch.mm(X, X.t())
        return sum_mat
