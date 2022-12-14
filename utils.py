# coding=utf-8
import numpy as np
import scipy.sparse as sp
import torch
import sys
import re
import pickle as pkl
import numpy as np
import networkx as nx
import os
import torch_geometric.utils.undirected as undirected

def common_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cost = torch.mean((cov1 - cov2)**2)
    # cost = -torch.cosine_similarity(emb1, emb2, dim=1).mean()
    return cost

def loss_dependence(emb1, emb2, dim):
    R = torch.eye(dim).cuda() - (1/dim) * torch.ones(dim, dim).cuda()
    # R = torch.eye(dim) - (1 / dim) * torch.ones(dim, dim)
    K1 = torch.mm(emb1, emb1.t())
    K2 = torch.mm(emb2, emb2.t())
    RK1 = torch.mm(R, K1)
    RK2 = torch.mm(R, K2)
    HSIC = torch.trace(torch.mm(RK1, RK2))
    return HSIC



def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def load_idx(config, time):
    train_p = os.path.join(config.root_path, str(time)+'train.txt')
    val_p = os.path.join(config.root_path, str(time)+'val.txt')
    test_p = os.path.join(config.root_path, str(time)+'test.txt')
    train = np.loadtxt(train_p, dtype=int)
    val = np.loadtxt(val_p, dtype=int)  # new
    test = np.loadtxt(test_p, dtype=int)

    idx_train = train.tolist()
    idx_val = val.tolist()
    idx_test = test.tolist()

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    return idx_train, idx_val, idx_test

def load_feature_and_lable(config):
    f = np.loadtxt(config.feature_path, dtype = float)
    l = np.loadtxt(config.label_path, dtype = int)
    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))

    label = torch.LongTensor(np.array(l))
    return features, label

def load_data(config):
    f = np.loadtxt(config.feature_path, dtype = float)
    l = np.loadtxt(config.label_path, dtype = int)
    test = np.loadtxt(config.test_path, dtype=int)
    val = np.loadtxt(config.val_path, dtype=int)  # new
    train = np.loadtxt(config.train_path, dtype=int)

    idx_train = train.tolist()
    idx_val = val.tolist()
    idx_test = test.tolist()

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    features = sp.csr_matrix(f, dtype=np.float32)
    features = torch.FloatTensor(np.array(features.todense()))

    label = torch.LongTensor(np.array(l))

    return features, label, idx_train, idx_val, idx_test

def load_graph_sadj(config):
    struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    nsadj = normalize(sadj+sp.eye(sadj.shape[0]))

    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)

    return nsadj, None


def load_graph(config):
    featuregraph_path = config.featuregraph_path + str(config.k) + '.txt'
    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape)
    fadj = sp.coo_matrix((np.ones(fedges.shape[0]), (fedges[:, 0], fedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    fadj = fadj + fadj.T.multiply(fadj.T > fadj) - fadj.multiply(fadj.T > fadj)
    nfadj = normalize(fadj + sp.eye(fadj.shape[0]))

    struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(config.n, config.n), dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    nsadj = normalize(sadj+sp.eye(sadj.shape[0]))

    nsadj = sparse_mx_to_torch_sparse_tensor(nsadj)
    nfadj = sparse_mx_to_torch_sparse_tensor(nfadj)

    return nsadj, nfadj

def load_graph_edge_form(config):
    featuregraph_path = config.featuregraph_path + str(config.k) + '.txt'
    feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
    fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape).transpose()
    fedges2 = torch.tensor(fedges, dtype=torch.int64)
    fedges2 = undirected.to_undirected(fedges2)

    struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape).transpose()
    sedges2 = torch.tensor(sedges, dtype=torch.int64)
    sedges2 = undirected.to_undirected(sedges2)
    return sedges2, fedges2

def load_graph_edgelist_no_loop(config):
    struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape).transpose()
    sedges2 = torch.tensor(sedges, dtype=torch.int64) 

    # # start remove self-loop
    temp, c = sedges2.tolist(), 0
    for i in range(len(temp[0])):
        if temp[0][i] != temp[1][i]:
            temp[0][c] = temp[0][i]
            temp[1][c] = temp[1][i]
            c += 1
    temp[0] = temp[0][:c]
    temp[1] = temp[1][:c]
    sedges2 = torch.tensor(temp, dtype=torch.int64)
    
    sedges2 = undirected.to_undirected(sedges2)
    return sedges2.t()


def load_graph_edge_form_knn(config):  #
    #
    fadjs = []
    for k in range(config.Kf_strat, config.Kf_strat+config.Kf):
        featuregraph_path = config.featuregraph_path + str(config.k) + '.txt'
        feature_edges = np.genfromtxt(featuregraph_path, dtype=np.int32)
        fedges = np.array(list(feature_edges), dtype=np.int32).reshape(feature_edges.shape).transpose()
        fadj = torch.tensor(fedges, dtype=torch.int64)
        fadj = undirected.to_undirected(fadj)  #
        fadjs.append(fadj)  #

    struct_edges = np.genfromtxt(config.structgraph_path, dtype=np.int32)
    sedges = np.array(list(struct_edges), dtype=np.int32).reshape(struct_edges.shape).transpose()
    sadj = torch.tensor(sedges, dtype=torch.int64)
    sadj = undirected.to_undirected(sadj)

    return sadj, fadjs

