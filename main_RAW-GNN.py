# coding=utf-8
from __future__ import division
from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import *
import re
import numpy
from sklearn.metrics import f1_score
import os
import torch.nn as nn
import argparse
from config import Config
import math
import random
import time
from tqdm import tqdm
from torch_sparse import SparseTensor
import torch_cluster  # noqa
from torch_geometric.nn import GCNConv


def add_self_loop_for_all_nodes(edge_list, node_num):
    loops = torch.tensor([[i for i in range(node_num)], [i for i in range(node_num)]], dtype=torch.int).t()
    edge_list = torch.cat([edge_list, loops], dim=0)
    return edge_list

def add_self_loop_for_isolated_nodes(edge_list, node_num):
    print("calculating isolated nodes.")
    l = torch.zeros(node_num)
    l[edge_list[:, 0]] = 1
    l[edge_list[:, 1]] = 1
    idx = torch.arange(0, node_num)[l==0]
    if idx.size(0) == 0:
        print("no isolated nodes.")
    else:
        print("found {} isolated nodes.".format(idx.size(0)))
    idx = torch.stack([idx, idx], dim = 1)
    edge_list = torch.cat([edge_list, idx], dim = 0)
    return edge_list

def generate_random_walk(adj, config, p, q): 
    long_walks_per_node = config.long_walks_per_node
    long_walk_len = config.long_walk_len
    walk_len = config.walk_len # 3
    batch = torch.arange(config.n)
    batch = batch.repeat(long_walks_per_node).to(config.device)
    rowptr, col, _ = adj.csr()
    rw = torch.ops.torch_cluster.random_walk(rowptr, col, batch, long_walk_len, p, q)
    if not isinstance(rw, torch.Tensor):
        rw = rw[0]
    walks = []
    num_walks_per_rw = 1 + long_walk_len + 1 - walk_len
    for j in range(num_walks_per_rw):
        walks.append(rw[:, j:j + walk_len])
    out_walks = torch.cat(walks, dim=0)
    out_walks = out_walks[:, torch.arange(start=walk_len - 1, end=-1, step=-1)]
    return out_walks.to(config.device)

class PathAgg_att_sample_layer(nn.Module):
    def __init__(self, in_dim, out_dim, config, strategy="DFS"): 
        super(PathAgg_att_sample_layer, self).__init__()
        self.dropout = config.dropout
        # GRU LSTM
        self.rnn = nn.GRU(input_size=in_dim, hidden_size=out_dim, batch_first=True)
        self.a = nn.Parameter(torch.rand((out_dim, config.head_num), dtype=torch.float) * 2 - 1)  # attention_head
        self.head_num = config.head_num
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.strategy = strategy

    def forward(self, x, adj, config):  
        #
        if self.strategy == "DFS":
            path_list = generate_random_walk(adj, config, p=10, q=0.1)
        else:  # BFS
            path_list = generate_random_walk(adj, config, p=0.1, q=10)
        M = SparseTensor(row=path_list[:, -1], col=torch.arange(len(path_list)), value=None, sparse_sizes=(config.n, len(path_list)))
        
        path_features = F.embedding(path_list, x)
        
        _, emb = self.rnn(path_features)  
        path_emb = emb[0]  
       
        path_att_un = (path_emb @ self.a).exp()  # path_num x head_num, path attention before softmax
        path_att_un_sum_by_node = M @ path_att_un  # node_num x head_num, path attention sum on nodes
        path_att_un_low = M.t() @ (1/path_att_un_sum_by_node)  # path_num x head_num, path attention
        path_att_n = path_att_un * path_att_un_low  # path_num x head_num, path attention on node after softmax
        
        path_att_n = path_att_n.reshape(-1, 1).repeat(1, self.out_dim).reshape(config.walks_per_node * config.n, -1)  # new
        path_emb = path_emb.repeat(1, self.head_num)
        
        path_att_emb = path_att_n * path_emb  # path_num x hid_num, weighted path hidden embbeding

        node_att_emb = M @ path_att_emb  # node_num x hid_num, weighted node hidden embbeding

        del path_list  # 
        return node_att_emb

# 
class PathAgg_att_sample2(nn.Module):  
    def __init__(self, config): 
        super(PathAgg_att_sample2, self).__init__()
        self.dropout = config.dropout

        self.layer1 = PathAgg_att_sample_layer(in_dim=config.fdim, out_dim=config.nhid1, config=config, strategy="DFS")

        self.layer2 = PathAgg_att_sample_layer(in_dim=config.fdim, out_dim=config.nhid1, config=config, strategy="BFS")

        self.lin1 = torch.nn.Linear(config.nhid1 * config.head_num * 2, config.class_num)
        self.outa = nn.LogSoftmax(dim=1)

    def forward(self, x0, adj, config):
        #
        x1 = self.layer1(x0, adj, config)

        x2 = self.layer2(x0, adj, config)

        x3 = torch.cat([x1, x2], dim = 1)
        x3 = F.leaky_relu(x3)
        output = self.lin1(x3)
        output = self.outa(output)
        return output

if __name__ == "__main__":
    torch.cuda.is_available()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parse = argparse.ArgumentParser()
    parse.add_argument("-tt", "--train_times", help="times of trains, the final result will be their average",
                       default=1, type=int, required=False)
    parse.add_argument("-d", "--dataset", help="dataset", default="cornell", type=str, required=False)
    parse.add_argument("-sh", "--show_interval", default=1, type=int, required=False)
    parse.add_argument("-p", "--patience", default=40, type=int, required=False)

    args = parse.parse_args()

    splits = 10
    for dataset in [
        'cornell',
        # 'texas',
        # 'wisconsin',
        # 'cora',
        # 'citeseer', 
        # 'pubmed',  
        # 'chameleon',
        # 'film',  
        # 'squirrel'  
    ]:
        args.dataset = dataset
        config_file = "./config/" + str(args.dataset) + ".ini"
        config = Config(config_file)
        config.epochs = 200  # 200
        config.dropout = 0.5
        config.lr = 0.05  #
        config.no_cuda = True
        config.nhid1 = 32 # 32
        config.nhid2 = 16 # 16
        config.p = 10  # DFS
        config.q = 0.1
        config.weight_decay = 0.001  # 0.001
        config.head_num = 2
        config.long_walks_per_node = 2  # 2
        config.long_walk_len = 4  # 4
        config.walk_len = 3  # 3
        config.walks_per_node = (1 + config.long_walk_len + 1 - config.walk_len) * config.long_walks_per_node

        config.device = 'cuda' if (not config.no_cuda and torch.cuda.is_available()) else 'cpu'


        print("dataset:", args.dataset)
        print("walks_per_node: {}".format(config.walks_per_node))
        for key, val in config.__dict__.items():
            print("{} : {}".format(key, val))


        edge_list = load_graph_edgelist_no_loop(config)
        edge_list = add_self_loop_for_all_nodes(edge_list, config.n)
        row, col = edge_list.t()
        adj = SparseTensor(row=row, col=col, sparse_sizes=(config.n, config.n)).to(config.device)
        features, labels = load_feature_and_lable(config)
        features = features.to(config.device)
        labels = labels.to(config.device)

        res_accs = []
        att_lists_mean = [] 
        att_lists_std = []
        att_list_std = []
        att_list_mean = []
        att_cur_mean = []
        att_cur_std = []
        for ti in range(splits):
            print(f"split No.{ti}:")
            idx_train, idx_val, idx_test = load_idx(config, ti)
            idx_train, idx_val, idx_test = idx_train.to(config.device), idx_val.to(config.device), idx_test.to(config.device)

            acc_val_maxs, acc_test_maxs, f1_maxs, seeds = [], [], [], []  # 
            train_times = args.train_times
            use_seed = not config.no_seed  # 
            if use_seed:
             
                random.seed(config.seed)
                seeds.append(config.seed)
                for i in range(1,args.train_times):
                    seeds.append(random.randint(1,200))

            for num_of_train in range(train_times):  

                cur_seed = seeds[num_of_train]
                if use_seed:
                    np.random.seed(cur_seed)
                    torch.manual_seed(cur_seed)
                    if config.device == 'cuda':
                        torch.cuda.manual_seed(cur_seed)

                model = PathAgg_att_sample2(config).to(config.device)

                optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

                def train(model, epochs):
                    model.train()
                    optimizer.zero_grad()
                    output = model(features, adj, config)
                    loss = F.nll_loss(output[idx_train], labels[idx_train])
                    acc = accuracy(output[idx_train], labels[idx_train])
                    loss.backward()
                    optimizer.step()
                    acc_val, macro_f1  = main_test(model, idx_val)
                    acc_test, _1 = main_test(model, idx_test)
                    att_list_mean.append(att_cur_mean)
                    att_list_std.append(att_cur_std)
                    if epoch>0 and epoch%args.show_interval==0:
                        print('epoch:{:2d}'.format(epochs),
                          'loss_train:{:.4f}'.format(loss.item()),
                          'acc_train:{:.4f}'.format(acc.item()),
                          'acc_val:{:.4f}'.format(acc_val.item()),
                          'acc_test:{:.4f}'.format(acc_test.item()),
                          )
                    return loss.item(), acc_val.item(), acc_test.item(), macro_f1.item()

                def main_test(model, idxs):
                    model.eval()
                    output = model(features, adj, config)
                    acc = accuracy(output[idxs], labels[idxs])
                    label_max = []
                    for idx in idxs:
                        label_max.append(torch.argmax(output[idx]).item())
                    labelcpu = labels[idxs].data.cpu()
                    macro_f1 = f1_score(labelcpu, label_max, average='macro')
                    return acc, macro_f1

                acc_val_max = 0
                acc_test_max = 0
                f1_max = 0
                epoch_max = 0
                for epoch in range(config.epochs):
                    loss, acc_val, acc_test, macro_f1 = train(model, epoch)
                    if acc_val >= acc_val_max:
                        acc_val_max = acc_val
                        acc_test_max = acc_test
                        f1_max = macro_f1
                        epoch_max = epoch
                    if epoch_max + args.patience < epoch:
                        print("early stopping at epoch {}".format(epoch))
                        break  # patience
                print('seed:{:3d}'.format(seeds[num_of_train]),
                    'num of train:{:3d}'.format(num_of_train),
                    'epoch_max:{:3d}'.format(epoch_max),
                    'acc_val_max: {:.4f}'.format(acc_val_max),
                    'acc_test_max: {:.4f}'.format(acc_test_max),
                    'f1_max: {:.4f}'.format(f1_max),
                      )
                att_lists_mean.append(att_list_mean[epoch_max])
                att_lists_std.append((att_list_std[epoch]))

                acc_val_maxs.append(acc_val_max)
                acc_test_maxs.append(acc_test_max)
                f1_maxs.append(f1_max)
            # print("seeds: ", seeds)
            print("acc_val:", acc_val_maxs)
            print("acc_test_maxs:", acc_test_maxs)
            res_acc = acc_test_maxs[np.argmax(np.array(acc_val_maxs))]
            print("result_acc:{:.2f}%".format(100 * res_acc))
            res_accs.append(res_acc)
        print(f"\n{args.dataset} result_accs: ")
        for acc in res_accs:
            print("{:.2f}".format(100 * acc), end=', ')
        print("\nmeam:{:.2f}".format(100 * np.mean(res_accs)), "±{:.2f}".format(100 * np.std(res_accs)))
        print("-"*100)


    
    
