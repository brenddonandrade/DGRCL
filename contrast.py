import math

import numpy as np
import torch
from torch_geometric.utils import degree, to_undirected

from utils import compute_pr, eigenvector_centrality


def drop_edge_random(edge_index, p):
    drop_mask = torch.empty((edge_index.size(1),), dtype=torch.float32, device=edge_index.device).uniform_(0, 1) < p
    x = edge_index.clone()
    x[:, drop_mask] = 0
    return x


def degree_drop_weights(edge_index):
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights



def drop_edge_weighted(edge_index, edge_weights, p: float = 0.3, threshold: float = 0.7):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

    return edge_index[:, sel_mask]


def drop_feature_random(x, drop_prob):
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x



def drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w

    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[:, drop_mask] = 0.

    return x


def feature_drop_weights(x, node_c):
    x = x.to(torch.bool).to(torch.float32)
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())
    return s


def feature_drop_weights_dense(x, node_c):
    x = x.abs()
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s


def get_views(data, args, relation_mask):
    edge, feature = relation_embedding(data, args, relation_mask)

    data.edge_index = edge.to(args.device)
    data.x = feature.to(args.device)
    data.num_nodes = args.num_nodes

    # edge drop
    # random
    if args.gca_method == 'random':
        edge_index_1 = drop_edge_random(data.edge_index, args.drop_edge_rate_1)
        edge_index_2 = drop_edge_random(data.edge_index, args.drop_edge_rate_2)
        x_1 = drop_feature_random(data.x, args.drop_feature_rate_1)
        x_2 = drop_feature_random(data.x, args.drop_feature_rate_2)
    # degree
    elif args.gca_method == 'degree':
        drop_weights = degree_drop_weights(data.edge_index)
        edge_index_1 = drop_edge_weighted(data.edge_index, drop_weights, args.drop_edge_rate_1, threshold=0.7)
        edge_index_2 = drop_edge_weighted(data.edge_index, drop_weights, args.drop_edge_rate_2, threshold=0.7)
        edge_index_ = to_undirected(data.edge_index)
        node_deg = degree(edge_index_[1], data.num_nodes)
        feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg)
        x_1 = drop_feature_weighted(data.x, feature_weights, args.drop_feature_rate_1)
        x_2 = drop_feature_weighted(data.x, feature_weights, args.drop_feature_rate_2)

    contrast_views = {
        "a_1": edge_index_1.to(args.device),
        "a_2": edge_index_2.to(args.device),
        "x_1": x_1.to(args.device),
        "x_2": x_2.to(args.device)
    }
    return contrast_views

def relation_embedding(data, args, relation_mask):
    adj_matrix = torch.zeros(args.num_nodes, args.num_nodes, dtype=torch.int)
    for edge_index in data.edge_index_list:
        for i in range(edge_index.size(1)):
            src = edge_index[0, i]
            dst = edge_index[1, i]
            adj_matrix[src, dst] += 1

    embed_adj = adj_matrix.numpy()+relation_mask
    embed_adj[embed_adj != 0] = 1

    edge_indices = np.argwhere(embed_adj == 1)
    edge = torch.tensor(edge_indices.T, dtype=torch.long)

    feature = torch.mean(torch.stack(data.x_list, dim=0), dim=0)

    return edge, feature


def get_exp_a_num(n):
    i = 1
    s = 0.0
    s_list = []
    s_res = 0
    # harmonic_sum
    for i in range(1, n + 1):
        s = s + 1 / i
        s_list.append(s)
    # expectation
    for each in s_list:
        s_res = s_res + 1/each
    return (math.ceil(s_res))*2 + n