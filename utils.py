import argparse
import yaml
import torch
import numpy as np
import time
import random
import math
import networkx as nx
from torch_scatter import scatter
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SGConv, SAGEConv, GATConv, GraphConv, GINConv
from torch_geometric.utils import degree, to_networkx
import torch.nn as nn
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef



def get_activation(name: str):
    activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': torch.nn.PReLU(),
        'rrelu': F.rrelu
    }

    return activations[name]


def get_base_model(name: str):
    # GATConv in base_models
    def gat_wrapper(in_channels, out_channels):
        return GATConv(
            in_channels=in_channels,
            out_channels=out_channels // 4,
            heads=4
        )

    # GINConv in base_models
    def gin_wrapper(in_channels, out_channels):
        mlp = nn.Sequential(
            nn.Linear(in_channels, 2 * out_channels),
            nn.ELU(),
            nn.Linear(2 * out_channels, out_channels)
        )
        return GINConv(mlp)

    base_models = {
        'GCNConv': GCNConv,
        'SGConv': SGConv,
        'SAGEConv': SAGEConv,
        'GATConv': gat_wrapper,
        'GraphConv': GraphConv,
        'GINConv': gin_wrapper
    }

    return base_models[name]


def pad_with_last_col(matrix,cols):
    out = [matrix]
    pad = [matrix[:,[-1]]] * (cols - matrix.size(1))
    out.extend(pad)
    return torch.cat(out,dim=1)

def pad_with_last_val(vect,k):
    device = 'cuda' if vect.is_cuda else 'cpu'
    pad = torch.ones(k - vect.size(0),
                         dtype=torch.long,
                         device = device) * vect[-1]
    vect = torch.cat([vect,pad])
    return vect



def sparse_prepare_tensor(tensor,torch_size, ignore_batch_dim = True):
    if ignore_batch_dim:
        tensor = sp_ignore_batch_dim(tensor)
    tensor = make_sparse_tensor(tensor,
                                tensor_type = 'float',
                                torch_size = torch_size)
    return tensor

def sp_ignore_batch_dim(tensor_dict):
    tensor_dict['idx'] = tensor_dict['idx'][0]
    tensor_dict['vals'] = tensor_dict['vals'][0]
    return tensor_dict

def aggregate_by_time(time_vector,time_win_aggr):
        time_vector = time_vector - time_vector.min()
        time_vector = time_vector // time_win_aggr
        return time_vector

def sort_by_time(data,time_col):
        _, sort = torch.sort(data[:,time_col])
        data = data[sort]
        return data

def print_sp_tensor(sp_tensor,size):
    print(torch.sparse.FloatTensor(sp_tensor['idx'].t(),sp_tensor['vals'],torch.Size([size,size])).to_dense())

def reset_param(t):
    stdv = 2. / math.sqrt(t.size(0))
    t.data.uniform_(-stdv,stdv)

def make_sparse_tensor(adj,tensor_type,torch_size):
    if len(torch_size) == 2:
        tensor_size = torch.Size(torch_size)
    elif len(torch_size) == 1:
        tensor_size = torch.Size(torch_size*2)

    if tensor_type == 'float':
        test = torch.sparse.FloatTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.float),
                                      tensor_size)
        return torch.sparse.FloatTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.float),
                                      tensor_size)
    elif tensor_type == 'long':
        return torch.sparse.LongTensor(adj['idx'].t(),
                                      adj['vals'].type(torch.long),
                                      tensor_size)
    else:
        raise NotImplementedError('only make floats or long sparse tensors')

def sp_to_dict(sp_tensor):
    return  {'idx': sp_tensor._indices().t(),
             'vals': sp_tensor._values()}

class Namespace(object):
    '''
    helps referencing object in a dictionary as dict.key instead of dict['key']
    '''
    def __init__(self, adict):
        self.__dict__.update(adict)

def set_seeds(rank):
    seed = int(time.time())+rank
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_param_value(param, param_min, param_max, type='int'):
    if str(param) is None or str(param).lower()=='none':
        if type=='int':
            return random.randrange(param_min, param_max+1)
        elif type=='logscale':
            interval=np.logspace(np.log10(param_min), np.log10(param_max), num=100)
            return np.random.choice(interval,1)[0]
        else:
            return random.uniform(param_min, param_max)
    else:
        return param

def load_data(file):
    with open(file) as file:
        file = file.read().splitlines()
    data = torch.tensor([[float(r) for r in row.split(',')] for row in file[1:]])
    return data

def load_data_from_tar(file, tar_archive, replace_unknow=False, starting_line=1, sep=',', type_fn = float, tensor_const = torch.DoubleTensor):
    f = tar_archive.extractfile(file)
    lines = f.read()#
    lines=lines.decode('utf-8')
    if replace_unknow:
        lines=lines.replace('unknow', '-1')
        lines=lines.replace('-1n', '-1')

    lines=lines.splitlines()

    data = [[type_fn(r) for r in row.split(sep)] for row in lines[starting_line:]]
    data = tensor_const(data)
    #print (file,'data size', data.size())
    return data

def create_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--config_file',default='experiments/parameters_dgrcl.yaml', type=argparse.FileType(mode='r'), help='optional, yaml file containing parameters to be used, overrides command line parameters')
    return parser

def parse_args(parser):
    args = parser.parse_args()
    if args.config_file:
        data = yaml.safe_load(args.config_file)
        delattr(args, 'config_file')
        arg_dict = args.__dict__
        for key, value in data.items():
            arg_dict[key] = value
    return args


def compute_pr(num_nodes, edge_index, damp: float = 0.85, k: int = 10):
    deg_out = degree(edge_index[0], num_nodes)
    x = torch.ones((num_nodes, )).to(edge_index.device).to(torch.float32)
    # Convergence usually takes about 10 iterations for K
    for i in range(k):
        # Normalize the PageRank scores by the out-degree
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]

        # Aggregate messages by summing them up for each target node
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum', dim_size=num_nodes)

        # Update PageRank scores with the damping factor
        x = (1 - damp) * x + damp * agg_msg

    return x


def eigenvector_centrality(data):
    data_geometric = Data(edge_index=data.edge_index, num_nodes=data.num_nodes)
    graph = to_networkx(data_geometric, to_undirected=True)
    # Compute eigenvector centrality using NetworkX
    centrality = nx.eigenvector_centrality_numpy(graph)

    # Extract centrality scores in the order of node indices
    centrality_scores = [centrality[i] for i in range(data.num_nodes)]

    centrality_tensor = torch.tensor(centrality_scores, dtype=torch.float32).to(data.edge_index.device)

    return centrality_tensor


def eva_metrix(preds, labels):
    # probs = torch.softmax(predictions, dim=1)[:, 1]
    pred_s = torch.softmax(preds, dim=1).argmax(dim=1)
    # pred_s = preds.argmax(dim=3).view(labels.shape)

    pred_s = pred_s.view(-1).cpu().numpy()
    label_s = labels.view(-1).cpu().numpy()

    print('pred_s == 0', (pred_s == 0).sum().item())
    print('pred_s == 1', (pred_s == 1).sum().item())
    print('label_s == 0', (label_s == 0).sum().item())
    print('label_s == 1', (label_s == 1).sum().item())
    if len(np.unique(pred_s)) < 2 or len(np.unique(label_s)) < 2:
        roc_auc = 0
    else:
        roc_auc = roc_auc_score(pred_s, label_s)

    acc = accuracy_score(pred_s, label_s)
    prec = precision_score(pred_s, label_s, average='micro')
    rec = recall_score(pred_s, label_s, zero_division=0)
    f1 = f1_score(pred_s, label_s)
    mcc = matthews_corrcoef(pred_s, label_s)

    return torch.tensor(acc), torch.tensor(prec), torch.tensor(rec), torch.tensor(f1), torch.tensor(roc_auc), torch.tensor(mcc)