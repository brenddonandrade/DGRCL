import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self,args,out_features=2, in_features = None):
        super(Classifier,self).__init__()
        activation = torch.nn.ReLU()
        num_feats = in_features
        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_features = num_feats,
                                                       out_features =args.gcn_parameters['cls_feats']),
                                       activation,
                                       torch.nn.Linear(in_features = args.gcn_parameters['cls_feats'],
                                                       out_features = out_features))

    def forward(self,x):
        return self.mlp(x)
