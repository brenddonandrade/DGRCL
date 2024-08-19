import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


class splitter():
    '''
    creates 3 splits
    train
    dev
    test
    '''
    def __init__(self,args,tasker,r_mask):

        #only the training one requires special handling on start, the others are fine with the split IDX.
        start = tasker.data.min_time + args.num_hist_steps #-1 + args.adj_mat_time_window
        end = args.train_proportion
        # end = 941*0.65 = 611
        end = int(np.floor(tasker.data.max_time.type(torch.float) * end))
        train = data_split(r_mask, tasker, start, end, test = False)

        index_column = train.tasker.data.nodes_feats[:, 0:1]  # Keeping the index column as a 2D array for easy concatenation later
        data_columns = train.tasker.data.nodes_feats[:, 1:]  # Last 100 columns
        # normalization
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_columns)
        train.tasker.data.nodes_feats = np.concatenate([index_column, scaled_data], axis=1)
        train.tasker.data.nodes_feats = torch.tensor(train.tasker.data.nodes_feats, dtype=torch.float32)
        train = DataLoader(train, **args.data_loading_params)

        start = end
        # end = 941*(0.65+0.1) = 705
        end = args.dev_proportion + args.train_proportion
        end = int(np.floor(tasker.data.max_time.type(torch.float) * end))
        dev = data_split(r_mask, tasker, start, end, test=True)

        index_column = dev.tasker.data.nodes_feats[:, 0:1]
        data_columns = dev.tasker.data.nodes_feats[:, 1:]
        scaled_data = scaler.transform(data_columns)
        dev.tasker.data.nodes_feats = np.concatenate([index_column, scaled_data], axis=1)
        dev.tasker.data.nodes_feats = torch.tensor(dev.tasker.data.nodes_feats, dtype=torch.float32)
        dev = DataLoader(dev, num_workers=args.data_loading_params['num_workers'])

        start = end


        end = int(tasker.max_time) + 1
        test = data_split(r_mask, tasker, start, end, test=True)

        index_column = test.tasker.data.nodes_feats[:, 0:1]
        data_columns = test.tasker.data.nodes_feats[:, 1:]
        scaled_data = scaler.transform(data_columns)
        test.tasker.data.nodes_feats = np.concatenate([index_column, scaled_data], axis=1)
        test.tasker.data.nodes_feats = torch.tensor(test.tasker.data.nodes_feats, dtype=torch.float32)
        test = DataLoader(test, num_workers=args.data_loading_params['num_workers'])

        # print ('Dataset splits sizes:  train',len(train), 'dev',len(dev), 'test',len(test))

        self.tasker = tasker
        self.train = train
        self.dev = dev
        self.test = test
        


class data_split(Dataset):
    def __init__(self, relation_mask, tasker, start, end, test, **kwargs):
        '''
        start and end are indices indicating what items belong to this split
        '''
        self.r_mask = relation_mask
        self.tasker = tasker
        self.start = start
        self.end = end
        self.test = test
        self.kwargs = kwargs

    def __len__(self):
        return self.end-self.start

    def __getitem__(self,idx):
        idx = self.start + idx
        t = self.tasker.get_sample(idx, self.r_mask, test = self.test, **self.kwargs)
        return t