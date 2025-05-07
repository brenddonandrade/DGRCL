import math
import re
import os
import numpy as np
import torch
import torch.nn as nn
from dtaidistance import dtw
from fastdtw import fastdtw
from numpy.fft import fft
from tqdm import tqdm
from scipy.spatial.distance import cosine
from pathlib import Path
from joblib import Parallel, delayed
from scipy.stats import zscore


def get_stock_list(directory):
    nasdaq_list = []
    nyse_list = []

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        nasdaq = re.compile('NASDAQ')
        nyse = re.compile('NYSE')
        if os.path.isfile(f):
            if nasdaq.match(filename):
                nasdaq_list.append(f)
            if nyse.match(filename):
                nyse_list.append(f)

    return nasdaq_list, nyse_list


def get_clean_volume(data):
    # if the data is not valid, use the closet one
    single_EOD = data
    volumes = []
    for i in range(window_size, len(single_EOD)):
        if single_EOD[i - window_size, 5] == -1234:
            volume = [x for x in single_EOD[i - window_size:i + 1:, 5] if x != -1234]
            volume = np.array(volume)
            volume[np.isnan(volume)] = 0
            volume[np.isinf(volume)] = 0
            volumes.append(volume)
        else:
            volume = single_EOD[i - window_size:i + 1:, 5]
            volume[np.isnan(volume)] = 0
            volume[np.isinf(volume)] = 0
            volumes.append(volume)
    return volumes

def save_graph_a_step_magno(market_list, window_size=20, normalize=False, n_jobs=-1):
    all_volumes = []

    # 1. Load and process volume data
    for filepath in market_list:
        stock_name = re.split(r"[\\/]", filepath)[-1].split(".")[0]
        single_EOD = np.genfromtxt(filepath, dtype=np.float32, delimiter=',', skip_header=False)

        # Reindex and clean
        single_EOD = reindex(single_EOD)
        volume_windows = get_clean_volume(single_EOD)
        all_volumes.append([stock_name, volume_windows])

    n_stocks = len(all_volumes)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 2. Time step loop
    for time_step in tqdm(range(window_size + 1, len(single_EOD) - window_size)):
        adj_matrix = np.zeros((n_stocks, n_stocks))

        # Cache and normalize volume windows
        windows = []
        for _, vol_series in all_volumes:
            x = vol_series[time_step]
            if normalize:
                x = zscore(x)
            windows.append(x)

        # 3. Parallel DTW computation
        def compute_dtw_pair(i, j):
            a, b = windows[i], windows[j]
            if a.size > 0 and b.size > 0:
                dist = dtw.distance_fast(a, b)
                return (i, j, dist)
            return (i, j, 0.0)

        pairs = [(i, j) for i in range(n_stocks) for j in range(i + 1, n_stocks)]

        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_dtw_pair)(i, j) for i, j in pairs
        )

        for i, j, dist in results:
            adj_matrix[i, j] = dist
            adj_matrix[j, i] = dist  # symmetry

        # 4. Save
        filename = f"{time_step}({time_step - window_size - 1})_a_matrix.txt"
        np.savetxt(Path(save_dir) / filename, adj_matrix, fmt='%.4f', delimiter='\t')

def save_graph_a_step(market_list):
    all_volumes = []
    for each in market_list:
        stock_name = re.split(r"\\", each)[-1].split(".")[0]
        single_EOD = np.genfromtxt(each, dtype=np.float32, delimiter=',', skip_header=False)

        # reindex
        single_EOD = reindex(single_EOD)

        each_volumes = get_clean_volume(single_EOD)
        all_volumes.append([stock_name, each_volumes])
    for time_steps in tqdm(range(window_size + 1, len(single_EOD) - window_size)):
    # for time_steps in tqdm(range(779, len(single_EOD) - window_size)):
        # dtw for same stock will be zero
        # adj_matrix = [[0 for _ in range(len(all_volumes))] for _ in range(len(all_volumes))]
        adj_matrix = np.zeros((len(all_volumes), len(all_volumes)))
        for a_stock in range(len(all_volumes)):
            # stock_name = all_volumes[stocks][0]
            # stock_volume = all_volumes[stocks][1][time_steps]
            for b_stock in range(a_stock + 1, len(all_volumes)):
                a = all_volumes[a_stock][1][time_steps]
                b = all_volumes[b_stock][1][time_steps]
                # dtw - python
                if len(a) and len(b) != 0:
                    # alignment_distance = dtw.distance(a, b)
                    alignment_distance =  dtw.distance_fast(a, b)
                    adj_matrix[a_stock][b_stock] = int(alignment_distance)
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.savetxt(save_dir + str(time_steps) +'('+ str(time_steps-window_size-1) + ')_a_matrix.txt', adj_matrix, fmt='%d', delimiter='\t')


def get_graph_label_and_feature(market_list):
    lab_all = []
    feature_all = []
    for each in tqdm(market_list, desc="Processing get_graph_label_and_feature"):
        single_EOD = np.genfromtxt(each, dtype=np.float32, delimiter=',', skip_header=False)

        # reindex
        single_EOD = reindex(single_EOD)

        single_close = single_EOD[:, 4]
        single_feature = single_EOD[:, 1:6]

        clean_close = fill_missing_values_mid(single_close)
        clean_features = [fill_missing_values_mid(single_feature[i, :]) for i in range(single_feature.shape[0])]
        clean_feature = np.stack(clean_features)

        label_list = np.array([])
        feature_list = []
        for i in range(2*window_size, len(single_EOD) - 1):
            if clean_close[i + 1] > clean_close[i]:
                label = 1
            elif clean_close[i + 1] < clean_close[i]:
                label = -1
            else:
                label = 0
            label_list = np.append(label_list, np.array(label))
            feature_data = clean_feature[i - window_size:i + 1]

            # get best feature
            # feature_data, _ = get_best_features(feature_data, loss_type, div_step)

            feature_list.append(torch.tensor(feature_data))

        stacked_feature = torch.stack(feature_list, dim=0)

        lab_all.append(torch.tensor(label_list))
        feature_all.append(stacked_feature)

    return lab_all, feature_all


def save_label_feature_time_step(label, feature, time_steps):
    for each_step in range(time_steps - (window_size + 1)):
        step_label = []
        step_feature = []
        for each in range(len(label)):
            each_label = label[each][each_step]
            each_feature = feature[each][each_step]
            step_label.append(each_label)
            step_feature.append(each_feature)

        stacked_label = np.stack([tensor.numpy() for tensor in step_label], axis=0)
        stacked_feature = np.stack([tensor.numpy() for tensor in step_feature], axis=0)

        np.save(save_dir+str(each_step)+'_label.npy', stacked_label)
        np.save(save_dir+str(each_step)+'_feature.npy', stacked_feature)
        print(each_step)


def reindex(data):
    mask_EOD = data[mask[0]:mask[-1], ]
    new_index = np.arange(0, mask_EOD.shape[0]).reshape(-1, 1)
    result = np.hstack((new_index, mask_EOD))
    single_EOD = np.delete(result, 1, axis=1)
    return single_EOD


def get_best_features(data, l_type, step_len):
    features_count = data.shape[1]
    features_lenth = data.shape[0]

    all_feature_list = []
    all_feature_index = []
    # separate lenth of 20 to 11 lenth of 10
    for each_col in range(features_count):
        col = data[:, each_col]
        each_feature_list = []
        for each_div in range(features_lenth - step_len+1):
            each = col[each_div:each_div + step_len, ]
            each_feature_list.append(each)
        each_best, each_index = cal_pair_cos(each_feature_list)
        all_feature_list.append(np.concatenate((each_best[0], each_best[1])))
        all_feature_index.append(each_index)

    return np.column_stack(all_feature_list), all_feature_index


def fill_missing_values_mid(data):
    # Find the indices of non-NaN values
    valid_indices = np.where(~np.isnan(data) & (data != -1234) & ~np.isinf(data))[0]

    # Use numpy.interp for linear interpolation
    interpolated_data = np.interp(range(len(data)), valid_indices, data[valid_indices])

    if len(valid_indices) == 0:
        # If there are no valid values, return the original array
        return data

    return interpolated_data


# cutting off both ends from valid start and last
def get_valid_day_mask(stock_lists):
    first_mask = []
    last_mask = []
    for each_list in stock_lists:
        for each in each_list:
            single_EOD = np.genfromtxt(each, dtype=np.float32, delimiter=',', skip_header=False)
            data = single_EOD[:, 1]
            valid_indices = np.where(np.isfinite(data) & (data != -1234) & (data != 0))
            first = valid_indices[0][0]
            last = valid_indices[0][-1]
            first_mask.append(first)
            last_mask.append(last)
    return [max(first_mask), min(last_mask)], min(last_mask) - max(first_mask) + 1


def cal_pair_cos(data):
    cosine_similarity = nn.CosineSimilarity(dim=0)
    largest_value = float('-inf')  # Start with negative infinity as a placeholder
    largest_index = -1
    # trans_loss = TransferLoss(l_type)
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            loss = cosine_similarity(torch.tensor(fft(data[i]).real), torch.tensor(fft(data[j]).real))
            if loss > largest_value:
                # the best pair should be in different direction
                if data[i][0] < data[i][-1] and data[j][0] > data[j][-1]:
                    largest_value = loss
                    best_index = [i, j]
                elif data[i][0] > data[i][-1] and data[j][0] < data[j][-1]:
                    largest_value = loss
                    best_index = [i, j]
                else:
                    largest_value = loss
                    best_index = [0, len(data) // 2]
            # print(i, j, loss)
    # print('largest_value and largest_index:', largest_value, best_index)
    # best_result = [fft(data[best_index[0]]).real, fft(data[best_index[1]]).real]
    best_pair = [data[best_index[0]], data[best_index[1]]]
    return best_pair, best_index

# exp_a_num = exp*2 + self
def get_exp_a_num(n):
    return get_exp(n)*2 + n


def get_exp(n):
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
    return math.ceil(s_res)

# for i in [2169, 2736]:
#     print(i, get_exp(i))


def gen_clean_a(steps):
    threshold_list = []
    for time_steps in range(0, steps):
        # adj_matrix = np.loadtxt(data_path+'\\'+str((window_size+1)+time_steps)+'('+str(time_steps)+')_a_matrix.txt', dtype=int)
        adj_path='/mnt/c/Users/carlo/Desktop/importante/Desktop/Versatus/DGRCL/preprocessing/backup/nasdaq_sector_industry.txt'
        adj_matrix = np.loadtxt(adj_path, dtype=int)
        eye = np.eye(adj_matrix.shape[0], dtype=int)
        exp_edge = get_exp_a_num(adj_matrix.shape[0])
        for value_threshold in range(1, np.max(adj_matrix)):
            edge_num = (np.sum((adj_matrix > 0) & (adj_matrix < value_threshold))*2 + adj_matrix.shape[0])
            if edge_num > exp_edge:
                threshold_list.append(value_threshold)
                mask = (adj_matrix > 0) & (adj_matrix < value_threshold)
                adj_matrix_res = np.where(mask, 1, 0)
                adj_matrix_res = adj_matrix_res + adj_matrix_res.T + eye
                np.savetxt(out_path+str(time_steps)+'_clean_a.txt', adj_matrix_res, fmt='%d', delimiter='\t')
                break


# def load_graph_a(path):
#     graph_a = []
#     for each_a in range(0, steps):
#         single_a = np.genfromtxt(path + '/' + str(each_a) + '_clean_a.txt', dtype=int, delimiter='\t')
#         graph_a.append(single_a)
#     print(0)

if __name__ == '__main__':

    # use for gen_feat
    data_dir = r'/mnt/c/Users/carlo/Desktop/importante/Desktop/Versatus/DGRCL/data/raw_data/2013-01-01-2016-12-30'

    # use for gen_a
    # data_dir = r'/data/raw_data/2013010120161230'

    market = 'nyse'
    data_path = r'../data/' + market
    out_path = data_path + '_clean_a/'
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    

    # my_array = np.random.rand(20, 5)
    loss_type = 'cosine'
    div_step = 10
    # result = get_best_features(my_array, loss_type, div_step)

    window_size = 19
    nasdaq_list, nyse_list = get_stock_list(data_dir)

    mask, time_range = get_valid_day_mask([nasdaq_list, nyse_list])
    

    # save_dir = 'nyse/'
    # save_dir = 'nyse_f_l/'
    # save_dir = 'nyse_f_l_b/'

    # save_dir = 'nasdaq/'
    save_dir = 'nasdaq_f_l/'
    # save_dir = 'nasdaq_f_l_b/'

    # stock_list = nyse_list
    stock_list = nasdaq_list

    # A
    # save_graph_a_step_magno(stock_list, window_size=20, normalize=False, n_jobs=-1)
    save_graph_a_step(stock_list)
    # gen_clean_a(time_range - 2*(window_size+1))

    # X, Y
    # labels, features = get_graph_label_and_feature(stock_list)
    # save_label_feature_time_step(labels, features, time_range - (window_size + 1))

