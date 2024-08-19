import utils as u
import os
import torch
import numpy as np
import re
from sklearn.preprocessing import StandardScaler


def extract_number(file_name):
	return int(re.search(r'\d+', file_name).group())


class Stock_Temporal_Dataset():
	def __init__(self,args):
		self.data_test = args.data_test
		self.args = args
		stock_name = args.stock_name

		if args.ablation_best_feature:
			feature_use = 'f_l_b'
			fl_path = r'data_source' + stock_name + '_' + feature_use
			a_path = r'data_source' + stock_name + '_clean_a'
		else:
			feature_use = 'f_l'
			fl_path = r'data_source' + stock_name + '_' + feature_use
			a_path = r'data_source' + stock_name + '_clean_a'

		self.nodes_labels_times = self.load_node_labels(fl_path)

		self.edges = self.load_transactions(a_path)

		self.nodes, self.nodes_feats = self.load_node_feats(fl_path)

	def load_node_feats(self, path):
		keyword = 'feature'
		file_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and keyword in f]
		sorted_files = sorted(file_names, key=extract_number)
		if self.data_test:
			sorted_files = sorted_files[0:100]

		nodes_feature_times = []
		for i in range(len(sorted_files)):
			loaded_array = np.load(path + '/' + sorted_files[i])

			to_tensor = torch.from_numpy(loaded_array)
			for nid in range(to_tensor.shape[0]):
				each_node_feature = to_tensor[nid].flatten()
				all_in_one = torch.cat((torch.tensor(nid).view(1), torch.tensor(i).view(1), each_node_feature))
				nodes_feature_times.append(all_in_one)

			# print('load_node_feats', i)

		nodes_feature_time = torch.stack(nodes_feature_times)

		nodes = nodes_feature_time

		nodes_feats = nodes[:,1:]

		self.num_nodes = self.args.num_nodes
		self.feats_per_node = nodes_feature_time.size(1) - 2

		return nodes, nodes_feats.float()


	def load_node_labels(self, path):
		# return (nodes, labels, time step)
		keyword = 'label'
		file_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)) and keyword in f]
		sorted_files = sorted(file_names, key=extract_number)
		if self.data_test:
			sorted_files = sorted_files[0:100]
		nodes_labels_times = []
		for i in range(len(sorted_files)):
			loaded_array = np.load(path+'/'+sorted_files[i])
			count_label_1 = np.count_nonzero(loaded_array == 1)
			count_label_m1 = np.count_nonzero(loaded_array == -1)

			to_tensor = torch.from_numpy(loaded_array)

			# Make Balance between labels of 1 and -1 using the count for labels of 1,0,-1
			if count_label_1 > count_label_m1:
				label_tensor = torch.where(to_tensor == -1, 0, to_tensor)
				# a = a+1
			else:
				label_tensor = torch.where(to_tensor == 0, 1, to_tensor)
				label_tensor = torch.where(to_tensor == -1, 0, label_tensor)
				# b = b+1

			for nid in range(label_tensor.size(0)):
				nodes_labels_times.append([nid, label_tensor[nid], i])

			# print('load_node_labels', i)

		nodes_labels_times = torch.tensor(nodes_labels_times)
		nodes_labels_times = nodes_labels_times.to(torch.int64)
		return nodes_labels_times


	def load_transactions(self, path):
		file_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
		sorted_files = sorted(file_names, key=extract_number)

		if self.data_test:
			sorted_files = sorted_files[0:100]
		# Initialize an empty list to store tensor pairs
		all_tensor_pairs = []

		print('start loading edge pairs')
		for i in range(len(sorted_files)):
			loaded_array = np.genfromtxt(path + '/' + sorted_files[i], dtype=int, delimiter='\t')

			# Find the indices of non-zero elements using np.nonzero
			nonzero_indices = np.nonzero(loaded_array)

			# Filter out pairs where x equals y
			valid_indices_mask = nonzero_indices[0] != nonzero_indices[1]
			x = torch.tensor(nonzero_indices[0][valid_indices_mask])
			y = torch.tensor(nonzero_indices[1][valid_indices_mask])

			index_pairs = torch.stack((x, y), dim=1)

			# Create a tensor filled with the current 'i' value for the third dimension
			time_tensor = torch.full((index_pairs.size(0), 1), i, dtype=torch.int)

			# Concatenate the original tensor pairs with the time tensor along the third dimension
			tensor_pairs_with_time = torch.cat((index_pairs, time_tensor), dim=1)

			# Append the current tensor pairs to the list
			all_tensor_pairs.append(tensor_pairs_with_time)

		# Concatenate all tensor pairs along the first dimension to create the final tensor
		final_tensor_pairs = torch.cat(all_tensor_pairs, dim=0)

		self.max_time = torch.tensor(i)
		self.min_time = torch.tensor(0)

		return {'idx': final_tensor_pairs, 'vals': torch.ones(final_tensor_pairs.size(0))}
