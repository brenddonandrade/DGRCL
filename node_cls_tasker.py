import taskers_utils as tu
import utils as u

class Node_Cls_Tasker():
	def __init__(self,args,dataset):
		self.data = dataset

		self.max_time = dataset.max_time

		self.args = args

		self.num_classes = 2

		self.feats_per_node = dataset.feats_per_node

		self.nodes_labels_times = dataset.nodes_labels_times

		self.get_node_feats = self.build_get_node_feats(args,dataset)

		self.prepare_node_feats = self.build_prepare_node_feats(args,dataset)

		self.is_static = False

	def build_get_node_feats(self, args, dataset):
		def get_node_feats(i,adj):
			dynamic_feats_with_i = dataset.nodes_feats[dataset.nodes_feats[:, 0] == i]
			# Remove the index column to get (N, 100) tensor
			dynamic_feats = dynamic_feats_with_i[:, 1:]
			return dynamic_feats
		return get_node_feats

	def build_prepare_node_feats(self,args,dataset):
		if args.use_2_hot_node_feats or args.use_1_hot_node_feats:
			def prepare_node_feats(node_feats):
				return u.sparse_prepare_tensor(node_feats,
											   torch_size= [dataset.num_nodes,
											   				self.feats_per_node])

		else:
			def prepare_node_feats(node_feats):
				return node_feats[0]

		return prepare_node_feats

	def get_sample(self, idx, r_mask, test):
		hist_adj_list = []
		hist_ndFeats_list = []
		hist_mask_list = []
		for i in range(idx - self.args.num_hist_steps, idx+1):
			#all edgess included from the beginning
			cur_adj = tu.get_sp_adj(edges = self.data.edges,
									time = i,
									weighted = True,
									time_window = self.args.adj_mat_time_window) #changed this to keep only a time window
			node_mask = tu.get_node_mask(cur_adj, self.data.num_nodes)
			node_feats = self.get_node_feats(i, cur_adj)
			# cur_adj = tu.normalize_adj(r_mask, adj = cur_adj, num_nodes = self.data.num_nodes)

			hist_adj_list.append(cur_adj)
			hist_ndFeats_list.append(node_feats)
			hist_mask_list.append(node_mask)

		label_adj = self.get_node_labels(idx)

		return {'idx': idx,
				'hist_adj_list': hist_adj_list,
				'hist_ndFeats_list': hist_ndFeats_list,
				'label_sp': label_adj,
				'node_mask_list': hist_mask_list}


	def get_node_labels(self,idx):
		node_labels = self.nodes_labels_times
		subset = node_labels[:,2]==idx
		label_idx = node_labels[subset,0]
		label_vals = node_labels[subset,1]

		return {'idx': label_idx,
				'vals': label_vals}
