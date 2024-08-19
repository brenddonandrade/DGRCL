import torch
import utils as u
import pandas as pd
import numpy as np
import contrast as c


class Trainer():
	def __init__(self, args, splitter, model, dynamic_loss, dataset, num_classes):
		self.args = args
		self.splitter = splitter
		self.tasker = splitter.tasker
		self.model = model
		self.d_loss = dynamic_loss
		self.num_nodes = dataset.num_nodes
		self.data = dataset
		self.num_classes = num_classes

		self.gcn_opt = torch.optim.Adam(self.model.gcn.parameters(), lr = args.learning_rate)
		self.classifier_opt = torch.optim.Adam(self.model.classifier.parameters(), lr=args.learning_rate)
		self.cl_opt = torch.optim.Adam(self.model.cl.parameters(), lr=args.learning_rate)

	def train(self, s, relation_mask):
		self.model.gcn.train()
		self.model.classifier.train()
		self.model.cl.train()

		self.gcn_opt.zero_grad()
		self.classifier_opt.zero_grad()
		self.cl_opt.zero_grad()

		s = self.prepare_sample(s)
		# hist_adj_list, hist_ndFeats_list, node_indices, mask_list
		predictions, nodes_embs = self.predict(s.hist_adj_list,
											   s.hist_ndFeats_list,
											   s.label_sp['idx'],
											   s.node_mask_list)

		# predictions(1026,2) s.label_sp['vals'](1026,)
		d_loss = self.d_loss(predictions, s.label_sp['vals'])
		# torch.nn.functional(predictions, s.label_sp['vals'])
		s_loss = 0
		if self.args.ablation_contrastive is True:
			c_views = c.get_views(s, self.args, relation_mask)

			z1 = self.model.cl(c_views['x_1'], c_views['a_1'])
			z2 = self.model.cl(c_views['x_2'], c_views['a_2'])
			s_loss = self.model.cl.loss(z1, z2)

			# print('d_loss:{:.4f}, s_loss{:.4f}'.format(d_loss.item(), s_loss.item()))

		loss = d_loss + self.args.c_tem*s_loss

		loss.backward()
		self.gcn_opt.step()
		self.classifier_opt.step()
		self.cl_opt.step()

		return loss.item()

	def eval(self, s):
		self.model.gcn.eval()
		self.model.classifier.eval()
		self.model.cl.eval()

		with torch.no_grad():
			s = self.prepare_sample(s)
			# hist_adj_list, hist_ndFeats_list, node_indices, mask_list
			predictions, _ = self.predict(s.hist_adj_list,
												   s.hist_ndFeats_list,
												   s.label_sp['idx'],
												   s.node_mask_list)

			acc, prec, rec, f1, roc_auc, mcc = u.eva_metrix(predictions, s.label_sp['vals'])
			# predictions(1026,2) s.label_sp['vals'](1026,)
			d_loss = self.d_loss(predictions, s.label_sp['vals'])
			loss = d_loss

		return loss.item(), acc.item(), prec.item(), rec.item(),  f1.item(), roc_auc.item(), mcc.item()

	def predict(self,hist_adj_list,hist_ndFeats_list,node_indices,mask_list):
		nodes_embs = self.model.gcn(hist_adj_list,
							  hist_ndFeats_list,
							  mask_list)

		predict_batch_size = 100000
		gather_predictions=[]
		for i in range(1 +(node_indices.size(1)//predict_batch_size)):
			cls_input = self.gather_node_embs(nodes_embs, node_indices[:, i*predict_batch_size:(i+1)*predict_batch_size])
			predictions = self.model.classifier(cls_input)
			gather_predictions.append(predictions)
		gather_predictions=torch.cat(gather_predictions, dim=0)
		return gather_predictions, nodes_embs

	def gather_node_embs(self,nodes_embs,node_indices):
		cls_input = []
		# gather all valid node_embeddings via cat
		for node_set in node_indices:
			cls_input.append(nodes_embs[node_set.to(torch.int)])
		return torch.cat(cls_input,dim = 1)

	def prepare_sample(self, sample):
		sample = u.Namespace(sample)
		edge_index_list = []
		x_list = []
		for i in sample.hist_adj_list:
			each_edge_index = i['idx'].permute(2, 1, 0).squeeze()
			edge_index_list.append(each_edge_index.to(self.args.device))
		for j in sample.hist_ndFeats_list:
			each_x = torch.squeeze(j, dim=0)
			x_list.append(each_x.to(self.args.device))

		for i, adj in enumerate(sample.hist_adj_list):
			adj = u.sparse_prepare_tensor(adj, torch_size=[self.num_nodes])
			sample.hist_adj_list[i] = adj.to(self.args.device)

			nodes = self.tasker.prepare_node_feats(sample.hist_ndFeats_list[i])

			sample.hist_ndFeats_list[i] = nodes.to(self.args.device)
			node_mask = sample.node_mask_list[i]
			sample.node_mask_list[i] = node_mask.to(self.args.device).t() #transposed to have same dimensions as scorer

		label_sp = self.ignore_batch_dim(sample.label_sp)

		if self.args.task in ["link_pred", "edge_cls"]:
			label_sp['idx'] = label_sp['idx'].to(self.args.device).t()   ####### ALDO TO CHECK why there was the .t() -----> because I concatenate embeddings when there are pairs of them, the embeddings are row vectors after the transpose
		else:
			label_sp['idx'] = label_sp['idx'].to(self.args.device)

		label_sp['vals'] = label_sp['vals'].type(torch.long).to(self.args.device)
		sample.label_sp = label_sp

		sample.edge_index_list = edge_index_list
		sample.x_list = x_list
		return sample

	def ignore_batch_dim(self,adj):
		if self.args.task in ["link_pred", "edge_cls"]:
			adj['idx'] = adj['idx'][0]
		adj['vals'] = adj['vals'][0]
		return adj

	def save_node_embs_csv(self, nodes_embs, indexes, file_name):
		csv_node_embs = []
		for node_id in indexes:
			orig_ID = torch.DoubleTensor([self.tasker.data.contID_to_origID[node_id]])

			csv_node_embs.append(torch.cat((orig_ID,nodes_embs[node_id].double())).detach().numpy())

		pd.DataFrame(np.array(csv_node_embs)).to_csv(file_name, header=None, index=None, compression='gzip')
