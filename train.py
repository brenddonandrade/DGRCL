import logging
import os
import random
import time

import egcn_h
# models
import r_models as clf
# taskers
import node_cls_tasker as nct
import splitter as sp
# datasets
import stock_temporal_dl as stock_temp
import trainer as tr
from contrast import *
from dgrcl import *


def random_param_value(param, param_min, param_max, type='int'):
	if str(param) is None or str(param).lower()=='none':
		if type == 'int':
			return random.randrange(param_min, param_max+1)
		elif type == 'logscale':
			interval = np.logspace(np.log10(param_min), np.log10(param_max), num=100)
			return np.random.choice(interval, 1)[0]
		else:
			return random.uniform(param_min, param_max)
	else:
		return param


def build_gcn(args,tasker):
	gcn_args = u.Namespace(args.gcn_parameters)
	gcn_args.feats_per_node = tasker.feats_per_node
	return egcn_h.EGCN(gcn_args, activation=torch.nn.RReLU(), device=args.device)



def build_classifier(args,tasker):
	mult = 1
	in_feats = args.gcn_parameters['layer_2_feats'] * mult
	return clf.Classifier(args,in_features = in_feats, out_features = tasker.num_classes).to(args.device)

def load_relation_data(args):
	if args.stock_name == 'nasdaq':
		if args.relation_type == 'sector':
			relation_path = r'nasdaq_sector_industry.npy'
		elif args.relation_type == 'wiki':
			relation_path = r'nasdaq_wiki.npy'
	elif args.stock_name == 'nyse':
		if args.relation_type == 'sector':
			relation_path = r'nyse_sector_industry.npy'
		elif args.relation_type == 'wiki':
			relation_path = r'nyse_wiki.npy'

	mask = np.load(relation_path)
	if args.relation_self_loop is True:
		return mask
	elif args.relation_self_loop is False:
		np.fill_diagonal(mask, 0)
		return mask

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] =str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
	save = './save/'
	parser = u.create_parser()
	args = u.parse_args(parser)

	set_seed(args.seed)

	args.data_test = True
	args.c_tem = 0.1

	if args.stock_name == 'nasdaq':
		args.num_nodes = 1026
	elif args.stock_name == 'nyse':
		args.num_nodes = 1737

	print(args)
	timestamp = time.strftime("%Y%m%d-%H%M%S")
	cur_dir = os.getcwd()
	rel_dir = "save"
	log_file = os.path.join(cur_dir, rel_dir, f"training_log_{timestamp}.txt")
	logging.basicConfig(filename=log_file, level=logging.INFO)
	logging.info(args)

	relation_mask = load_relation_data(args)

	#build the dataset
	dataset = stock_temp.Stock_Temporal_Dataset(args)
	dataset.edges['idx'].to(args.device)
	dataset.edges['vals'].to(args.device)
	dataset.nodes.to(args.device)
	dataset.nodes_feats.to(args.device)
	dataset.nodes_labels_times.to(args.device)

	#build the tasker
	tasker = nct.Node_Cls_Tasker(args,dataset)
	#build the splitter
	splitter = sp.splitter(args, tasker, relation_mask)

	model = DGRCL(args, tasker)
	#build a loss
	cross_entropy = F.cross_entropy

	#trainer
	trainer = tr.Trainer(args,
						 splitter = splitter,
						 model = model,
						 dynamic_loss = cross_entropy,
						 dataset = dataset,
						 num_classes = tasker.num_classes)

	nparam_gcn = sum([p.nelement() for p in model.gcn.parameters()])
	nparam_classifier = sum([p.nelement() for p in model.classifier.parameters()])
	nparam_cl = sum([p.nelement() for p in model.cl.parameters()])
	print('nparam_classifier: ' + str(nparam_classifier) + ' nparam_cl: ' + str(nparam_cl))
	print('nparam:', nparam_gcn + nparam_classifier + nparam_cl)

	print('Start training...')
	his_loss = []
	his_mcc = []
	train_time = []
	# min_loss = float('inf')
	max_mcc = -float('inf')
	for e in range(args.num_epochs):

		# train
		train_loss = []
		t1 = time.time()
		for s in splitter.train:
			loss = trainer.train(s, relation_mask)
			train_loss.append(loss)

		t2 = time.time()
		train_time.append(t2 - t1)

		# validation
		valid_loss = []
		valid_acc = []
		valid_prec = []
		valid_rec = []
		valid_f1 = []
		valid_roc_auc = []
		valid_mcc = []
		for s in splitter.dev:
			loss, acc, prec, rec, f1, roc_auc, mcc = trainer.eval(s)
			valid_loss.append(loss)
			valid_acc.append(acc)
			valid_prec.append(prec)
			valid_rec.append(rec)
			valid_f1.append(f1)
			valid_roc_auc.append(roc_auc)
			valid_mcc.append(mcc)

		mvalid_loss = np.mean(valid_loss)
		mvalid_acc = np.mean(valid_acc)
		mvalid_prec = np.mean(valid_prec)
		mvalid_rec = np.mean(valid_rec)
		mvalid_f1 = np.mean(valid_f1)
		mvalid_roc_auc = np.mean(valid_roc_auc)
		mvalid_mcc = np.mean(valid_mcc)
		his_loss.append(mvalid_loss)
		his_mcc.append(mvalid_mcc)

		log = 'Epoch: {:03d}, Valid Acc: {:.4f}, Valid Prec: {:.4f}, Valid Rec: {:.4f}, Valid F1: {:.4f}, Valid Roc: {:.4f}, Valid Mcc: {:.4f}, Training Time: {:.4f}/epoch'
		print(log.format(e, mvalid_acc, mvalid_prec, mvalid_rec, mvalid_f1, mvalid_roc_auc, mvalid_mcc, (t2 - t1)),
			  flush=True)
		logging.info(
			log.format(e, mvalid_acc, mvalid_prec, mvalid_rec, mvalid_f1, mvalid_roc_auc, mvalid_mcc, (t2 - t1)))

		# if mvalid_loss < min_loss:
		# 	torch.save(model, save + 'model_epoch_' + str(e) + '_' + str(round(mvalid_loss, 2)) + '.pth')
		# 	min_loss = mvalid_loss
		if mvalid_mcc > max_mcc:
			torch.save(model, save + 'model_epoch_' + str(e) + '_' + str(round(mvalid_mcc, 4)) + '.pth')
			max_mcc = mvalid_loss

	bestid = np.argmax(his_mcc)
	trainer.model = torch.load(save + 'model_epoch_' + str(bestid) + '_' + str(round(his_mcc[bestid], 4)) + '.pth')

	print("Training finished")
	print("The best mcc on best model is", str(round(his_mcc[bestid], 4)))
	logging.info("The best mcc on best model is %s", str(round(his_mcc[bestid], 4)))

	# test
	test_acc = []
	test_prec = []
	test_rec = []
	test_f1 = []
	test_roc_auc = []
	test_mcc = []

	for s in splitter.test:
		loss, acc, prec, rec, f1, roc_auc, mcc = trainer.eval(s)
		test_acc.append(acc)
		test_prec.append(prec)
		test_rec.append(rec)
		test_f1.append(f1)
		test_roc_auc.append(roc_auc)
		test_mcc.append(mcc)

	mtest_acc = np.mean(test_acc)
	mtest_prec = np.mean(test_prec)
	mtest_rec = np.mean(test_rec)
	mtest_f1 = np.mean(test_f1)
	mtest_roc_auc = np.mean(test_roc_auc)
	mtest_mcc = np.mean(test_mcc)

	log = 'Results on best model -- test Acc: {:.4f}, test Prec: {:.4f}, test Rec: {:.4f}, test F1: {:.4f}, test Roc: {:.4f}, test Mcc: {:.4f}'
	print(log.format(mtest_acc, mtest_prec, mtest_rec, mtest_f1, mtest_roc_auc, mtest_mcc))
	logging.info(log.format(mtest_acc, mtest_prec, mtest_rec, mtest_f1, mtest_roc_auc, mtest_mcc))

	print("Training finished")
	print('_' * 100)
	logging.info("Training finished")
	logging.info('_' * 100)
	print('___________________________________________________________________________________')

