
import numpy as np
from itertools import product as cartesian_prod

import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn import cluster
from sklearn import svm

from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import os
import wandb

np.set_printoptions(precision=2)

def set_torchseed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	
class Args:
    def __init__(self):
        self.numlayer=4
        self.numnodes=1000
        self.beta=3.
        self.lr=0.001     
		
class DLGN_FC(nn.Module):
	def __init__(self, input_dim=None, output_dim=None, num_hidden_nodes=[], beta=30, mode='pwc'):		
		super(DLGN_FC, self).__init__()
		set_torchseed(6675)
		self.num_hidden_layers = len(num_hidden_nodes)
		self.beta=beta  # Soft gating parameter
		self.mode = mode
		self.num_nodes=[input_dim]+num_hidden_nodes+[output_dim]
		self.gating_layers=nn.ModuleList()
		self.value_layers=nn.ModuleList()
		
		for i in range(self.num_hidden_layers+1):
			if i!=self.num_hidden_layers:
				temp = nn.Linear(self.num_nodes[i], self.num_nodes[i+1], bias=False)
				# a = temp.weight.detach() 
				# a /= a.norm(dim=1, keepdim=True)
				# Append a gating layer
				self.gating_layers.append(temp)
			temp = nn.Linear(self.num_nodes[i], self.num_nodes[i+1], bias=False)
			# a = temp.weight.detach()
			# a /= a.norm(dim=1, keepdim=True)
			# Append a normal layer
			self.value_layers.append(temp)


	def set_parameters_with_mask(self, to_copy, parameter_masks):
		# self and to_copy are DLGN_FC objects with same architecture
		# parameter_masks is compatible with dict(to_copy.named_parameters())
		# This function is not used
		for (name, copy_param) in to_copy.named_parameters():
			copy_param = copy_param.clone().detach()
			orig_param  = self.state_dict()[name]
			if name in parameter_masks:
				param_mask = parameter_masks[name]>0
				orig_param[param_mask] = copy_param[param_mask]
			else:
				orig_param = copy_param.data.detach()
	

								

	def return_gating_functions(self):
		'''
		Returns the effective weights and biases of the gating functions
		'''
		effective_weights = []
		for i in range(self.num_hidden_layers):
			curr_weight = self.gating_layers[i].weight.detach()
			if i==0:
				effective_weights.append(curr_weight)
			else:
				effective_weights.append(torch.matmul(curr_weight,effective_weights[-1]))
		return effective_weights
		# effective_weights (and effective biases) is a list of size num_hidden_layers
							

	def forward(self, x):
		'''
		Forward pass of the DLGN
		'''
		gate_scores=[x]

		device = self.gating_layers[0].weight.device

		if self.mode=='pwc':
			values=[torch.ones(x.shape).to(device)]
		else:
			values=[x]
		
		for i in range(self.num_hidden_layers):
			gate_scores.append(self.gating_layers[i](gate_scores[-1]))
			curr_gate_on_off = torch.sigmoid(self.beta * gate_scores[-1])
			values.append(self.value_layers[i](values[-1])*curr_gate_on_off)
		values.append(self.value_layers[self.num_hidden_layers](values[-1]))
		# Values is a list of size 1+num_hidden_layers+1
		# gate_scores is a list of size 1+num_hidden_layers
		# Since DLGN is a Linear network, there are no non-linearities except for the gating function
		return values,gate_scores

def train_dlgn (DLGN_obj, train_data_curr,test_data_curr,
				train_labels_curr,test_labels_curr,num_epoch=1,parameter_mask=dict(),
				**kwargs):
	'''
	Train the DLGN model with given parameters
  	'''
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	DLGN_obj.to(device)
	set_torchseed(5000)
	criterion = nn.CrossEntropyLoss()

	use_wandb = False
	if 'use_wandb' in kwargs:
		use_wandb = True
    
	lr_ratio = 1
	if 'lr_ratio' in kwargs:
		lr_ratio = kwargs['lr_ratio']

	learning_rate = 0.0001
	if 'lr' in kwargs:
		learning_rate = kwargs['lr']
	
	optimizer = optim.Adam(DLGN_obj.parameters(), lr=learning_rate)



	train_data_torch = torch.Tensor(train_data_curr)
	test_data_torch = torch.Tensor(test_data_curr)

	train_labels_torch = torch.tensor(train_labels_curr, dtype=torch.int64)
	test_labels_torch = torch.tensor(test_labels_curr, dtype=torch.int64)

	num_batches = 10
	batch_size = len(train_data_curr)//num_batches
	losses=[]
	DLGN_obj_store = []
	best_test_error = len(test_labels_curr)
	
	saved_epochs = np.arange(0,3000,10)

	train_losses = []
	running_loss = 0.7*num_batches # initial random loss = 0.7 
	for epoch in tqdm(range(saved_epochs[-1])):  # loop over the dataset multiple times
		if epoch in saved_epochs:
			DLGN_obj_copy = deepcopy(DLGN_obj)
			DLGN_obj_copy.to(torch.device('cpu'))
			DLGN_obj_store.append(DLGN_obj_copy)
			train_losses.append(running_loss/num_batches)
			# print("Epoch: ",epoch," Loss: ",running_loss/num_batches)
			if running_loss/num_batches < 1e-5:
				break
		running_loss = 0.0
		train_acc = 0
		for batch_start in range(0,len(train_data_curr),batch_size):
			if (batch_start+batch_size)>len(train_data_curr):
				break
			optimizer.zero_grad()
			inputs = train_data_torch[batch_start:batch_start+batch_size]
			targets = train_labels_torch[batch_start:batch_start+batch_size].reshape(batch_size)
			inputs = inputs.to(device)
			targets = targets.to(device)
			values,gate_scores = DLGN_obj(inputs)
			outputs = torch.cat((-1*values[-1], values[-1]), dim=1)
			train_acc += torch.sum(torch.argmax(outputs, dim=1)==targets).cpu().detach().numpy()
			loss = criterion(outputs, targets)			
			loss.backward()
			for name,param in DLGN_obj.named_parameters():
				parameter_mask[name] = parameter_mask[name].to(device)
				param.grad *= parameter_mask[name]   
			for name,param in DLGN_obj.named_parameters():
				if "val" in name:
					param.grad /= lr_ratio
			optimizer.step()
			running_loss += loss.item()    
		train_acc = train_acc/len(train_labels_curr)
		losses.append(running_loss/num_batches)
		inputs = test_data_torch.to(device)
		targets = test_labels_torch.to(device)
		values,gate_scores =DLGN_obj(inputs)
		test_preds = torch.cat((-1*values[-1], values[-1]), dim=1)
		test_preds = torch.argmax(test_preds, dim=1)
		test_error= torch.sum(targets!=test_preds)
		if(use_wandb):
			wandb.log({"Train_loss":running_loss/num_batches, 
					   "epoch":epoch, 
					   "Train_accuracy": train_acc,
					   "Test_accuracy":1-(test_error.cpu().detach().numpy()/float(len(test_labels_curr)))
			  }) 
		if test_error < best_test_error:
			DLGN_obj_return = deepcopy(DLGN_obj)
			best_test_error = test_error
	DLGN_obj_return.to(torch.device('cpu'))
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	return train_losses, DLGN_obj_return, DLGN_obj_store

def get_trained_dlgn(DLGN_init,data,**kwargs):
	'''
	Returns the trained model
 	'''
	train_parameter_masks=dict()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	for name,parameter in DLGN_init.named_parameters():
		if name[:5]=="value_"[:5]:
			train_parameter_masks[name]=torch.ones_like(parameter) # Updating all value network layers
		if name[:5]=="gating_"[:5]:
			train_parameter_masks[name]=torch.ones_like(parameter)
		train_parameter_masks[name].to(device)
	set_torchseed(5000)
	train_data = data['train_data']
	test_data = data['test_data']
	train_data_labels = data['train_labels']
	test_data_labels = data['test_labels']
	train_losses, DLGN_obj_final, DLGN_obj_store = train_dlgn(train_data_curr=train_data,                           
                                                test_data_curr=test_data,
                                                train_labels_curr=train_data_labels,
                                                test_labels_curr=test_data_labels,
                                                DLGN_obj=deepcopy(DLGN_init),
												parameter_mask=train_parameter_masks,
												**kwargs
                                                )
	torch.cuda.empty_cache() 
	return DLGN_obj_final, DLGN_obj_store

def test_acc_dlgn(X_test,y_test,DLGN_obj):
	'''
	Computes the test accuracy of the DLGN model
	'''
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	X_test = torch.Tensor(X_test).to(device)
	preds_DLGN = DLGN_obj(X_test)[0]
	test_preds = torch.cat((-1*preds_DLGN[-1], preds_DLGN[-1]), dim=1)
	test_preds = torch.argmax(test_preds, dim=1)
	targets = torch.tensor(y_test, dtype=torch.int64).to(device)
	test_error= torch.sum(targets!=test_preds)
	test_error = test_error.to(torch.device('cpu'))
	return 1-test_error.detach().numpy()/float(len(y_test))

