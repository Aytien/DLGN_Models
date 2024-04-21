
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import BinaryHingeLoss
from custom_loss_fns import HingeLoss
from data_gen import CustomDataset
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from pegasos_solver import Pegasos, Pegasos_kernel
from DLGN_kernel import DLGN_Kernel
from DLGN_VN import DLGN_FC
from DLGN_VT import DLGN_VT
from DLGN_enums import ModelTypes, KernelTrainMethod, LossTypes, VtFit, Optim
from copy import deepcopy
import matplotlib.pyplot as plt

def sigmoid(u):
    return 1/(1+np.exp(-u))

def cal_loss(model, loss_fn, data, labels, config, print_acc=False):

        if config.model_type == ModelTypes.KERNEL:
            preds = model(data, data)
        else:
            preds = model(data)

        if config.loss_fn_type == LossTypes.HINGE:
            preds = preds.reshape(-1)
            preds_lab = torch.sign(preds) * 0.5 + 0.5
            preds_lab = preds_lab.to(torch.int64).reshape(-1)
            if type(labels) == np.ndarray:
                labels = torch.tensor(labels, dtype=torch.int64).reshape(-1)
            else:
                labels = labels.to(torch.int64).reshape(-1)
            if print_acc:
                # var = 1 - preds * (2*labels - 1)
                # plt.hist(var.cpu().detach().numpy())
                # plt.savefig('outputs/hinge_loss_hist.png')
                # plt.show()
                print("Accuracy: ", torch.sum(preds_lab == labels).item()/len(labels))
            return loss_fn(preds, labels)
        elif config.loss_fn_type == LossTypes.CE:
            preds = preds.reshape(-1,1)
            outputs = torch.cat((-1*preds, preds), dim=1)
            if type(labels) == np.ndarray:
                labels = torch.tensor(labels, dtype=torch.int64)
            else:
                labels = labels.to(torch.int64)
            if print_acc:
                print("Accuracy: ", torch.sum(torch.argmax(outputs, dim=1) == labels).item()/len(labels))
            return loss_fn(outputs, labels)


def train_model(data, config):

    device = config.device

    model = None
    model_type = config.model_type
    
    if model_type == ModelTypes.KERNEL:
        model = DLGN_Kernel(config.num_data, config.dim_in, config.width, config.depth, config.beta, config.alpha_init, config.loss_fn_type)
    elif model_type == ModelTypes.VN:
        model = DLGN_FC(config.dim_in, 1, config.num_hidden_nodes, config.beta, config.mode)
    elif model_type == ModelTypes.VT:
        model = DLGN_VT(config.dim_in, 1, config.num_hidden_nodes, config.beta, config.mode, config.value_scale, config.BN, config.prod, config.feat)
    

    if model == None:
        raise ValueError('Invalid model type')

    loss_fn = None
    loss_fn_type = config.loss_fn_type

    if loss_fn_type == LossTypes.HINGE:
        loss_fn = HingeLoss()
    elif loss_fn_type == LossTypes.BCE:
        loss_fn = nn.BCELoss()
    elif loss_fn_type == LossTypes.CE:
        loss_fn = nn.CrossEntropyLoss()
    

    # if loss_fn == None:
    #     print(loss_fn)
    #     raise ValueError('Invalid loss function type')



    model.to(device)
    loss_fn.to(device)
    data['train_data'] = data['train_data'].to(device)
    data['test_data'] = data['test_data'].to(device)
    data['train_labels'] = data['train_labels'].to(device).to(torch.int64)
    data['test_labels'] = data['test_labels'].to(device).to(torch.int64)

    ret_model = None

    if model_type == ModelTypes.KERNEL:
        ret_model = kernel_train_methods(model, loss_fn, data, config)
    elif model_type == ModelTypes.VN:
        ret_model = vn_train_methods(model, loss_fn, data, config)
    elif model_type == ModelTypes.VT:
        ret_model = vt_train_methods(model, loss_fn, data, config)
    return ret_model
    # write a switch statement for different train_methods

def kernel_train_methods(model, loss_fn, data, config):
    train_method = config.train_method
    
    if train_method == KernelTrainMethod.PEGASOS or train_method == KernelTrainMethod.SVC:
        return svc_train(model, loss_fn, data, config)#[0]
    elif train_method == KernelTrainMethod.VANILA:
        return vanila_train(model, loss_fn, data, config)[0]
    # elif train_method == KernelTrainMethod.SVC:
        # return svc_train(model, loss_fn, data, config)
    return None

def pegasos_train(model, loss_fn, data, config):

    train_data = data['train_data']
    test_data = data['test_data']
    train_labels = data['train_labels']
    test_labels = data['test_labels']

    log_features = config.log_features

    lr_pg = config.lr_pg # learning rate for the pegasos algorithm
    reg_pg = config.reg_pg # lambda for the pegasos algorithm

    optimizer = None
    if(config.optimizer_type == Optim.SGD):
        optimizer = torch.optim.SGD([{'params': [param for param in model.gates], 'lr': 0.1},])


    log_epochs = 10
    log_weight = log_epochs
    num_epochs = 1000
    value_freq = config.value_freq   #frequency of going thru value update sub-routine
    value_epochs = config.value_epochs # number of epochs for which value update happens
    update_value_epochs = list(range(0,num_epochs+1,value_freq))

    batch_size = 32
    train_dataset = CustomDataset(train_data, train_labels)
    test_dataset = CustomDataset(test_data, test_labels)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if(log_features):
        features_initial = model.log_features()
        features_train=[]

    Train_losses=[]
    for epoch in range(num_epochs):
        if epoch in update_value_epochs:
            num_iter = 1000
            clf = Pegasos_kernel(config.reg, num_iter, model.forward)
            clf.fit(train_data.cpu(), train_labels.cpu())
            model.alphas.data = torch.tensor(clf.alpha, requires_grad=False).to(config.device)
        #     # pick a random training example index from train_data and update the corresponding alpha
        #     #print loss before pegasos update
        #     print('Loss before pegasos update')
        #     print(f'Loss {loss_fn(model(train_data, train_data),train_labels).item():.4f}')
        #     # for i in range(value_epochs):
        #     for i in range(value_epochs):
        #         rand_index = np.random.randint(len(train_data))
        #         x = train_data[rand_index]
        #         y = train_labels[rand_index]
        #         model.alphas.data -= lr_pg*reg_pg*model.alphas.data
        #         model_output = model(train_data[rand_index],train_data)
        #         if y*model_output < 1:
        #             model.alphas.data[rand_index] += lr_pg*y
        #     print('Loss after pegasos update')
        #     print(f'Loss {loss_fn(model(train_data, train_data),train_labels).item():.4f}')

                
        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            pred = model(x_batch,train_data)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
        
        if epoch % log_weight == 0  and log_features:
            features_train.append(model.log_features())

        if epoch % log_epochs == 0:
            y_pred = model(train_data, train_data)
            loss_full = loss_fn(y_pred,train_labels)
            train_accuracy = 0.0
            if config.loss_fn_type == LossTypes.HINGE:
                train_accuracy = np.sum((y_pred.cpu().detach().numpy() > 0) == (train_labels.cpu().numpy() > 0))/len(train_labels)
            else:
                train_accuracy = np.sum((y_pred.cpu().detach().numpy() > 0.5) == (train_labels.cpu().numpy() > 0.5))/len(train_labels)
            if(config.use_wandb):
                wandb.log({'train_loss': loss_full.item(), 'train_accuracy': train_accuracy, 'epoch': epoch})
            Train_losses.append(loss_full.item())

            # fs = feature_stats(features_train[-1],data_dim=dim_in,tree_depth=tree_depth,dim_in=dim_in)
            print(f'Epoch {epoch} Loss {loss_full.item():.4f}')

            if loss_full.item() < 0.01:
                print(f'Early stopping at epoch {epoch} because loss is below 0.01')
                break

    if(log_features):
        features_final = model.log_features()

    return model, Train_losses

def svc_train(model, loss_fn, data, config):
    train_data = data['train_data']
    test_data = data['test_data']
    train_labels = data['train_labels']
    test_labels = data['test_labels']

    batch_size = 32
    train_dataset = CustomDataset(train_data, train_labels)
    test_dataset = CustomDataset(test_data, test_labels)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = None
    if(config.optimizer_type == Optim.SGD):
        optimizer = torch.optim.SGD([
        {'params': [param for param in model.gating_layers], 'lr': config.gates_lr},
        # {'params': model.alphas, 'lr': config.alpha_lr, 'weight_decay': config.weight_decay}
    ])
    elif config.optimizer_type == Optim.ADAM:
        optimizer = torch.optim.Adam([
        {'params': [param for param in model.gating_layers], 'lr': config.gates_lr},
        # {'params': model.alphas, 'lr': config.alpha_lr, 'weight_decay': config.weight_decay}
    ])

    log_epochs = 10
    log_weight = log_epochs
    num_epochs = config.epochs

    # log_features = config.log_features

    # if(log_features):
    #     features_initial = model.log_features()
    #     features_train=[]
    # Train_losses=[]
    update_value_epochs = list(range(0,num_epochs+1,config.value_freq))
    tepoch = tqdm(range(num_epochs+1))
    for epoch in tepoch:
        if epoch in update_value_epochs:
            pre_update_loss = cal_loss(model, loss_fn, train_data, train_labels, config, print_acc=True)
            print("Loss before updating alphas at epoch", epoch, " is ", pre_update_loss.item())
            npk = model.get_npk
            if config.train_method == KernelTrainMethod.SVC:
                clf = SVC(C=config.reg, kernel=npk)
                clf.fit(train_data.cpu(), train_labels.cpu())
                dual_coef = clf.dual_coef_.T
                model.alphas.data = torch.zeros_like(model.alphas.data)
                for (idx, sv_idx) in enumerate(clf.support_):
                    model.alphas.data[sv_idx] = torch.tensor(dual_coef[idx])
            elif config.train_method == KernelTrainMethod.PEGASOS:
                num_iter = config.num_iter
                kernel_loss_fn_type = 'hinge' if config.loss_fn_type == LossTypes.HINGE else 'logistic'
                clf = Pegasos_kernel(lambd=config.reg, num_iter=num_iter, kernel=npk, loss_fn_type=kernel_loss_fn_type)
                clf.fit(train_data.cpu(), train_labels.cpu())
                model.alphas.data = torch.tensor((clf.alpha * clf.y) / (config.reg * num_iter), requires_grad=False, dtype=torch.float32).to(config.device)
            post_update_loss = cal_loss(model, loss_fn, train_data, train_labels, config, print_acc=True)
            print("Loss after updating alphas at epoch", epoch, " is ", post_update_loss.item())
        
        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            outputs = model(x_batch,train_data).reshape(-1)
            if(config.loss_fn_type == LossTypes.CE):
                outputs.reshape(-1,1)
                outputs = torch.cat((-1*outputs, outputs), dim=1)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        train_loss = cal_loss(model, loss_fn, train_data, train_labels, config, print_acc=False)
        tepoch.set_description(f"Train Loss: {train_loss.item():.4f}")
    
    # cal_loss(model, loss_fn, test_data, test_labels, config, print_acc=True)
    return model


def vanila_train(model, loss_fn, data, config):
    train_data = data['train_data']
    test_data = data['test_data']
    train_labels = data['train_labels']
    test_labels = data['test_labels']

    batch_size = 32
    train_dataset = CustomDataset(train_data, train_labels)
    test_dataset = CustomDataset(test_data, test_labels)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = None
    if(config.optimizer_type == Optim.SGD):
        optimizer = torch.optim.SGD([
        {'params': [param for param in model.gates], 'lr': config.gates_lr},
        {'params': model.alphas, 'lr': config.alpha_lr, 'weight_decay': config.weight_decay}
    ])

    log_epochs = 10
    log_weight = log_epochs
    num_epochs = 1000

    log_features = config.log_features

    if(log_features):
        features_initial = model.log_features()
        features_train=[]
    Train_losses=[]

    for epoch in range(num_epochs):
        # model.train()
        # if epoch%100 ==0 :
        #     if epoch%100 ==0:    ## alternating every 20 epochs can we do better??
        #         flag = not(flag)
        #     if flag:
        #         print('Gates')
        #     else:
        #         print('Alphas')
        #     if flag:
        #         for param in model.gates:
        #             param.requires_grad = True
        #             #print(param)
        #         model.alphas.requires_grad = False
        #         #print(model.alphas)
        #     else:
        #         for param in model.gates:
        #             param.requires_grad = False
        #             #print(param)
        #         model.alphas.requires_grad = True
        #         #print(model.alphas)

        for x_batch, y_batch in train_dataloader:
            x_batch = x_batch
            y_batch = y_batch
            pred = model(x_batch,train_data)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


        if epoch % log_weight == 0 and log_features:
            features_train.append(model.log_features())
        if epoch % log_epochs == 0:
            y_pred = model(train_data, train_data)
            loss_full = loss_fn(y_pred,train_labels)
            train_accuracy = 0.0
            if config.loss_fn_type == LossTypes.HINGE:
                train_accuracy = np.sum((y_pred.cpu().detach().numpy() > 0) == (train_labels.cpu().numpy() > 0))/len(train_labels)
            else:
                train_accuracy = np.sum((y_pred.cpu().detach().numpy() > 0.5) == (train_labels.cpu().numpy() > 0.5))/len(train_labels)
            if(config.use_wandb):
                wandb.log({'train_loss': loss_full.item(), 'train_accuracy': train_accuracy, 'epoch': epoch})
            print(f'Train Accuracy: {train_accuracy}')
            Train_losses.append(loss_full.item())

            print(f'Epoch {epoch} Loss {loss_full.item():.4f}')

            if loss_full.item() < 0.01:
                print(f'Early stopping at epoch {epoch} because loss is below 0.01')
                break

    if(log_features):
        features_final = model.log_features()

    return model, Train_losses

def vn_train_methods(model, loss_fn, data, config):
    train_data = data['train_data']
    test_data = data['test_data']
    train_labels = data['train_labels']
    test_labels = data['test_labels']
    
    device = config.device
    lr_ratio = config.lr_ratio

    batch_size = 32
    train_dataset = CustomDataset(train_data, train_labels)
    test_dataset = CustomDataset(test_data, test_labels)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = None
    if(config.optimizer_type == Optim.ADAM):
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    
    num_epochs = 3000
    Train_losses = []
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
        train_acc = 0

        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            values, gate_scores = model(x_batch)
            # Check the sign of values and assign the label accordingly
            outputs = torch.cat((-1*values[-1], values[-1]), dim=1)
            pred = values[-1].clamp(0,1).to(torch.int16)
            train_acc += torch.sum(pred == y_batch).cpu().detach().numpy()
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            for name,param in model.named_parameters():
                parameter_mask[name] = parameter_mask[name].to(device)
                param.grad *= parameter_mask[name]   
            for name,param in model.named_parameters():
                if "val" in name:
                    param.grad /= lr_ratio
            optimizer.step()
            running_loss += loss.item()
        train_acc = train_acc/len(train_data)
        Train_losses.append(running_loss/num_batches)

def vt_train_methods(model, loss_fn, data, config):

    train_data = data['train_data']
    test_data = data['test_data']
    train_labels = data['train_labels']
    test_labels = data['test_labels']

    batch_size = 32
    train_dataset = CustomDataset(train_data, train_labels)
    test_dataset = CustomDataset(test_data, test_labels)

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    optimizer = None
    if(config.optimizer_type == Optim.SGD):
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
    elif(config.optimizer_type == Optim.ADAM):
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    epochs = config.epochs
    save_freq = config.save_freq
    value_freq = config.value_freq

    losses=[]
    acc_dict = {'train':[],'test':[]}
    model_store = []
    debug_models= []
    saved_epochs = list(range(0,epochs+1,save_freq))
    update_value_epochs = list(range(0,epochs+1,value_freq))

    best_test_error=len(test_data)
    # filename_suffix = str(data_config.num_points)+'_'+str(data_config.depth)+'_val_'+str(value_freq)

    # original_stdout = sys.stdout
    # filename = 'outputs/'+filename_suffix+'.txt'
    # if not os.path.exists('outputs'):
    #     os.mkdir('outputs')
    # with open(filename,'w') as f:
    #     sys.stdout = f
    #     print("Setup:")
    #     print("Num neurons : ", model.num_nodes)
    #     print(" Beta :", model.beta)
    #     print(" lr :", train_config.lr)
    #     print("value update:",update_value_epochs)
    #     sys.stdout = original_stdout


    tepoch = tqdm(range(saved_epochs[-1]+1))
    for epoch in tepoch:  # loop over the dataset multiple times

        # Value net training
        if epoch in update_value_epochs:
            # updating the value pathdim vector by optimising 

            # train_preds = model(train_data).reshape((-1,1))
            # outputs = torch.cat((-1*train_preds,train_preds), dim=1)
            # targets = torch.tensor(train_labels, dtype=torch.int64)
            # train_loss = loss_fn(outputs, targets)

            train_loss = cal_loss(model, loss_fn, train_data, train_labels, config, print_acc=True)
            # with open(filename,'a') as f:
            #     sys.stdout = f
            pre_update_loss = train_loss.item()
            print("Loss lefore updating value_net at epoch", epoch, " is ", train_loss.item())
            #     # print("Total path squared value", (model.value_layers.cpu().detach()**2).sum().numpy())
            #     print("Total path abs value", torch.abs(model.value_layers.cpu().detach()).sum().numpy())
            #     sys.stdout = original_stdout

            ew = model.return_gating_functions()
            beta = model.beta

            feat_vec=[]
            if model.prod=='op':
                rsh = [-1]+[1]*model.num_layer

            for i in range(len(ew)):
                if model.prod=='op':
                    rsh_new = rsh.copy()
                    rsh_new[i+1] = model.num_hidden_nodes[i]
                    if model.feat=='cf' and i!=0:
                        ew[i] = ew[i]@ew[i-1]
                    feat_vec.append(sigmoid(beta*np.dot(train_data.cpu().detach().numpy(),ew[i].cpu().T).reshape(tuple(rsh_new))))
                elif model.prod=='ip':
                    if model.feat=='cf' and i!=0:
                        ew[i] = ew[i]@ew[i-1]
                    feat_vec.append(sigmoid(beta*np.dot(train_data.cpu().detach().numpy(),ew[i].cpu().T)))

            cp_feat = feat_vec[0]
            for i in range(1,len(feat_vec)):
                cp_feat = cp_feat*feat_vec[i]

            cp_feat_vec = cp_feat.reshape((len(cp_feat),-1))
            if config.vt_fit == VtFit.NPKSVC:
                npk = model.get_npk
                clf = SVC(C=config.reg, kernel=npk)
                clf.fit(train_data.cpu(), train_labels.cpu())
                support_vectors = train_data[clf.support_]
                kernel_values = model.npk_forward(support_vectors.to(config.device)).cpu().detach().numpy()
                dual_coef = clf.dual_coef_.T.reshape(tuple([-1]+[1]*model.num_hidden_layers))
                value_wts = np.sum(dual_coef*kernel_values, axis=0)
            elif config.vt_fit == VtFit.PEGASOS:
                clf = Pegasos(config.reg, 10000)
                clf.fit(2*cp_feat_vec, train_labels.cpu())
                value_wts = clf.w.reshape(tuple([1]+model.num_hidden_nodes))
            elif config.vt_fit == VtFit.PEGASOSKERNEL:
                npk = model.get_npk
                num_iter = 10000   
                # kernel_loss_fn_type = 'hinge' if config.loss_fn_type == LossTypes.HINGE else 'logistic'
                kernel_loss_fn_type = 'hinge' if config.loss_fn_type == LossTypes.HINGE else 'hinge'
                clf = Pegasos_kernel(config.reg, num_iter, npk, loss_fn_type=kernel_loss_fn_type)
                clf.fit(train_data.cpu(), train_labels.cpu())
                kernel_values = model.npk_forward(train_data).cpu().detach().numpy()
                dual_coef = clf.alpha * clf.y
                dual_coef = dual_coef.reshape(tuple([-1]+[1]*model.num_hidden_layers))
                value_wts = np.sum(dual_coef*kernel_values, axis=0) / (config.reg * num_iter)
            else:
                if config.vt_fit == VtFit.LOGISTIC:
                    clf = LogisticRegression(C=config.reg, fit_intercept=False,max_iter=1000, penalty="l2", solver='liblinear')
                elif config.vt_fit == VtFit.LINEARSVC:
                    clf = LinearSVC(C=config.reg, fit_intercept=False,max_iter=1000, dual='auto', loss='hinge')
                elif config.vt_fit == VtFit.SVC:
                    clf = SVC(C=config.reg, kernel='linear', degree=1, gamma = 1, max_iter=1000)
                else :
                    # Throw error
                    print("Invalid value fitting method")
                    return None        
                clf.fit(2*cp_feat_vec, train_labels.cpu())
                if model.prod=='op':
                    value_wts  = clf.decision_function(np.eye(np.prod(model.num_hidden_nodes))).reshape(tuple([1]+model.num_hidden_nodes))
                elif model.prod=='ip':
                    value_wts  = clf.decision_function(np.eye(model.num_hidden_nodes[0]))

            A = model.value_layers.detach()
            A[:] = torch.Tensor(value_wts)

            train_loss = cal_loss(model, loss_fn, train_data, train_labels, config, print_acc=True)

            # with open(filename,'a') as f:
            #     sys.stdout = f
            post_update_loss = train_loss.item()
            print("Loss after updating value_net at epoch", epoch, " is ", train_loss.item())			
            # # print("Total path squared value", (model.value_layers.cpu().detach()**2).sum().numpy())
            #     print("Total path abs value", torch.abs(model.value_layers.cpu().detach()).sum().numpy())
            #     sys.stdout = original_stdout

        if( post_update_loss / pre_update_loss > 2.5):
            print("Early stopping at epoch ", epoch, " as the loss increased by a factor of 2.5")
            break
        # saving models to cpu
        # if epoch in saved_epochs:
        #     model_copy = deepcopy(model)
        #     model_copy.to(torch.device('cpu'))
        #     model_store.append(model_copy)

        # training the gating functions
        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad()
            outputs = model(x_batch).reshape(-1)
            if config.loss_fn_type == LossTypes.CE:
                outputs = outputs.reshape(-1,1)
                outputs = torch.cat((-1*outputs, outputs), dim=1)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            for name,param in model.named_parameters():
                if "val" in name:
                    param.grad *= 0.0
                if "gat" in name:
                    param.grad *= 1.
                if epoch<update_value_epochs[1]:
                    param.grad *= 0.5
            optimizer.step()

        train_loss = cal_loss(model, loss_fn, train_data, train_labels, config)

        # if epoch%10 == 0:
        #     print("Loss after updating at epoch ", epoch, " is ", train_loss.item())
        if train_loss < 5e-3:
            break
        if np.isnan(train_loss.detach().cpu().numpy()):
            break

        losses.append(train_loss.cpu().detach().clone().numpy())
        inputs = test_data
        targets = test_labels
        preds = model(inputs).reshape(-1,1)
        test_preds = torch.cat((-1*preds, preds), dim=1)
        test_preds = torch.argmax(test_preds, dim=1)
        test_error= torch.sum(targets!=test_preds)
        acc_dict['test'].append(1-test_error.item()/len(test_labels))

        if test_error.item() < best_test_error:
            model_return = deepcopy(model)
            best_test_error = test_error.item()

        train_preds = model(train_data).reshape(-1,1)
        train_preds = torch.cat((-1*train_preds, train_preds), dim=1)
        train_preds = torch.argmax(train_preds, dim=1)
        train_targets = train_labels
        train_error= torch.sum(train_targets!=train_preds)
        acc_dict['train'].append(1-train_error.item()/len(train_labels))

        tepoch.set_description("Loss %f" % train_loss)


        # if epoch%10 == 0:

        #     if model.feat=='cf':
        #         weights = model.return_gating_functions()
        #         Feature_list = [weights[0]]
        #         for w in weights[1:]:
        #             Feature_list.append(w @ Feature_list[-1])
        #         features = torch.cat(Feature_list, axis = 0)
        #     else:
        #         features = torch.cat(model.return_gating_functions(), axis = 0)
            
        #     features = features.cpu().detach()
        #     p_1 = feature_stats(features,data_dim=data_config.dim_in,tree_depth=data_config.depth,dim_in=data_config.dim_in,threshold=0.1)
        #     p_2 = feature_stats(features,data_dim=data_config.dim_in,tree_depth=data_config.depth,dim_in=data_config.dim_in,threshold=0.2)
        #     p_3 = feature_stats(features,data_dim=data_config.dim_in,tree_depth=data_config.depth,dim_in=data_config.dim_in,threshold=0.3)
        #     p_5 = feature_stats(features,data_dim=data_config.dim_in,tree_depth=data_config.depth,dim_in=data_config.dim_in,threshold=0.5)
        #     p_7 = feature_stats(features,data_dim=data_config.dim_in,tree_depth=data_config.depth,dim_in=data_config.dim_in,threshold=0.7)

        #     with open(filename,'a') as f:
        #         sys.stdout = f
        #         print(f'epoch:{epoch} train_error:{train_error.item()} test_error:{test_error.item()} p_1:{p_1}, p_2:{p_2}, p_3:{p_3}, p_5:{p_5} p_7:{p_7}')
        #         sys.stdout = original_stdout
    # logging.info(f'p_1:{p_1}, p_2:{p_2}, p_3:{p_3}, p_5:{p_5}', end='')
        
    # return losses, acc_dict, model_return,best_test_error, model_store
    return model

