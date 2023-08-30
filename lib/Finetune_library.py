# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 00:03:23 2022

@author: ZML
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from random import sample
import gc

def train(args, device, epoch, net, trainloader, optimizer, net_peers=None, attack = None, unknown_dis = None):
    net.train()
    for peer_net in net_peers:        
        peer_net.eval()    
    train_loss = 0
    pred_list = []
    label_list = []
    output_list = []
    criterion = nn.CrossEntropyLoss()
    
    net_peers_sample_number = args.client_num-1
    if args.dataset == 'Hyperkvasir':
        if args.unknown_class == 3:
            p_lower = 0
            p_upper = 1.
        if args.unknown_class == 9:            
            p_lower = 0
            p_upper = 1. 
    
    if args.dataset == 'Bloodmnist':
        if args.unknown_class == 3:
            p_lower = 0
            p_upper = 13./16
        if args.unknown_class == 5:            
            p_lower = 0
            p_upper = 14./16
    
    if args.dataset == 'OrganMNIST3D':
        if args.unknown_class == 4:
            p_lower = 0
            p_upper = 1.
        if args.unknown_class == 7:            
            p_lower = 0
            p_upper = 1.    
    unknown_dict = [None for i in range(args.known_class)]
    mean_dict = [None for i in range(args.known_class)]
    cov_dict = [None for i in range(args.known_class)]
    number_dict = torch.zeros(args.known_class)
    for batch_idx, (inputs, targets, img_dirs) in enumerate(trainloader):
        gc.collect()
        torch.cuda.empty_cache()
        inputs, targets = inputs.to(device), targets.long().to(device)        
        outs = net(inputs)
        outputs = outs['outputs']    
        aux_outputs = outs['aux_out']
        boundary_feats = outs['boundary_feats'] 
        discrete_feats = outs['discrete_feats']
        loss = criterion(outputs, targets)        
        loss += criterion(aux_outputs, targets) 
        if epoch>=0:      
            #Client Inconsistency-based Boundary Samples Recognition 
            net_peers_sample = sample(net_peers, net_peers_sample_number) # 随机抽取n个peer        
            _, aux_pred = aux_outputs.max(1)
            aux_preds_peers = torch.eq(aux_pred, targets).float()
            assert len(net_peers)== (args.client_num-1)        
            for idx, peer_net in enumerate(net_peers_sample):
                with torch.no_grad():
                    outs_peer = peer_net.aux_forward(boundary_feats.clone().detach())
                    aux_out_peer = outs_peer['aux_out']
                    _, aux_pred_peer = aux_out_peer.max(1)
                    aux_preds_peers += torch.eq(aux_pred_peer, targets).float()
            is_boundary_upper = torch.lt(aux_preds_peers/(net_peers_sample_number+1), p_upper)
            is_boundary_lower = torch.gt(aux_preds_peers/(net_peers_sample_number+1), p_lower)
            is_boundary = is_boundary_lower & is_boundary_upper    

            if (is_boundary.sum()>0 and args.dataset=='Hyperkvasir') or (is_boundary.sum()>1 and (args.dataset=='Bloodmnist' or args.dataset=='OrganMNIST3D')): #  batchnorm error when batchsize = 1               
                discrete_feats = discrete_feats[is_boundary]
                discrete_targets = targets[is_boundary]
                inputs_unknown, targets_unknown = attack.i_DUS(net, discrete_feats, discrete_targets, net_peers_sample)
                if inputs_unknown is not None:                                        
                    outs_unknown = net.discrete_forward(inputs_unknown.clone().detach()) 
                    outputs_unknown = outs_unknown['outputs']
                    # probabilistic distance
                    prob_unknown = torch.softmax(outputs_unknown,dim=-1)
                    PDs = prob_unknown[:,-1] - prob_unknown[:,:-1].max(-1)[0]                     
                    gt_unknown=torch.ones(outputs_unknown.shape[0]).long().to(device)*args.known_class                
                    for i in range(len(outputs_unknown)):
                        nowlabel=targets_unknown[i]
                        outputs_unknown[i][nowlabel]=-1e9                 
                    loss += criterion(outputs_unknown, gt_unknown) * args.unknown_weight          
                    
                    #start save unknown data
                    if epoch in args.start_epoch:
                        targets_unknown_numpy = targets_unknown.cpu().data.numpy() 
                        for index in range(len(targets_unknown)):
                            if (args.dataset=='Hyperkvasir' and PDs[index]>0) or ((args.dataset=='Bloodmnist' or args.dataset=='OrganMNIST3D') and PDs[index]>-1):
                                dict_key = targets_unknown_numpy[index]
                                unknown_sample = inputs_unknown[index].clone().detach().view(1, -1)
                                if unknown_dict[dict_key] == None:
                                    unknown_dict[dict_key] = unknown_sample
                                else:
                                    unknown_dict[dict_key] = torch.cat((unknown_dict[dict_key], unknown_sample),dim=0)                                    
                    if unknown_dis is not None: 
                        sample_c = torch.randint(0, args.known_class, (args.sample_from,))
                        sample_num = {index: 0 for index in range(args.known_class)}
                        for it in sample_c:
                            sample_num[it.item()] = sample_num[it.item()] + 1 
                        ood_samples = None
                        ood_targets = None                                                
                        for index in range(args.known_class):
                            if sample_num[index] > 0 and unknown_dis[index] != None:                            
                                generated_unknown_samples = unknown_dis[index].rsample((100,))
                                prob_density = unknown_dis[index].log_prob(generated_unknown_samples)
                                # keep the data in the low density area.
                                _, index_prob = torch.topk(- prob_density, sample_num[index])
                                generated_unknown_samples = generated_unknown_samples[index_prob].to(device)
                                if args.dataset=='Hyperkvasir':
                                    generated_unknown_samples = generated_unknown_samples.reshape(sample_num[index], 256, 8, 8)
                                elif args.dataset=='Bloodmnist':
                                    generated_unknown_samples = generated_unknown_samples.reshape(sample_num[index], 256, 2, 2)
                                elif args.dataset=='OrganMNIST3D':
                                    generated_unknown_samples = generated_unknown_samples.reshape(sample_num[index], 256, 2, 2, 2)                                    
                                else:
                                    assert False                                     
                                generated_unknown_targets = (torch.ones(sample_num[index])*index).long().to(device) 
                                if ood_samples is None:
                                    ood_samples = generated_unknown_samples
                                    ood_targets = generated_unknown_targets
                                else:
                                    ood_samples = torch.cat((ood_samples, generated_unknown_samples), 0) 
                                    ood_targets = torch.cat((ood_targets, generated_unknown_targets), 0)
                                del generated_unknown_samples
                        if ood_samples is not None and ood_samples.shape[0]>1:        
                            outs_unknown = net.discrete_forward(ood_samples.clone().detach()) 
                            outputs_unknown = outs_unknown['outputs'] 
                            gt_unknown=torch.ones(outputs_unknown.shape[0]).long().to(device)*args.known_class                
                            for i in range(len(outputs_unknown)):
                                nowlabel=ood_targets[i]
                                outputs_unknown[i][nowlabel]=-1e9                 
                            loss += criterion(outputs_unknown, gt_unknown) * args.unknown_weight                                
                                
        optimizer.zero_grad()        
        loss.backward()
        optimizer.step()     
        train_loss += loss.item()
        _, predicted = outputs[:, :args.known_class].max(1)
        
        pred_list.extend(predicted.cpu().numpy().tolist())
        label_list.extend(targets.cpu().numpy().tolist())    
        output_list.append(torch.nn.functional.softmax(outputs, dim=-1).cpu().detach().numpy())
    
        del inputs, loss
        gc.collect()

    if epoch in args.start_epoch:      
        for index in range(args.known_class):            
            if unknown_dict[index] is not None:                
                mean_dict[index] = unknown_dict[index].mean(0).cpu()
                X = unknown_dict[index] - unknown_dict[index].mean(0)
                cov_matrix = torch.mm(X.t(), X) / len(X)
                cov_dict[index] = cov_matrix.cpu()
                number_dict[index] =  len(X)
                del cov_matrix, X                
            else:
                for i in range(args.known_class):   
                    if unknown_dict[i] is not None:
                       break
                D = unknown_dict[i].shape[1]   
                mean_dict[index] = torch.zeros(D) 
                cov_dict[index] = torch.zeros(D, D)                          
        del unknown_dict
        gc.collect()
        
        mean_dict = torch.stack(mean_dict, dim = 0)
        cov_dict = torch.stack(cov_dict, dim = 0)

    for peer_net in net_peers:        
        peer_net.train()          

    loss_avg = train_loss/(batch_idx+1)
    mean_acc = 100*metrics.accuracy_score(label_list, pred_list)
    precision = 100*metrics.precision_score(label_list, pred_list, average='macro')    
    recall_macro = 100*metrics.recall_score(y_true=label_list, y_pred=pred_list, average='macro')      
    f1_macro = 100*metrics.f1_score(y_true=label_list, y_pred=pred_list, average='macro')    

    result = {'loss':loss_avg,
              'acc':mean_acc,
              'f1': f1_macro,
              'recall':recall_macro,
              'precision': precision,
              'mean_dict': mean_dict,
              'cov_dict':cov_dict,
              'number_dict': number_dict
              }
    return result

def val(args, device, epoch, net, valloader):
    net.eval()
    
    val_loss = 0
    pred_list = []
    label_list = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch_idx, (inputs, targets, img_dirs) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.long().to(device)
            outs = net(inputs)
            outputs = outs['outputs']    
            aux_outputs = outs['aux_out']
            loss = criterion(outputs, targets)        
            loss += criterion(aux_outputs, targets)
            val_loss += loss.item()               
            _, predicted = outputs[:, :args.known_class].max(1)
            pred_list.extend(predicted.cpu().numpy().tolist())
            label_list.extend(targets.cpu().numpy().tolist())    
            
        loss_avg = val_loss/(batch_idx+1)
        mean_acc = 100*metrics.accuracy_score(label_list, pred_list)
        precision = 100*metrics.precision_score(label_list, pred_list, average='macro')        
        recall_macro = 100*metrics.recall_score(y_true=label_list, y_pred=pred_list, average='macro')      
        f1_macro = 100*metrics.f1_score(y_true=label_list, y_pred=pred_list, average='macro')    
        confusion_matrix = metrics.confusion_matrix(y_true=label_list, y_pred=pred_list)   
        
        result = {'loss':loss_avg,
                      'acc':mean_acc,
                      'f1': f1_macro,
                      'recall':recall_macro,
                      'precision': precision,
                      'confusion_matrix':confusion_matrix,
                      }
    return result


def test(args, device, epoch, net, closerloader, openloader, threshold=0):
    net.eval()
    
    temperature = 1.
    with torch.no_grad():
        pred_list=[]
        targets_list=[]
        test_loss=0
        criterion = nn.CrossEntropyLoss()
        
        pred_list_temp = []
        label_list_temp = []
        
        for batch_idx, (inputs, targets, img_dirs) in enumerate(closerloader):
            inputs, targets = inputs.to(device), targets.long().to(device)
            outs = net(inputs)
            outputs = outs['outputs']    
            aux_outputs = outs['aux_out']
            loss = criterion(outputs, targets)        
            loss += criterion(aux_outputs, targets)       
            test_loss += loss.item()
            _, predicted = outputs[:, :args.known_class].max(1)
            pred_list_temp.extend(predicted.cpu().numpy().tolist())
            label_list_temp.extend(targets.cpu().numpy().tolist())    

        loss_avg = test_loss/(batch_idx+1)
        mean_acc = 100*metrics.accuracy_score(label_list_temp, pred_list_temp)
        precision = 100*metrics.precision_score(label_list_temp, pred_list_temp, average='macro')          
        recall_macro = 100*metrics.recall_score(y_true=label_list_temp, y_pred=pred_list_temp, average='macro')      
        f1_macro = 100*metrics.f1_score(y_true=label_list_temp, y_pred=pred_list_temp, average='macro')    
        confusion_matrix = metrics.confusion_matrix(y_true=label_list_temp, y_pred=pred_list_temp)   
        
        close_test_result = {'loss':loss_avg,
                      'acc':mean_acc,
                      'f1': f1_macro,
                      'recall':recall_macro,
                      'precision':precision,
                      'confusion_matrix':confusion_matrix}        
        
        prob_total = None
        for batch_idx, (inputs, targets, img_dirs) in enumerate(closerloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outs = net(inputs)
            outputs = outs['outputs']
            prob=nn.functional.softmax(outputs/temperature,dim=-1)
            if prob_total == None:
                prob_total = prob
            else:
                prob_total = torch.cat([prob_total, prob])
            targets_list.append(targets.cpu().numpy())
        
        for batch_idx, (inputs, targets, img_dirs) in enumerate(openloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outs = net(inputs)
            outputs = outs['outputs']
            prob=nn.functional.softmax(outputs/temperature,dim=-1)
            prob_total = torch.cat([prob_total, prob])
            
            targets = np.ones_like(targets.cpu().numpy())*args.known_class
            targets_list.append(targets)

        # openset recognition    
        targets_list=np.reshape(np.array(targets_list),(-1))
        _, pred_list = prob_total.max(1)          
        pred_list = pred_list.cpu().numpy()

        mean_acc = 100.0 * metrics.accuracy_score(targets_list, pred_list)
        precision = 100*metrics.precision_score(targets_list, pred_list, average='macro')                  
        recall_macro = 100.0*metrics.recall_score(y_true=targets_list, y_pred=pred_list, average='macro')      
        f1_macro = 100*metrics.f1_score(y_true=targets_list, y_pred=pred_list, average='macro')    
        confusion_matrix = metrics.confusion_matrix(y_true=targets_list, y_pred=pred_list)
                        
        osr_result = {'acc':mean_acc,
                      'f1': f1_macro,
                      'recall':recall_macro,
                      'precision':precision,
                      'confusion_matrix': confusion_matrix}

    return osr_result, close_test_result

