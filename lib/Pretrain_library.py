# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 00:03:23 2022

@author: ZML
"""
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics

def train(args, device, epoch, net, trainloader, optimizer):
    net.train()  
    train_loss = 0
    pred_list = []
    label_list = []
    output_list = []
    criterion = nn.CrossEntropyLoss()
 
    for batch_idx, (inputs, targets, img_dirs) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.long().to(device)
        optimizer.zero_grad()
        outs = net(inputs)
        outputs = outs['outputs']    
        aux_outputs = outs['aux_out']
        loss = criterion(outputs, targets)        
        loss += criterion(aux_outputs, targets)  
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs[:, :args.known_class].max(1)

        pred_list.extend(predicted.cpu().numpy().tolist())
        label_list.extend(targets.cpu().numpy().tolist())    
        output_list.append(torch.nn.functional.softmax(outputs, dim=-1).cpu().detach().numpy())
        
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



