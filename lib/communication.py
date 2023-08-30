# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 23:45:51 2022

@author: ZML
"""
import torch
import gc

def communication(args, server_model, models, client_weights):

    with torch.no_grad():
        # aggregate params
        for key in server_model.state_dict().keys():
            # num_batches_tracked is a non trainable LongTensor and
            # num_batches_tracked are the same for all clients for the given datasets
            if 'num_batches_tracked' in key:
                server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
            else:
                temp = torch.zeros_like(server_model.state_dict()[key])
                for client_idx in range(len(client_weights)):
                    temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                server_model.state_dict()[key].data.copy_(temp)
                for client_idx in range(len(client_weights)):
                    models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])             
              
    return server_model, models


def communication_FedOSR(args, server_model, models, client_weights):

    with torch.no_grad():
        # aggregate params
        for key in server_model.state_dict().keys():
            # num_batches_tracked is a non trainable LongTensor and
            # num_batches_tracked are the same for all clients for the given datasets
            if not 'auxiliary' in key:
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:                    
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

    return server_model, models

def compute_global_statistic(args, mean_clients,cov_clients,number_clients):
    
    D = mean_clients.shape[-1]
    number_total = number_clients.sum(0, keepdim = True) # 1, C
    mean_weights = number_clients/number_total.float() # K, C
    mean_clients_weighted = mean_clients * mean_weights.unsqueeze(2).expand([-1, -1, D])
    g_mean = mean_clients_weighted.sum(0) # C, D
    
    if (number_total > 1).all():
        cov_weight1 = (number_clients-1)/(number_total-1).float() # K, C 
        cov_weight2 = (number_clients)/(number_total-1).float() # K, C
        cov_weight3 = number_total/(number_total-1).float() # 1, C
    else:  
        cov_weight1 = (number_clients)/(number_total+1e-9).float() # K, C 
        cov_weight2 = (number_clients)/(number_total+1e-9).float() # K, C
        cov_weight3 = number_total/(number_total+1e-9).float() # 1, C    
    
    cov_term1 = cov_clients * cov_weight1.unsqueeze(2).unsqueeze(3).expand([-1, -1, D, D]) # K, C, D, D
    cov_term1 = cov_term1.sum(0) # C, D, D
    
    cov_term2 = torch.einsum('abcd, abde->abce', mean_clients.unsqueeze(3), mean_clients.unsqueeze(2)) # K, C, D, D
    cov_term2 = cov_term2 * cov_weight2.unsqueeze(2).unsqueeze(3).expand([-1, -1, D, D]) # K, C, D, D
    cov_term2 = cov_term2.sum(0) # C, D, D    
    cov_term3 = torch.einsum('abc, acd->abd', g_mean.unsqueeze(2), g_mean.unsqueeze(1)) # C, D, D
    cov_term3 = cov_term3 * cov_weight3.permute(1, 0).unsqueeze(2).expand([-1, D, D]) # C, D, D

    g_cov = cov_term1 + cov_term2 - cov_term3 # C, D, D  
    # b为n阶矩阵,e为单位矩阵,a为正实数。ae+b在a充分大时,ae+b为正定矩阵
    eye_matrix = torch.eye(g_cov.shape[1]).expand(g_cov.shape[0], g_cov.shape[1], g_cov.shape[1])
    # 保证g_cov为正定矩阵
    g_cov += 0.0001 * eye_matrix
    
    unknown_dis = []                    
    for index in range(args.known_class):
        #    if ((args.dataset =='OrganMNIST3D' or args.dataset =='Bloodmnist') and number_total[0][index] > 10) or args.dataset == 'Hyperkvasir': #and (not torch.isnan(g_mean[index]).any()) and (not torch.isnan(g_cov[index]).any()):
        if number_total[0][index] > 10:
            unknown_dis.append(torch.distributions.multivariate_normal.MultivariateNormal(g_mean[index], covariance_matrix=g_cov[index]))
        else:
            unknown_dis.append(None)

    del cov_term1, cov_term2, cov_term3, cov_weight1, cov_weight2, cov_weight3
    del g_cov, g_mean, eye_matrix
    gc.collect()
    
    return unknown_dis

def communication_FedOSR_DUS_CUS(args, server_model, models, client_weights, mean_clients,cov_clients,number_clients, unknown_dis):
    
    if len(mean_clients)>0:
        mean_clients = torch.stack(mean_clients, 0)
        cov_clients = torch.stack(cov_clients, 0)
        number_clients = torch.stack(number_clients, 0)        
        unknown_dis = compute_global_statistic(args, mean_clients, cov_clients, number_clients)

    with torch.no_grad():
        # aggregate params
        for key in server_model.state_dict().keys():
            # num_batches_tracked is a non trainable LongTensor and
            # num_batches_tracked are the same for all clients for the given datasets
            if not 'auxiliary' in key:
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:                    
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

    return server_model, models, unknown_dis


if __name__=="__main__":
    a = torch.randn(8, 6, 10)
    b = torch.randn(8, 6, 10, 10)
    c = torch.randint(1, 100, (8, 6))
    
    torch.manual_seed(1)
    sample_num = torch.randint(1,100, (8,6))
    
    known_class = 6
    
    
    mean_clients = []
    cov_clients = []
    number_clients = []      
    
    for i in range(8):
        mean_dict = []
        cov_dict = []
        number_dict = torch.zeros(known_class)
        unknown_dict = [torch.randn(sample_num[i][c], 100) for c in range(known_class)]
        for index in range(known_class):
            mean_dict.append(unknown_dict[index].mean(0))
            
            X = unknown_dict[index] - unknown_dict[index].mean(0)            
            cov_matrix = torch.mm(X.t(), X) / len(X)
            eye_matrix = torch.eye(unknown_dict[index].shape[1])
            cov_matrix += 0.0001 * eye_matrix
            cov_dict.append(cov_matrix)
            number_dict[index] =  len(X)
        
        
        mean_dict = torch.stack(mean_dict, dim = 0) # C, D
        cov_dict = torch.stack(cov_dict, dim = 0) # C, D, D
        
        mean_clients.append(mean_dict)
        cov_clients.append(cov_dict)
        number_clients.append(number_dict)

    mean_clients = torch.stack(mean_clients, 0)
    cov_clients = torch.stack(cov_clients, 0)
    number_clients = torch.stack(number_clients, 0)
    g_mean, g_cov = compute_global_statistic(mean_clients, cov_clients, number_clients)

    
    for index in range(g_mean.shape[0]):
        unknown_dis = torch.distributions.multivariate_normal.MultivariateNormal(g_mean[index], covariance_matrix=g_cov[index])
    
    
    
    