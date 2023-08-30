# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 23:51:20 2022

@author: ZML
"""
import platform
import torch
from copy import deepcopy
import numpy as np
import os

def pretrained_3D(model, path):
    
    pretrained_dict = torch.load(path)['net']
    net_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict) and (v.shape == net_dict[k].shape)}
    net_dict.update(pretrained_dict)
    model.load_state_dict(net_dict, strict=True)
    print('Initializing 3D ResNet')
    print(len(net_dict))
    print(len(pretrained_dict)) 
    
    return model 

def setup(args, trainloaders):
    print("\n>>> Setting up")
    if platform.system()=='Windows':
        device = torch.device("cpu")
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    pretrained = True
    print(f'==> Building [{args.backbone}] model..')
    base0 = './results/'
    if args.mode == 'DUS_baseline':
         if args.backbone== 'Resnet18':
            from models.ResNet_DUS import resnet18
            server_model=resnet18(pretrained=pretrained, num_classes=args.known_class) 
         if args.backbone== 'Resnet34':
            from models.ResNet_DUS import resnet34
            server_model=resnet34(pretrained=pretrained, num_classes=args.known_class) 
         if args.backbone== 'Resnet18_3D':
            from models.ResNet_DUS import resnet18
            server_model=resnet18(pretrained=pretrained, num_classes=args.known_class)
            #https://discuss.pytorch.org/t/inconsistent-results-with-3d-maxpool-on-gpu/38558/3
            #torch.nn.MaxPool3d(kernel_size=3, stride=2, padding=1) leads to non-deterministic results
            server_model.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            from acsconv.converters import ACSConverter
            server_model = ACSConverter(server_model)
                            
    elif args.mode == 'DUS_CUS_finetune':
        if args.backbone== 'Resnet18':
            from models.ResNet_DUS_finetune import resnet18     
            base1 = 'MDUS_baseline-D'+args.dataset+'-Msoftmax-BResnet18'
            base2 = 'LR'+str(0.0005)+'-K'+str(args.known_class)+'-U'+str(args.unknown_class)+'-Seed'+str(args.seed)
            server = 'best_ckpt_DUS_baseline_known_class_'+str(args.known_class)+'_unknown_class_'+str(args.unknown_class)+'_seed_'+str(args.seed)+'.pth'
            pretrained = os.path.join(base0, base1, base2, server)
            server_model=resnet18(pretrained=pretrained, num_classes=args.known_class) 
            models = []
            for client_idx in range(args.client_num):
                client = 'best_ckpt_DUS_baseline_known_class_'+str(args.known_class)+'_unknown_class_'+str(args.unknown_class)+'_seed_'+str(args.seed)+'_C_'+str(client_idx)+'.pth'
                pretrained = os.path.join(base0, base1, base2, client)
                client_model=resnet18(pretrained=pretrained, num_classes=args.known_class)
                models.append(client_model)
            server_model = server_model.to(device)     
            sample_num = np.array([trainloader.dataset.__len__() for trainloader in trainloaders])
            client_weights = sample_num / sample_num.sum()
            models = [model.to(device) for model in models] # client的模型列表，从server端深拷贝5份        
            return server_model, models, device, client_weights
        if args.backbone== 'Resnet18_3D':
            from models.ResNet_DUS_finetune import resnet18
            base1 = 'MDUS_baseline-D'+args.dataset+'-Msoftmax-BResnet18_3D'
            base2 = 'LR'+str(0.0005)+'-K'+str(args.known_class)+'-U'+str(args.unknown_class)+'-Seed'+str(args.seed)
            server = 'best_ckpt_DUS_baseline_known_class_'+str(args.known_class)+'_unknown_class_'+str(args.unknown_class)+'_seed_'+str(args.seed)+'.pth'
            pretrained = os.path.join(base0, base1, base2, server)
            server_model=resnet18(pretrained=False, num_classes=args.known_class)
            #https://discuss.pytorch.org/t/inconsistent-results-with-3d-maxpool-on-gpu/38558/3
            #torch.nn.MaxPool3d(kernel_size=3, stride=2, padding=1) leads to non-deterministic results
            server_model.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            from acsconv.converters import ACSConverter
            server_model = ACSConverter(server_model)
            server_model = pretrained_3D(server_model, pretrained)            
            models = []
            for client_idx in range(args.client_num):
                client = 'best_ckpt_DUS_baseline_known_class_'+str(args.known_class)+'_unknown_class_'+str(args.unknown_class)+'_seed_'+str(args.seed)+'_C_'+str(client_idx)+'.pth'
                pretrained = os.path.join(base0, base1, base2, client)
                client_model=resnet18(pretrained=False, num_classes=args.known_class)
                #https://discuss.pytorch.org/t/inconsistent-results-with-3d-maxpool-on-gpu/38558/3
                #torch.nn.MaxPool3d(kernel_size=3, stride=2, padding=1) leads to non-deterministic results
                client_model.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
                client_model = ACSConverter(client_model)
                client_model = pretrained_3D(client_model, pretrained)                 
                models.append(client_model)
            server_model = server_model.to(device)     
            sample_num = np.array([trainloader.dataset.__len__() for trainloader in trainloaders])
            client_weights = sample_num / sample_num.sum()
            models = [model.to(device) for model in models] # client的模型列表，从server端深拷贝5份        
            return server_model, models, device, client_weights                
            
    server_model = server_model.to(device) 
    #client_weights = [1/args.client_num for i in range(args.client_num)] # client importance
    sample_num = np.array([trainloader.dataset.__len__() for trainloader in trainloaders])
    client_weights = sample_num / sample_num.sum()
    models = [deepcopy(server_model).to(device) for idx in range(args.client_num)] # client的模型列表，从server端深拷贝5份        
        
    return server_model, models, device, client_weights

def update_lr(lr, epoch, n_epoch, lr_step=20, lr_gamma=0.5):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    if (epoch + 1) % (n_epoch//4) == 0 and (epoch + 1) != n_epoch:  # Yeah, ugly but will clean that later
        lr *= lr_gamma
        print(f'>> New learning Rate: {lr}')
        
    return lr
