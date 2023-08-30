# -*- coding: utf-8 -*-
"""
Created on Mon Aug 22 23:44:05 2022

@author: ZML
"""
import torch
from .communication import communication_Pretrain
from .common import setup, update_lr
from .Pretrain_library import train, val, test
import os.path as osp

def run(args):
      
    best_f1 = 0
    best_epoch = 0
    print('==> Preparing data..')
    param = {'Known_class': args.known_class, 'unKnown_class': args.unknown_class, 'Rotation': args.rotation, 'Resize': args.resize, 'CropSize':args.cropsize, 'Batchsize': args.batchsize, 'dirichlet': args.dirichlet}          
    if args.dataset=='Hyperkvasir':
        from data.fed_hyper_kvasir_relabel import get_dataloaders
    elif args.dataset=='Bloodmnist':
        param = {'dataset': args.dataset, 'Known_class': args.known_class, 'unKnown_class': args.unknown_class, 'Rotation': args.rotation, 'Resize': args.resize, 'CropSize':args.cropsize, 'Batchsize': args.batchsize, 'dirichlet': args.dirichlet}        
        from data.fed_MedMINIST_relabel import get_dataloaders
    elif args.dataset=='OrganMNIST3D':
        param = {'dataset': args.dataset, 'Known_class': args.known_class, 'unKnown_class': args.unknown_class, 'Rotation': args.rotation, 'Resize': args.resize, 'CropSize':args.cropsize, 'Batchsize': args.batchsize, 'dirichlet': args.dirichlet}        
        from data.fed_MedMINIST3D_relabel import get_dataloaders
    else:
        assert False
    trainloaders, valloader, closerloader, openloader, train_val_loaders = get_dataloaders(args.client_num, args.data_root, args.seed, param) 
    server_model, models, device, client_weights  = setup(args, trainloaders)   
    epoch = 0
    for epoch_it in range(args.epoches // args.worker_steps):        
        args.lr = update_lr(args.lr, epoch, args.epoches, lr_step=20, lr_gamma=0.5)        
        optimizers = [torch.optim.Adam(params=models[idx].parameters(), lr=args.lr, betas=(0.9, 0.99), amsgrad=False) for idx in range(args.client_num)]        
        for ws in range(args.worker_steps): 
            for client_idx in range(args.client_num):
                client_name = args.client_names[client_idx] 
                model, train_loader, optimizer= models[client_idx], trainloaders[client_idx], optimizers[client_idx]       
                # Do training and validation loops
                train_result = train(args, device, epoch, model, train_loader, optimizer)
                train_loss, train_acc, train_f1, train_recall, train_precision = train_result['loss'], train_result['acc'],train_result['f1'],train_result['recall'], train_result['precision']
                print(f"Train {client_name} [{epoch}/{args.epoches}] LR={args.lr:.7f} loss={train_loss:.3f} ACC={train_acc:.3f} F1={train_f1:.3f} Rec={train_recall:.3f} Prec={train_precision:.3f}")     
        server_model, models = communication_Pretrain(args, server_model, models, client_weights)
        val_result = val(args, device, epoch, server_model, valloader)
        val_loss, val_acc, val_f1, val_recall, val_prec = val_result['loss'], val_result['acc'],val_result['f1'],val_result['recall'], val_result['precision']               
        print()
        print(f"Val    [{epoch}/{args.epoches}] LR={args.lr:.7f} loss={val_loss:.3f} ACC={val_acc:.3f} F1={val_f1:.3f} Rec={val_recall:.3f} Prec={val_prec:.3f}") 
        print()

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch            
            osr_result, close_test_result = test(args, device, epoch, server_model, closerloader, openloader)
            osr_acc, osr_f1, osr_recall, osr_precision = osr_result['acc'],osr_result['f1'],osr_result['recall'],osr_result['precision']
            test_loss, test_acc, test_f1, test_recall, test_precision = close_test_result['loss'], close_test_result['acc'],close_test_result['f1'],close_test_result['recall'],close_test_result['precision']   
            print(f"Test-  OSR [{epoch}/{args.epoches}] LR={args.lr:.7f} ACC={osr_acc:.3f} F1={osr_f1:.3f} Rec={osr_recall:.3f} Prec={osr_precision:.3f}")
            print(f"Test-Close [{epoch}/{args.epoches}] LR={args.lr:.7f} loss={test_loss:.3f} ACC={test_acc:.3f} F1={test_f1:.3f} Rec={test_recall:.3f} Prec={test_precision:.3f}")                     
            #server
            state = {
                'net': server_model.state_dict(),
                }
            name_model = 'best_ckpt_'+args.mode+'_known_class_'+str(args.known_class)+'_unknown_class_'+str(args.unknown_class)+'_seed_'+str(args.seed)+'.pth'
            torch.save(state, osp.join(args.save_path,name_model))             
            #clients
            for clint_idx, mo in enumerate(models):           
                state = {
                    'net': mo.state_dict(),
                    }
                name_model = 'best_ckpt_'+args.mode+'_known_class_'+str(args.known_class)+'_unknown_class_'+str(args.unknown_class)+'_seed_'+str(args.seed)+'_C_'+str(clint_idx)+'.pth'
                torch.save(state, osp.join(args.save_path,name_model))        
            print(f'Saving best model . . . . . . . .')
            print() 
            
        epoch += 1
    
    print('------>Best performance--->>>>>>')
    print()
    print(f"Test-  OSR [{best_epoch}/{args.epoches}] ACC={osr_acc:.3f} F1={osr_f1:.3f} Rec={osr_recall:.3f} Prec={osr_precision:.3f}")
    print('=====================================================================================================================================')
        