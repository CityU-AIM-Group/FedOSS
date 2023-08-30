# -*- coding: utf-8 -*-
"""
Created on Sun Jun 12 10:05:53 2022

@author: ZML
"""

import argparse
import os.path as osp
import numpy as np
import torch
import random
import warnings
from models.utils import pprint, ensure_path
warnings.filterwarnings('ignore')


def main(args):
    if args.mode == 'Pretain':
        from lib.Pretain import run
        run(args)        
    elif args.mode == 'Finetune':
        from lib.Finetune import run
        run(args)

 
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False #for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--model_type', default='softmax', type=str, help='Recognition Method')
    parser.add_argument('--backbone', default='Resnet18', type=str, help='Backbone type.')
    parser.add_argument('--dataset', default='Hyperkvasir',type=str,help='dataset configuration')
    parser.add_argument('--known_class', default=5,type=int,help='number of known class')
    parser.add_argument('--unknown_class', default=3,type=int,help='number of unknown class')
    parser.add_argument('--seed', default='0',type=int,help='random seed for dataset generation.')
    
    parser.add_argument('--data_root', default='./dataset/', type=str, help='data_root')
    parser.add_argument('--rotation', default=45,type=int,help='Rotation Angle')
    parser.add_argument('--resize', default=144,type=int,help='resize')
    parser.add_argument('--cropsize', default=128,type=int,help='crop size')
    parser.add_argument('--batchsize', default=16,type=int,help='minibatch size')    
    parser.add_argument('--epoches', default=200,type=int,help='epoches')
    
    parser.add_argument('--client_num', type=int, default=8, help='the number of clients')
    parser.add_argument('--worker_steps', type=int, default=1, help='step of worker')
    parser.add_argument('--mode', type=str, default='Pretain', help='Pretain, Finetune')

    parser.add_argument('--dirichlet', type=float, default=0.5,help='dirichlet alpha')    
    
    #Attack
    parser.add_argument('--eps', type=float, default=1.,help='eps') 
    parser.add_argument('--num_steps', type=int, default=10,help='num_steps') 
    parser.add_argument('--unknown_weight', type=float, default=1.,help='unknown_weight')

    parser.add_argument('--start_epoch', type=str, default='[5, 10, 15, 20, 25]', help='start_epoch') 
    parser.add_argument('--sample_from', type=int, default=8, help='sample_from') 

    import time
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print('The starting time ：{}'.format(now))
    args = parser.parse_args()
    client_names = ['Client'+str(i) for i in range(args.client_num)]
    args.client_names = client_names
    
    args.start_epoch = eval(args.start_epoch)
    
    pprint(vars(args))
    
    #os.environ['CUDA_VISIBLE_DEVICES'] =args.gpu
    set_seed(args.seed)

    save_path1 = osp.join('results','M{}-D{}-M{}-B{}'.format(args.mode, args.dataset,args.model_type, args.backbone))
    save_path2 = 'LR{}-K{}-U{}-Seed{}'.format(str(args.lr),str(args.known_class),str(args.unknown_class),str(args.seed))
    args.save_path = osp.join(save_path1, save_path2)
    ensure_path(save_path1, remove=False)
    ensure_path(args.save_path, remove=False)

    main(args)
    
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    print('The ending time ：{}'.format(now)) 
    
    
