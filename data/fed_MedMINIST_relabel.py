# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 16:25:16 2022

@author: ZML
"""
import os
import os.path as osp
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import copy
from torch.utils import data
import platform
from collections import Counter

def dirichlet_split_noniid(train_labels, alpha, n_clients,state):
    '''
    参数为alpha的Dirichlet分布将数据索引划分为n_clients个子集
    '''
    n_classes = train_labels.max()+1
    np.random.set_state(state)
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少
    class_idcs = [np.argwhere(train_labels==y).flatten() for y in range(n_classes)]
    # 记录每个K个类别对应的样本下标
    client_idcs = [[] for _ in range(n_clients)]
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs

def get_dataloaders(client_num, data_root, seed, param = {'dataset': 'Bloodmnist', 'Known_class': 5, 'unKnown_class': 3, 'Rotation': 45, 'Resize': 144, 'CropSize':128, 'Batchsize': 8, 'dirichlet':1.0}):
    known_class=param['Known_class']
    unknown_class=param['unKnown_class'] 
    batchsize =  param['Batchsize']
    dirichlet_alpha = param['dirichlet']
    total_class = 8
    assert known_class + unknown_class <= total_class
    #random choose knwon and unknown class
    known_class_list_candidate = np.arange(0, total_class)
    np.random.seed(seed)
    state = np.random.get_state()
    np.random.shuffle(known_class_list_candidate)
    known_class_list = known_class_list_candidate[:known_class]
    unknown_class_list = known_class_list_candidate[known_class:]
     
    data_set = np.load(data_root)
    trainx = data_set['train_images']
    trainy = data_set['train_labels']
    valx = data_set['val_images']
    valy = data_set['val_labels']
    testx = data_set['test_images']
    testy = data_set['test_labels']
    d = Counter(trainy[:,0])
    d_s = sorted(d.items(),key=lambda x:x[1],reverse=True)
    print('The number of class each',d_s)    
    print('The numbers of Training, Val, Test sets, {}, {}, {}'.format(len(trainx), len(valx), len(testx)))

    #relabel dataset
    knowndict={}
    unknowndict={}
    for i in range(len(known_class_list)):
        knowndict[known_class_list[i]]=i
    for j in range(len(unknown_class_list)):
        unknowndict[unknown_class_list[j]]=j+len(known_class_list)
    print(knowndict, unknowndict)

    trainy = np.array(trainy)
    valy = np.array(valy)        
    testy = np.array(testy)        
    copytrainy=copy.deepcopy(trainy)
    copyvaly=copy.deepcopy(valy)
    copytesty=copy.deepcopy(testy)  
    for i in range(len(known_class_list)):
        #修改已知类标签
        trainy[copytrainy==known_class_list[i]]=knowndict[known_class_list[i]] 
        valy[copyvaly==known_class_list[i]]=knowndict[known_class_list[i]]             
        testy[copytesty==known_class_list[i]]=knowndict[known_class_list[i]]
    for j in range(len(unknown_class_list)):
        #修改未知类标签
        trainy[copytrainy==unknown_class_list[j]]=unknowndict[unknown_class_list[j]]
        valy[copyvaly==unknown_class_list[j]]=unknowndict[unknown_class_list[j]]             
        testy[copytesty==unknown_class_list[j]]=unknowndict[unknown_class_list[j]]   
    origin_known_list=known_class_list
    origin_unknown_list=unknown_class_list
    new_known_list=np.arange(known_class)
    new_unknown_list=np.arange(known_class, known_class+len(unknown_class_list))
    print(origin_known_list, new_known_list, origin_unknown_list, new_unknown_list)
     
    #获取已知类的index 便于索引
    train_data_known_index=[]
    val_data_known_index=[]        
    test_data_known_index=[]
    for item in new_known_list:
        index=np.where(trainy==item)
        index=list(index[0])
        train_data_known_index=train_data_known_index+index
        index=np.where(valy==item)
        index=list(index[0])
        val_data_known_index=val_data_known_index+index            
        index=np.where(testy==item)
        index=list(index[0])
        test_data_known_index=test_data_known_index+index
        
    #获得未知类的index
    train_data_index_perm=np.arange(len(trainy))
    train_data_unknown_index=np.setdiff1d(train_data_index_perm,train_data_known_index)
    val_data_index_perm=np.arange(len(valy))
    val_data_unknown_index=np.setdiff1d(val_data_index_perm,val_data_known_index)        
    test_data_index_perm=np.arange(len(testy))
    test_data_unknown_index=np.setdiff1d(test_data_index_perm,test_data_known_index)
    print('Known and Unknow in Train:',len(train_data_known_index),len(train_data_unknown_index))
    print('Known and Unknow in Val:',len(val_data_known_index), len(val_data_unknown_index))
    print('Known and Unknow in Test:',len(test_data_known_index), len(test_data_unknown_index))

    assert (len(test_data_unknown_index)+len(test_data_known_index)==len(testy))
          
    #training  dataloader   
    trainloaders = []
    train_val_loaders = []    
    #np.random.set_state(state)
    #np.random.shuffle(train_data_known_index)
    if platform.system()=='Windows':
        num_workers = 0
    else:
        num_workers = 4
        
    train_labels = trainy[train_data_known_index]
    train_images = [trainx[idx] for idx in train_data_known_index]
    client_idcs = dirichlet_split_noniid(train_labels, alpha=dirichlet_alpha, n_clients=client_num, state=state)
    
    for i in range(client_num): 
        print('Client'+str(i)+' sample num:', len(client_idcs[i]))
        subtrain_data_known_index = client_idcs[i]
        client_trainset=MedMINIST(data_root, subtrain_data_known_index, 'train', train_images, train_labels, param)
        client_trainloader=torch.utils.data.DataLoader(client_trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers, drop_last=True)
        trainloaders.append(client_trainloader)
        
        client_trainset=MedMINIST(data_root, subtrain_data_known_index, 'train_val', train_images, train_labels, param)
        client_train_val_loader=torch.utils.data.DataLoader(client_trainset, batch_size=1, shuffle=False, num_workers=num_workers)
        train_val_loaders.append(client_train_val_loader)        

    # val dataloader
    valset=MedMINIST(data_root,val_data_known_index,'valclose', valx, valy, param)
    valloader=torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=num_workers)

    closeset=MedMINIST(data_root,test_data_known_index,'testclose', testx, testy, param=param)
    closeloader=torch.utils.data.DataLoader(closeset, batch_size=1, shuffle=False, num_workers=num_workers)

    openset=MedMINIST(data_root,test_data_unknown_index,'testopen',testx, testy, param=param)
    openloader=torch.utils.data.DataLoader(openset, batch_size=1, shuffle=False, num_workers=num_workers) 
    
    return trainloaders, valloader, closeloader, openloader, train_val_loaders

class MedMINIST(Dataset):
    def __init__(self, data_root, data_index, setname, datax, datay, param = {'Known_class': 6, 'unKnown_class': 9, 'Rotation': 45, 'Resize': 144, 'CropSize':128}):
        self.data_root = data_root 
        self.data_index = data_index
        self.setname = setname
        self.datax = datax
        self.datay = datay

        self.transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])            
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])        
    
    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, index):
        img_id = self.data_index[index]
        img = self.datax[img_id]        
        label = self.datay[img_id][0]
            
        if self.setname=='train':
            img = self.transform_train(img)
        else:
            img = self.transform_test(img)
        
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        
        return img, label, img_id




if __name__ == '__main__':
    datadir = './dataset/bloodmnist.npz'
  
    trainloaders, valloader, closeloader, openloader, train_val_loaders = get_dataloaders(8, datadir, 0)

    for c, trainloader in enumerate(trainloaders):
        print(trainloader.dataset.__len__())
        for i, data_ in enumerate(trainloader):
            img, label, img_dir = data_
            print(c,label)
            if i > 1:
                break

        img = img[0].numpy()
        print(img.shape)
        import matplotlib.pyplot as plt
        img = img.swapaxes(0,1)
        img = img.swapaxes(1,2)
        #plt.imshow(img)
        #plt.show()
        #print(c, i,img_dir)
