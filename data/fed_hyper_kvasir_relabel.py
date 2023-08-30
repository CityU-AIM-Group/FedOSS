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

types_ = {'bbps': 0, 'polyps':1, 'cecum':2, 'dyed-lifted-polyps':3, 'pylorus':4,
          'dyed-resection-margins':5,'z-line':6, 'ulcerative-colitis':7,'retroflex-stomach':8,
          'esophagitis':9, 'retroflex-rectum':10,'impacted-stool':11,
          'barretts':12,'ileum':13,'hemorrhoids':14}

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



def get_dataloaders(client_num, data_root, seed, param = {'Known_class': 6, 'unKnown_class': 9, 'Rotation': 45, 'Resize': 144, 'CropSize':128, 'Batchsize': 8, 'dirichlet':0.5}):
    known_class=param['Known_class']
    unknown_class=param['unKnown_class'] 
    batchsize =  param['Batchsize']
    dirichlet_alpha = param['dirichlet']
    total_class=15
    assert known_class + unknown_class <= total_class
    #random choose knwon and unknown class
    known_class_list_candidate = np.arange(known_class + 3)
    np.random.seed(seed)
    state = np.random.get_state()
    np.random.shuffle(known_class_list_candidate)
    known_class_list = known_class_list_candidate[:known_class]
    known_class_list_rest = known_class_list_candidate[known_class:]
    unknow_classes_perm=np.arange(known_class+3, total_class)
    total_unknow_classes_perm = np.hstack([known_class_list_rest, unknow_classes_perm])
    np.random.set_state(state)
    np.random.shuffle(total_unknow_classes_perm)
    unknown_class_list = total_unknow_classes_perm[:unknown_class]
    types_new = {}
    for key in types_.keys():
        if (types_[key] in unknown_class_list) or (types_[key] in known_class_list) :
            types_new[key] = types_[key]
    
    print('Known class list:', known_class_list,'Unknown class list', unknown_class_list)
    
    img_list = []
    label_list = []
    for ty in types_new.keys():
        img_dir = osp.join(data_root, ty)
        img_list_ = os.listdir(img_dir)
        for im in img_list_:
            im_name = osp.join(ty, im)
            img_list.append(im_name)
            label_list.append(types_new[ty])
            #print(im_name,type_[ty])
    print('Total number of images: {}'.format(len(img_list))) 
    np.random.set_state(state)       
    np.random.shuffle(img_list)
    np.random.set_state(state)
    np.random.shuffle(label_list)   
    
    train_n = int(len(img_list)*0.70)
    val_n = int(len(img_list)*0.10)       
    trainx = img_list[:train_n]        
    trainy = label_list[:train_n]
    valx = img_list[train_n:train_n+val_n]        
    valy = label_list[train_n:train_n+val_n]
    testx = img_list[train_n+val_n:]        
    testy = label_list[train_n+val_n:]        
    assert len(trainx) + len(valx) + len(testx) == len(img_list)
    assert len(img_list) == len(label_list)
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
        client_trainset=HyperKvasir(data_root, subtrain_data_known_index, 'train', train_images, train_labels, param)
        client_trainloader=torch.utils.data.DataLoader(client_trainset, batch_size=batchsize, shuffle=True, num_workers=num_workers)
        trainloaders.append(client_trainloader)
        
        client_trainset=HyperKvasir(data_root, subtrain_data_known_index, 'train_val', train_images, train_labels, param)
        client_train_val_loader=torch.utils.data.DataLoader(client_trainset, batch_size=1, shuffle=False, num_workers=num_workers)
        train_val_loaders.append(client_train_val_loader)        

    # val dataloader
    valset=HyperKvasir(data_root,val_data_known_index,'valclose', valx, valy, param)
    valloader=torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False, num_workers=num_workers)

    closeset=HyperKvasir(data_root,test_data_known_index,'testclose', testx, testy, param=param)
    closeloader=torch.utils.data.DataLoader(closeset, batch_size=1, shuffle=False, num_workers=num_workers)

    openset=HyperKvasir(data_root,test_data_unknown_index,'testopen',testx, testy, param=param)
    openloader=torch.utils.data.DataLoader(openset, batch_size=1, shuffle=False, num_workers=num_workers) 
    
    return trainloaders, valloader, closeloader, openloader, train_val_loaders

class HyperKvasir(Dataset):
    def __init__(self, data_root, data_index, setname, datax, datay, param = {'Known_class': 6, 'unKnown_class': 9, 'Rotation': 45, 'Resize': 144, 'CropSize':128}):
        self.data_root = data_root 
        self.data_index = data_index
        self.setname = setname
        self.datax = datax
        self.datay = datay

        self.transform_train = transforms.Compose([
            transforms.RandomAffine(degrees=param['Rotation'], shear=5.729578),    
            transforms.Resize((param['Resize'],param['Resize'])),    
            transforms.RandomCrop((param['CropSize'],param['CropSize'])),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])    
        self.transform_test = transforms.Compose([
            transforms.Resize((param['CropSize'], param['CropSize'])),
            transforms.ToTensor(),                
        ])
    
    def __len__(self):
        return len(self.data_index)

    def __getitem__(self, index):
        img_id = self.data_index[index]
        img_dir = self.datax[img_id]            
        label = self.datay[img_id]
            
        img = Image.open(os.path.join(self.data_root, img_dir))
        image = None
        if self.setname=='train':
            image = self.transform_train(img)
        else:
            image = self.transform_test(img)
        
        return image, label, img_dir




if __name__ == '__main__':
    datadir = './dataset/'
    from collections import Counter
    trainloaders, valloader, closeloader, openloader,train_val_loaders = get_dataloaders(8, datadir, 0)
    
    for c, trainloader in enumerate(trainloaders):
        print(trainloader.dataset.__len__())
        labels = []
        for i, data_ in enumerate(trainloader):
            img, label, img_dir = data_
            labels += label.data.tolist()
        d = Counter(labels)
        d_s = sorted(d.items(),key=lambda x:x[1],reverse=True)
        print('The number of class each',d_s)              
            
        '''
        img = img[0].numpy()
        print(img.shape)
        import matplotlib.pyplot as plt
        img = img.swapaxes(0,1)
        img = img.swapaxes(1,2)
        plt.imshow(img)
        plt.show()
        print(c, i,img_dir)
        if i > 1:
            break
        '''
            
