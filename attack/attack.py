# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 15:01:20 2022

@author: meiluzhu
"""
import torch
import torch.nn.functional as F
from .helpers import l2_norm
from torch.autograd import Variable


class Attack:
    def __init__(self, known_class=None, eps=5.0 * 16 / 255.0, num_steps=10, x_val_min= -2.5, x_val_max=2.5):

        self.eps = eps
        self.num_steps = num_steps
        self.criterion = F.cross_entropy
        self.x_val_min = x_val_min
        self.x_val_max = x_val_max
        self.known_class = known_class
        
  
    def DUS(self,model, inputs, targets, eps=0.03):
        x_adv = Variable(inputs.data, requires_grad=True)
        model.eval()
        h_adv = model(x_adv)

        cost = -self.criterion(h_adv, targets)

        self.net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        x_adv = x_adv - eps*x_adv.grad
        x_adv = torch.clamp(x_adv, self.x_val_min, self.x_val_max)


        h_adv = self.net(x_adv)
        model.train()
        
        return x_adv, h_adv

    def i_DUS(self, model, inputs, targets, net_peers_sample = None):
        
        x_adv = inputs.clone().detach().data
        model.eval()    
        for i in range(self.num_steps):
            x_adv = Variable(x_adv.data, requires_grad=True)
            outs = model.discrete_forward(x_adv) 
            h_adv = outs['outputs']
            cost = -self.criterion(h_adv, targets)
            model.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            cost.backward()       
            x_adv = x_adv - self.eps*x_adv.grad.data          
        inputs_unknown = x_adv.data
        targets_unknown =  targets.data       
        model.train()

        return inputs_unknown, targets_unknown     